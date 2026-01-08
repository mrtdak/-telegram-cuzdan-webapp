"""
QuantumTree - Akilli Guvenlik Kamerasi
EV ve KUYUMCU modlari

Kullanim:
  python guvenlik_kamera.py --mod ev
  python guvenlik_kamera.py --mod kuyumcu

Ozellikler:
  EV MODU: Hareket algilama + AI analiz + bildirim
  KUYUMCU MODU: Tehlike tespiti (silah/bicak/maske) + ALARM

Gelistirici: QuantumTree
Versiyon: 1.0
"""
import cv2
import base64
import json
import os
import time
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============== AYARLAR ==============
BILDIRIM_DOSYASI = "C:/Projects/quantumtree/kamera_bildirim.json"
KAYIT_KLASORU = "C:/Projects/quantumtree/kamera_kayitlar"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ev modu ayarlari
EV_HAREKET_ESIK = 25
EV_MIN_ALAN = 3000
EV_BEKLEME = 30  # saniye

# Kuyumcu modu ayarlari
KUYUMCU_ANALIZ_SURESI = 10  # her 10 saniyede bir analiz (test icin)

KAMERA_ID = 0

os.makedirs(KAYIT_KLASORU, exist_ok=True)

# ============== PROMPTLAR ==============
EV_PROMPT = """Bu ev guvenlik kamerasi goruntusu. Analiz et:

1. Kim/ne var? (kisi, hayvan, arac)
2. Ne yapiyor? (yuruyor, duruyor, bakiniyor)
3. Supheli durum var mi?

Kisa ve net Turkce cevap ver.
Ornek: "Bahcede bir kisi kapiya dogru yuruyor."
"""

KUYUMCU_PROMPT = """Bu kuyumcu guvenlik kamerasi. SADECE TEHLIKE KONTROLU YAP:

Kontrol et:
- Silah var mi? (tabanca, tufek)
- Bicak var mi?
- Maskeli/yuzu kapali kisi var mi?
- Saldiri veya rehin durumu var mi? (eller havada, yerde yatan)
- Tezgaha atlama veya saldirgan hareket var mi?

ONEMLI:
- Tehlike YOKSA sadece "OK" yaz, baska bir sey yazma
- Tehlike VARSA detayli acikla (ne, nerede, kim)

Ornek tehlike: "TEHLIKE: Maskeli bir kisi tezgaha yaklasıyor, elinde metalik cisim var."
"""


def ai_analiz(foto_path, prompt):
    """OpenRouter/Gemini ile goruntu analizi"""
    try:
        with open(foto_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'google/gemini-2.0-flash-001',
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:image/jpeg;base64,{img_base64}'}
                            }
                        ]
                    }
                ],
                'max_tokens': 200
            },
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            print(f"  API Hatasi: {response.status_code}")
            return None
    except Exception as e:
        print(f"  AI hatasi: {e}")
        return None


def bildirim_gonder(foto_path, ai_sonuc, mod, tehlike=False):
    """QuantumTree'ye bildirim gonder"""
    bildirim = {
        "timestamp": datetime.now().isoformat(),
        "foto_path": foto_path,
        "ai_analiz": ai_sonuc,
        "okundu": False,
        "mod": mod,
        "tehlike": tehlike
    }

    with open(BILDIRIM_DOSYASI, 'w', encoding='utf-8') as f:
        json.dump(bildirim, f, ensure_ascii=False, indent=2)

    if tehlike:
        print(f"  [!!!ALARM!!!] QuantumTree'ye gonderildi!")
    else:
        print(f"  [BILDIRIM] QuantumTree'ye gonderildi!")


def hareket_algilama(frame, onceki_frame, esik, min_alan):
    """OpenCV ile hareket algilama"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if onceki_frame is None:
        return gray, False

    frame_delta = cv2.absdiff(onceki_frame, gray)
    thresh = cv2.threshold(frame_delta, esik, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) >= min_alan:
            return gray, True

    return gray, False


def ev_modu():
    """EV MODU: Hareket algilandiginda bildirim"""
    print("=" * 60)
    print("    EV GUVENLIK MODU")
    print("    Hareket algilandiginda analiz ve bildirim")
    print("=" * 60)
    print()
    print(f"Ayarlar:")
    print(f"  - Hareket esigi: {EV_HAREKET_ESIK}")
    print(f"  - Minimum alan: {EV_MIN_ALAN} piksel")
    print(f"  - Bekleme suresi: {EV_BEKLEME} saniye")
    print()
    print("Cikmak icin Ctrl+C basin")
    print("-" * 60)
    print()

    cap = cv2.VideoCapture(KAMERA_ID)
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[KAMERA] Baslatildi, izleniyor...")
    print()

    onceki_frame = None
    son_bildirim = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            onceki_frame, hareket = hareket_algilama(
                frame, onceki_frame, EV_HAREKET_ESIK, EV_MIN_ALAN
            )

            if hareket:
                simdi = time.time()
                if simdi - son_bildirim >= EV_BEKLEME:
                    son_bildirim = simdi

                    tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"[{tarih}] HAREKET ALGILANDI!")

                    foto_path = f"{KAYIT_KLASORU}/ev_{tarih}.jpg"
                    cv2.imwrite(foto_path, frame)
                    print(f"  Fotograf: {foto_path}")

                    print("  AI analiz yapiliyor...")
                    ai_sonuc = ai_analiz(foto_path, EV_PROMPT)

                    if ai_sonuc:
                        print(f"  AI: {ai_sonuc}")
                        bildirim_gonder(foto_path, ai_sonuc, "ev")
                    print()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nIzleme durduruldu.")
    finally:
        cap.release()


def kuyumcu_modu():
    """KUYUMCU MODU: Periyodik analiz, sadece tehlikede bildirim"""
    print("=" * 60)
    print("    KUYUMCU GUVENLIK MODU")
    print("    Tehlike algilandiginda ALARM")
    print("=" * 60)
    print()
    print(f"Ayarlar:")
    print(f"  - Analiz araligi: {KUYUMCU_ANALIZ_SURESI} saniye")
    print(f"  - Tehlike tespiti: Silah, bicak, maske, saldiri")
    print()
    print("Cikmak icin Ctrl+C basin")
    print("-" * 60)
    print()

    cap = cv2.VideoCapture(KAMERA_ID)
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[KAMERA] Baslatildi, izleniyor...")
    print("[MOD] Sadece TEHLIKE durumunda bildirim gonderilecek")
    print()

    son_analiz = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            simdi = time.time()

            # Her X saniyede bir analiz
            if simdi - son_analiz >= KUYUMCU_ANALIZ_SURESI:
                son_analiz = simdi

                tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"[{tarih}] Periyodik kontrol...")

                foto_path = f"{KAYIT_KLASORU}/kuyumcu_{tarih}.jpg"
                cv2.imwrite(foto_path, frame)

                ai_sonuc = ai_analiz(foto_path, KUYUMCU_PROMPT)

                if ai_sonuc:
                    # Tehlike kontrolu
                    if ai_sonuc.strip().upper() == "OK":
                        print(f"  Durum: Normal (tehlike yok)")
                        # Fotografi sil (gereksiz)
                        os.remove(foto_path)
                    else:
                        # TEHLIKE VAR!
                        print(f"  !!! TEHLIKE ALGILANDI !!!")
                        print(f"  AI: {ai_sonuc}")
                        bildirim_gonder(foto_path, ai_sonuc, "kuyumcu", tehlike=True)
                        print()
                else:
                    print(f"  AI analiz basarisiz")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nIzleme durduruldu.")
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description='QuantumTree Guvenlik Kamerasi')
    parser.add_argument('--mod', type=str, choices=['ev', 'kuyumcu'],
                        default='ev', help='Calisma modu: ev veya kuyumcu')
    args = parser.parse_args()

    if args.mod == 'ev':
        ev_modu()
    elif args.mod == 'kuyumcu':
        kuyumcu_modu()


if __name__ == "__main__":
    main()
