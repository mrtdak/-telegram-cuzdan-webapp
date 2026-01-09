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
import winsound
import threading
import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write as write_wav
from gtts import gTTS
import pygame
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
KUYUMCU_ANALIZ_SURESI = 2  # her 2 saniyede bir analiz

# Ses analizi ayarlari
SES_KAYIT_SURESI = 3  # saniye
SES_ORNEKLEME_HIZI = 16000  # 16kHz (Whisper icin ideal)

KAMERA_ID = 0

# Tehlikeli kelime listesi (lokal filtreleme)
TEHLIKELI_KELIMELER = [
    # Yardim cagrisi
    "yardım", "yardim", "imdat", "kurtarın", "kurtarin",
    # Tehdit
    "vuracağım", "vuracagim", "öldüreceğim", "oldurecegim", "seni öldürürüm",
    "gebertir", "kafana sıkarım", "sıkarım", "sikarim",
    # Silah/bicak
    "silah", "tabanca", "bıçak", "bicak", "tüfek", "tufek",
    # Soygun
    "soygun", "parayı ver", "parayi ver", "kasayı aç", "kasayi ac",
    "ellerini kaldır", "yere yat", "kıpırdama", "kipirdama",
    # Kavga
    "döverim", "doverim", "patlatırım", "patlatirim",
    # Genel tehlike
    "bomba", "patlayıcı", "patlayici", "rehin"
]

# Whisper modeli (global - bir kez yukle)
WHISPER_MODEL = None

os.makedirs(KAYIT_KLASORU, exist_ok=True)


# ============== SES ANALİZİ FONKSİYONLARI ==============
def whisper_yukle():
    """Whisper modelini yukle (bir kez)"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("[SES] Whisper modeli yukleniyor (ilk seferde yavas olabilir)...")
        WHISPER_MODEL = whisper.load_model("tiny")  # tiny model - hizli
        print("[SES] Whisper hazir!")
    return WHISPER_MODEL


def ses_kaydet(sure=SES_KAYIT_SURESI):
    """Mikrofondan ses kaydet"""
    try:
        ses = sd.rec(int(sure * SES_ORNEKLEME_HIZI),
                     samplerate=SES_ORNEKLEME_HIZI,
                     channels=1,
                     dtype='float32')
        sd.wait()  # Kayit bitene kadar bekle
        return ses
    except Exception as e:
        print(f"  [SES HATA] Mikrofon hatasi: {e}")
        return None


def ses_yaziya_cevir(ses_data):
    """Whisper ile ses -> yazi"""
    try:
        model = whisper_yukle()

        # Gecici wav dosyasi olustur
        temp_wav = f"{KAYIT_KLASORU}/temp_ses.wav"
        write_wav(temp_wav, SES_ORNEKLEME_HIZI, ses_data)

        # Whisper ile transkript
        result = model.transcribe(temp_wav, language="tr")
        metin = result["text"].strip().lower()

        # Gecici dosyayi sil
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        return metin
    except Exception as e:
        print(f"  [SES HATA] Whisper hatasi: {e}")
        return ""


def tehlikeli_kelime_kontrol(metin):
    """Metinde tehlikeli kelime var mi kontrol et"""
    metin_lower = metin.lower()
    bulunan_kelimeler = []

    for kelime in TEHLIKELI_KELIMELER:
        if kelime in metin_lower:
            bulunan_kelimeler.append(kelime)

    return bulunan_kelimeler


def ses_tehlike_analiz_llm(metin, foto_path=None):
    """LLM ile detayli ses tehlike analizi"""
    prompt = f"""Guvenlik kamerasi ses kaydi analizi.

Duyulan ses: "{metin}"

Bu seste tehlike var mi? Analiz et:
- Yardim cagrisi mi?
- Tehdit mi?
- Soygun/gasap durumu mu?
- Kavga mi?

ONEMLI: Tehlike YOKSA sadece "OK" yaz.
Tehlike VARSA detayli acikla ve onem derecesini belirt (DUSUK/ORTA/YUKSEK/KRITIK)

Ornek: "KRITIK TEHLIKE: Soygun girisimi - 'parayi ver kasayi ac' ifadesi duyuldu."
"""

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'google/gemini-2.0-flash-001',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 200
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            return None
    except Exception as e:
        print(f"  [SES HATA] LLM hatasi: {e}")
        return None


# ============== ALARM FONKSİYONU ==============
def alarm_cal():
    """Tehlike algılandığında alarm sesi + sesli uyarı"""
    def _alarm():
        # Önce alarm sesi
        for _ in range(3):
            winsound.Beep(2500, 200)
            winsound.Beep(1500, 200)
            winsound.Beep(2500, 200)
            winsound.Beep(1500, 200)
            time.sleep(0.3)

        # Sesli uyarı
        mesaj = "Dikkat! Güvenlik ihlali tespit edildi. Tüm güvenlik sistemleri devreye alındı. Teslim olun!"
        alarm_dosya = f"{KAYIT_KLASORU}/alarm_uyari.mp3"
        try:
            tts = gTTS(text=mesaj, lang='tr')
            tts.save(alarm_dosya)

            pygame.mixer.init()
            pygame.mixer.music.load(alarm_dosya)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.quit()

            # Dosyayı sil
            if os.path.exists(alarm_dosya):
                os.remove(alarm_dosya)
        except Exception as e:
            print(f"  [ALARM HATA] {e}")

    # Thread ile çal (bloklamadan)
    threading.Thread(target=_alarm, daemon=True).start()


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
    """KUYUMCU MODU: Goruntu + Ses analizi, tehlikede bildirim"""
    print("=" * 60)
    print("    KUYUMCU GUVENLIK MODU")
    print("    Goruntu + Ses Analizi - Tehlike ALARM")
    print("=" * 60)
    print()
    print(f"Ayarlar:")
    print(f"  - Goruntu analiz araligi: {KUYUMCU_ANALIZ_SURESI} saniye")
    print(f"  - Ses kayit suresi: {SES_KAYIT_SURESI} saniye")
    print(f"  - Tehlike tespiti: Silah, bicak, maske, saldiri")
    print(f"  - Ses tespiti: Tehdit, yardim cagrisi, soygun")
    print()
    print("Cikmak icin Ctrl+C basin")
    print("-" * 60)
    print()

    # Ses analizi devre disi (mikrofon alindiginda aktif edilecek)
    SES_ANALIZI_AKTIF = False

    cap = cv2.VideoCapture(KAMERA_ID)
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[KAMERA] Baslatildi, izleniyor...")
    if SES_ANALIZI_AKTIF:
        print("[SES] Mikrofon dinleniyor...")
    else:
        print("[SES] Devre disi (mikrofon alindiginda aktif edilecek)")
    print("[MOD] Sadece TEHLIKE durumunda bildirim gonderilecek")
    print()

    son_goruntu_analiz = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            simdi = time.time()

            # Her X saniyede bir goruntu analizi
            if simdi - son_goruntu_analiz >= KUYUMCU_ANALIZ_SURESI:
                son_goruntu_analiz = simdi

                tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"[{tarih}] Goruntu kontrol...")

                foto_path = f"{KAYIT_KLASORU}/kuyumcu_{tarih}.jpg"
                cv2.imwrite(foto_path, frame)

                ai_sonuc = ai_analiz(foto_path, KUYUMCU_PROMPT)

                if ai_sonuc:
                    # Tehlike kontrolu
                    if ai_sonuc.strip().upper() == "OK":
                        print(f"  Goruntu: Normal")
                        # Fotografi sil (gereksiz)
                        os.remove(foto_path)
                    else:
                        # TEHLIKE VAR!
                        print(f"  !!! GORUNTU TEHLIKE !!!")
                        print(f"  AI: {ai_sonuc}")
                        alarm_cal()
                        bildirim_gonder(foto_path, ai_sonuc, "kuyumcu", tehlike=True)
                        print()
                else:
                    print(f"  Goruntu analiz basarisiz")

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
