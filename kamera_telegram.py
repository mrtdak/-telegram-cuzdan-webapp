"""
Kamera + YOLO + LLM + Telegram Entegrasyonu
Önce sen test et, sonra çoklu kullanıcı

Kullanım:
  python kamera_telegram.py --kamera 0        # Yerel webcam
  python kamera_telegram.py --kamera rtsp://ip:port/stream  # IP kamera
"""
import cv2
import base64
import os
import time
import asyncio
import requests
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ============== AYARLAR ==============
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Senin chat ID'n
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

KAYIT_KLASORU = "kamera_kayitlar"
os.makedirs(KAYIT_KLASORU, exist_ok=True)

# YOLO ayarları
YOLO_MODEL = "yolov8n.pt"
INSAN_SINIF_ID = 0  # YOLO'da 0 = person
YOLO_GUVEN_ESIK = 0.5  # %50 güven eşiği

# Bildirim ayarları
BILDIRIM_BEKLEME = 30  # Aynı tespit için 30 saniye bekle

# ============== YOLO ==============
print("🔄 YOLO modeli yükleniyor...")
yolo_model = YOLO(YOLO_MODEL)
print("✅ YOLO hazır!")


def yolo_insan_tespit(frame):
    """YOLO ile insan tespiti yap"""
    results = yolo_model(frame, verbose=False)

    insanlar = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            sinif_id = int(box.cls[0])
            guven = float(box.conf[0])

            if sinif_id == INSAN_SINIF_ID and guven >= YOLO_GUVEN_ESIK:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                insanlar.append({
                    "bbox": (x1, y1, x2, y2),
                    "guven": guven
                })
                # Çerçeve çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Insan {guven:.0%}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return insanlar, frame


# ============== LLM DOĞRULAMA ==============
def llm_dogrula(foto_path):
    """LLM ile insan tespitini doğrula"""
    prompt = """Bu güvenlik kamerası görüntüsünde insan var mı?

SADECE şu formatta cevap ver:
- Insan varsa: "EVET: [kısa açıklama]"
- Insan yoksa: "HAYIR"

Örnek: "EVET: Bahçede bir kişi yürüyor"
"""

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
                'max_tokens': 100
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            cevap = data['choices'][0]['message']['content'].strip()
            return cevap
        else:
            print(f"  ❌ LLM API hatası: {response.status_code}")
            return None

    except Exception as e:
        print(f"  ❌ LLM hatası: {e}")
        return None


# ============== TELEGRAM BİLDİRİM ==============
def telegram_bildirim(foto_path, mesaj):
    """Telegram'a fotoğraf + mesaj gönder"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠️ TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID ayarlanmamış!")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

        with open(foto_path, 'rb') as foto:
            response = requests.post(
                url,
                data={
                    'chat_id': TELEGRAM_CHAT_ID,
                    'caption': mesaj
                },
                files={'photo': foto},
                timeout=30
            )

        if response.status_code == 200:
            print("  ✅ Telegram bildirimi gönderildi!")
            return True
        else:
            print(f"  ❌ Telegram hatası: {response.status_code}")
            return False

    except Exception as e:
        print(f"  ❌ Telegram hatası: {e}")
        return False


# ============== ANA İZLEME ==============
def kamera_izle(kamera_kaynak=0):
    """Kamerayı izle, insan tespitinde bildirim gönder"""
    print("=" * 60)
    print("  🎥 KAMERA GÖZETLEMESİ")
    print("  YOLO + LLM + Telegram")
    print("=" * 60)
    print()
    print(f"  Kamera: {kamera_kaynak}")
    print(f"  YOLO güven eşiği: {YOLO_GUVEN_ESIK:.0%}")
    print(f"  Bildirim bekleme: {BILDIRIM_BEKLEME} saniye")
    print()
    print("  Çıkmak için 'q' tuşuna bas")
    print("-" * 60)
    print()

    # Kamera aç
    cap = cv2.VideoCapture(kamera_kaynak)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ Kamera başlatıldı, izleniyor...")
    print()

    son_bildirim = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO ile insan tespiti
            insanlar, frame_isaretli = yolo_insan_tespit(frame)

            if insanlar:
                simdi = time.time()

                # Bekleme süresi geçti mi?
                if simdi - son_bildirim >= BILDIRIM_BEKLEME:
                    son_bildirim = simdi

                    tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"[{tarih}] 🚶 YOLO: {len(insanlar)} insan tespit etti!")

                    # Fotoğraf kaydet
                    foto_path = f"{KAYIT_KLASORU}/tespit_{tarih}.jpg"
                    cv2.imwrite(foto_path, frame_isaretli)
                    print(f"  📸 Kaydedildi: {foto_path}")

                    # LLM doğrulama
                    print("  🤖 LLM doğruluyor...")
                    llm_cevap = llm_dogrula(foto_path)

                    if llm_cevap:
                        print(f"  💬 LLM: {llm_cevap}")

                        if llm_cevap.upper().startswith("EVET"):
                            # İnsan onaylandı - Telegram bildirimi
                            mesaj = f"🚨 İNSAN ALGILANDI!\n\n"
                            mesaj += f"📍 Zaman: {datetime.now().strftime('%H:%M:%S')}\n"
                            mesaj += f"🤖 AI: {llm_cevap}\n"
                            mesaj += f"🎯 YOLO güven: {insanlar[0]['guven']:.0%}"

                            telegram_bildirim(foto_path, mesaj)
                        else:
                            print("  ℹ️ LLM: İnsan yok, bildirim atlanıyor")
                            # Yanlış tespit, fotoğrafı sil
                            os.remove(foto_path)
                    else:
                        print("  ⚠️ LLM yanıt vermedi, bildirim atlanıyor")

                    print()

            # Tarih/saat ekle
            tarih_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_isaretli, tarih_str, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Göster
            cv2.imshow("Kamera Gozetleme", frame_isaretli)

            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️ Durduruldu")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Kamera + YOLO + Telegram')
    parser.add_argument('--kamera', type=str, default='0',
                        help='Kamera kaynağı: 0 (webcam) veya rtsp://ip:port/stream')
    args = parser.parse_args()

    # Kamera kaynağını belirle
    kamera = int(args.kamera) if args.kamera.isdigit() else args.kamera

    kamera_izle(kamera)


if __name__ == "__main__":
    main()
