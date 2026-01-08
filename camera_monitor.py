"""
QuantumTree - Kamera Gözetleme Modülü
Hareket algılama + AI analiz
"""
import cv2
import numpy as np
import threading
import time
import os
import asyncio
import aiohttp
from datetime import datetime
from PIL import Image
import io
import base64
from dotenv import load_dotenv

load_dotenv()

class KameraGozetleme:
    def __init__(self, kamera_id=0):
        self.kamera_id = kamera_id
        self.cap = None
        self.running = False
        self.hareket_algilama = True
        self.hareket_esik = 25  # Hareket hassasiyeti (düşük = daha hassas)
        self.min_alan = 500  # Minimum hareket alanı (piksel)

        # Önceki frame (hareket karşılaştırma için)
        self.onceki_frame = None

        # Kayıt klasörü
        self.kayit_klasoru = "C:/Projects/quantumtree/kamera_kayitlar"
        os.makedirs(self.kayit_klasoru, exist_ok=True)

        # Callback fonksiyonları
        self.hareket_callback = None
        self.frame_callback = None

    def baslat(self):
        """Kamerayı başlat"""
        self.cap = cv2.VideoCapture(self.kamera_id)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            return False

        # Kamera ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        print(f"Kamera başlatıldı (ID: {self.kamera_id})")
        return True

    def durdur(self):
        """Kamerayı durdur"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("Kamera durduruldu")

    def frame_al(self):
        """Tek frame al"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def hareket_tespit(self, frame):
        """Hareket tespiti yap"""
        if frame is None:
            return False, None, []

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # İlk frame ise kaydet
        if self.onceki_frame is None:
            self.onceki_frame = gray
            return False, frame, []

        # Fark hesapla
        frame_delta = cv2.absdiff(self.onceki_frame, gray)
        thresh = cv2.threshold(frame_delta, self.hareket_esik, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Kontur bul
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hareket_var = False
        hareket_bolgeleri = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_alan:
                continue

            hareket_var = True
            (x, y, w, h) = cv2.boundingRect(contour)
            hareket_bolgeleri.append((x, y, w, h))

            # Hareket bölgesini çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Frame'i güncelle
        self.onceki_frame = gray

        return hareket_var, frame, hareket_bolgeleri

    def frame_kaydet(self, frame, sebep="hareket"):
        """Frame'i dosyaya kaydet"""
        if frame is None:
            return None

        tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
        dosya_adi = f"{self.kayit_klasoru}/{sebep}_{tarih}.jpg"
        cv2.imwrite(dosya_adi, frame)
        print(f"Kaydedildi: {dosya_adi}")
        return dosya_adi

    def frame_to_base64(self, frame):
        """Frame'i base64'e çevir (AI için)"""
        if frame is None:
            return None

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Base64'e çevir
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    async def ai_analiz(self, frame, soru="Bu görüntüde ne görüyorsun? Türkçe açıkla."):
        """AI ile görüntü analizi yap (Ollama moondream)"""
        if frame is None:
            return "Frame alınamadı"

        # Frame'i base64'e çevir
        img_base64 = self.frame_to_base64(frame)

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "moondream",
                    "prompt": soru,
                    "images": [img_base64],
                    "stream": False
                }

                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "Yanıt alınamadı")
                    else:
                        error = await response.text()
                        return f"Ollama hatası: {response.status}"

        except Exception as e:
            return f"Analiz hatası: {str(e)}"

    def ai_analiz_sync(self, frame, soru="What do you see in this image?"):
        """Senkron AI analiz (thread içinden çağırmak için)"""
        print("      [DEBUG] ai_analiz_sync başladı...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.ai_analiz(frame, soru))
            print(f"      [DEBUG] Sonuç alındı: {len(result)} karakter")
            return result
        except Exception as e:
            print(f"      [DEBUG] HATA: {e}")
            return f"Hata: {str(e)}"
        finally:
            loop.close()

    def canli_izle(self, pencere_adi="QuantumTree Kamera"):
        """Canlı izleme penceresi aç"""
        if not self.baslat():
            return

        print("Canlı izleme başladı. Çıkmak için 'q' tuşuna basın.")
        print("'s' = Screenshot kaydet, 'h' = Hareket algılama aç/kapa")
        print("'a' = AI analiz (görüntüyü AI'a sor)")

        while self.running:
            frame = self.frame_al()
            if frame is None:
                continue

            # Hareket tespiti
            if self.hareket_algilama:
                hareket, frame, bolgeler = self.hareket_tespit(frame)
                if hareket:
                    # Hareket yazısı ekle
                    cv2.putText(frame, "HAREKET ALGILANDI!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Callback varsa çağır
                    if self.hareket_callback:
                        self.hareket_callback(frame, bolgeler)

            # Tarih/saat ekle
            tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, tarih, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Göster
            cv2.imshow(pencere_adi, frame)

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.frame_kaydet(frame, "manuel")
            elif key == ord('h'):
                self.hareket_algilama = not self.hareket_algilama
                durum = "AÇIK" if self.hareket_algilama else "KAPALI"
                print(f"Hareket algılama: {durum}")
            elif key == ord('a'):
                print("AI analiz yapılıyor...")
                # Frame'i kaydet ve analiz et
                self.frame_kaydet(frame, "ai_analiz")
                sonuc = self.ai_analiz_sync(frame)
                print(f"\n{'='*50}")
                print("AI ANALİZ SONUCU:")
                print(f"{'='*50}")
                print(sonuc)
                print(f"{'='*50}\n")

        self.durdur()
        cv2.destroyAllWindows()


def main():
    """Test fonksiyonu"""
    print("=" * 50)
    print("QuantumTree Kamera Gözetleme")
    print("=" * 50)

    kamera = KameraGozetleme(kamera_id=0)

    # Hareket olunca otomatik kaydet
    def hareket_handler(frame, bolgeler):
        kamera.frame_kaydet(frame, "hareket")

    kamera.hareket_callback = hareket_handler
    kamera.canli_izle()


if __name__ == "__main__":
    main()
