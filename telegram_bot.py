"""
Telegram Bot - Arayüz

Akış:
Telegram → HafizaAsistani.prepare() → PersonalAI.generate() → HafizaAsistani.save() → Telegram
"""

import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update, BotCommand, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup, ForceReply, LabeledPrice
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, PreCheckoutQueryHandler
from telegram.request import HTTPXRequest
from typing import Dict, Tuple, Optional

from hafiza_asistani import HafizaAsistani
from personal_ai import PersonalAI
from belge_asistani import BelgeAsistani
import re
import io
import threading
import json
from db_manager import get_db, PlanType

load_dotenv()

# Admin ID'leri - rate limit yok, tüm özellikler açık
ADMIN_IDS = [6505503887]


def _parse_ozel_sure(metin: str) -> int:
    """
    Özel süre metnini dakikaya çevir.
    Örnekler: "40", "40 dk", "2 saat", "1 saat 30 dk", "90 dakika"
    Returns: dakika (int) veya None
    """
    import re
    metin = metin.lower().strip()

    toplam_dakika = 0

    # Sadece sayı girilmişse dakika olarak kabul et
    if re.match(r'^\d+$', metin):
        return int(metin)

    # Saat pattern: "2 saat", "2 sa", "2s"
    saat_match = re.search(r'(\d+)\s*(?:saat|sa|s)\b', metin)
    if saat_match:
        toplam_dakika += int(saat_match.group(1)) * 60

    # Dakika pattern: "30 dakika", "30 dk", "30d", "30 dak"
    dakika_match = re.search(r'(\d+)\s*(?:dakika|dak|dk|d)\b', metin)
    if dakika_match:
        toplam_dakika += int(dakika_match.group(1))

    # Hiçbiri eşleşmediyse None döndür
    if toplam_dakika == 0:
        # Belki sadece "2" gibi bir şey yazmıştır, dakika olarak al
        sayi_match = re.search(r'(\d+)', metin)
        if sayi_match:
            return int(sayi_match.group(1))
        return None

    return toplam_dakika


# ============== HATIRLATMA SİSTEMİ ==============

async def hatirlatma_gonder(context: ContextTypes.DEFAULT_TYPE):
    """JobQueue tarafından çağrılan hatırlatma gönderme fonksiyonu"""
    job = context.job
    user_id = job.data['user_id']
    not_data = job.data['not']

    mesaj = f"⏰ **HATIRLATMA**\n\n📝 {not_data['icerik']}\n\n_({not_data['tarih']} tarihli not)_"

    try:
        await context.bot.send_message(
            chat_id=user_id,
            text=mesaj,
            parse_mode='Markdown'
        )

        # Hatırlatma gönderildi olarak işaretle
        user = user_instances.get(user_id)
        if user:
            user["hafiza"].not_manager.hatirlatma_gonderildi_isaretle(not_data['id'])
        print(f"[HATIRLATMA] User {user_id} için not #{not_data['id']} gönderildi")
    except Exception as e:
        print(f"[HATA] Hatırlatma gönderilemedi: {e}")


def hatirlatma_job_ekle(application, user_id: int, not_data: dict):
    """Yeni hatırlatma job'ı ekle"""
    hatirlatma_str = not_data.get('hatirlatma')
    if not hatirlatma_str:
        return

    hatirlatma_zamani = datetime.fromisoformat(hatirlatma_str)
    now = datetime.now()

    if hatirlatma_zamani <= now:
        print(f"[HATIRLATMA] Geçmiş zaman, atlanıyor: {hatirlatma_str}")
        return

    gecikme = (hatirlatma_zamani - now).total_seconds()

    # Aynı isimli eski job varsa sil (duplicate önleme)
    job_name = f"hatirlatma_{user_id}_{not_data['id']}"
    existing_jobs = application.job_queue.get_jobs_by_name(job_name)
    for job in existing_jobs:
        job.schedule_removal()
        print(f"[HATIRLATMA] Eski job silindi: {job_name}")

    application.job_queue.run_once(
        hatirlatma_gonder,
        when=gecikme,
        data={'user_id': user_id, 'not': not_data},
        name=job_name
    )
    print(f"[HATIRLATMA] Job eklendi: {job_name}, {gecikme:.0f} saniye sonra")


async def mevcut_hatirlatmalari_yukle(application):
    """Bot başladığında mevcut hatırlatmaları JobQueue'ya ekle"""
    import glob
    import re

    notes_files = glob.glob("user_data/user_*/notes/notlar.json")
    toplam = 0

    for notes_file in notes_files:
        try:
            # user_id'yi path'ten çıkar (Windows ve Linux uyumlu)
            match = re.search(r'user_(\d+)', notes_file)
            if not match:
                continue
            user_id = int(match.group(1))

            with open(notes_file, 'r', encoding='utf-8') as f:
                notes = json.load(f)

            for n in notes:
                if n.get('hatirlatma') and not n.get('hatirlatma_gonderildi', False):
                    hatirlatma_job_ekle(application, user_id, n)
                    toplam += 1
        except Exception as e:
            print(f"[HATA] Hatırlatma yüklenemedi ({notes_file}): {e}")

    print(f"[HATIRLATMA] {toplam} mevcut hatırlatma yüklendi")


# Konum arama kategorileri (inline butonlar için)
KONUM_KATEGORILERI = [
    ("⛽ Benzinlik", "benzinlik"), ("💊 Eczane", "eczane"),
    ("🌙 Nöbetçi Eczane", "nobetci_eczane"), ("⛽ Yakıt Fiyatları", "yakit_fiyat"),
    ("🍽️ Restoran", "restoran"), ("☕ Kafe", "kafe"),
    ("🏧 ATM", "atm"), ("🏥 Hastane", "hastane"),
    ("🕌 Cami", "cami"), ("🛒 Market", "market"),
    ("🅿️ Otopark", "otopark"), ("🏨 Otel", "otel"),
    ("🏬 AVM", "avm"), ("🏫 Okul", "okul"),
]


# ============== KAMERA AĞINI TARAMA (MAC/IP) ==============

def mac_bul_ip_ile(ip: str) -> Optional[str]:
    """IP adresinden MAC adresini bul"""
    import subprocess
    try:
        # Önce ping at (ARP tablosuna eklensin)
        subprocess.run(['ping', '-n', '1', '-w', '1000', ip],
                      capture_output=True, timeout=3)
        # ARP tablosundan MAC'i al
        result = subprocess.run(['arp', '-a', ip],
                               capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if ip in line:
                # MAC adresini bul (xx-xx-xx-xx-xx-xx formatında)
                parts = line.split()
                for part in parts:
                    if len(part) == 17 and part.count('-') == 5:
                        return part.lower()
    except:
        pass
    return None

def ip_bul_mac_ile(mac: str) -> Optional[str]:
    """MAC adresinden IP'yi bul (ağı tarar)"""
    import subprocess
    mac = mac.lower()
    try:
        # ARP tablosunu tara
        result = subprocess.run(['arp', '-a'],
                               capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            line_lower = line.lower()
            if mac in line_lower:
                # IP adresini bul
                parts = line.split()
                for part in parts:
                    if part.count('.') == 3:  # IP formatı
                        return part
    except:
        pass
    return None

def agdaki_kamerayi_bul(mac: str, eski_ip: str) -> Optional[str]:
    """Ağda kamerayı bul - önce eski IP'yi dene, sonra MAC ile ara"""
    import subprocess

    # 1. Önce eski IP'yi dene (hızlı)
    if eski_ip:
        try:
            result = subprocess.run(['ping', '-n', '1', '-w', '1000', eski_ip],
                                   capture_output=True, timeout=3)
            if result.returncode == 0:
                return eski_ip
        except:
            pass

    # 2. MAC ile ara
    yeni_ip = ip_bul_mac_ile(mac)
    if yeni_ip:
        return yeni_ip

    # 3. Ağı tara (192.168.1.1-254 ping at)
    import concurrent.futures

    def ping_ip(ip):
        try:
            result = subprocess.run(['ping', '-n', '1', '-w', '500', ip],
                                   capture_output=True, timeout=2)
            return result.returncode == 0
        except:
            return False

    # Paralel ping
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        ips = [f"192.168.1.{i}" for i in range(1, 255)]
        executor.map(ping_ip, ips)

    # Tekrar MAC ile ara
    return ip_bul_mac_ile(mac)


# ============== KAMERA MANAGER (Multi-User) ==============

class KameraManager:
    """Kullanıcı bazlı kamera ayarları yönetimi"""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.config_dir = f"user_data/user_{user_id}"
        self.config_path = f"{self.config_dir}/kamera_ayarlari.json"
        os.makedirs(self.config_dir, exist_ok=True)

    def yukle(self) -> dict:
        """Kamera ayarlarını yükle"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"kameralar": [], "varsayilan_kamera": None}

    def kaydet(self, config: dict):
        """Kamera ayarlarını kaydet"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def kamera_ekle(self, ad: str, ip: str, port: int, kullanici: str, sifre: str, kanal: int) -> int:
        """Yeni kamera ekle, ID döndür"""
        config = self.yukle()

        # Yeni ID belirle
        mevcut_idler = [k["id"] for k in config["kameralar"]]
        yeni_id = max(mevcut_idler, default=0) + 1

        # MAC adresini bul
        mac = mac_bul_ip_ile(ip)

        yeni_kamera = {
            "id": yeni_id,
            "ad": ad,
            "ip": ip,
            "port": port,
            "kullanici": kullanici,
            "sifre": sifre,
            "kanal": kanal,
            "mac": mac,  # IP değişse bile MAC ile bulunabilir
            "aktif": False
        }

        config["kameralar"].append(yeni_kamera)

        # İlk kamera ise varsayılan yap
        if config["varsayilan_kamera"] is None:
            config["varsayilan_kamera"] = yeni_id

        self.kaydet(config)
        return yeni_id

    def ip_guncelle(self, kamera_id: int, yeni_ip: str) -> bool:
        """Kamera IP'sini güncelle"""
        config = self.yukle()
        for k in config["kameralar"]:
            if k["id"] == kamera_id:
                k["ip"] = yeni_ip
                self.kaydet(config)
                return True
        return False

    def ip_otomatik_bul(self, kamera_id: int) -> Optional[str]:
        """MAC adresi ile yeni IP'yi bul ve güncelle"""
        kamera = self.kamera_getir(kamera_id)
        if not kamera or not kamera.get("mac"):
            return None

        yeni_ip = agdaki_kamerayi_bul(kamera["mac"], kamera["ip"])
        if yeni_ip and yeni_ip != kamera["ip"]:
            self.ip_guncelle(kamera_id, yeni_ip)
            return yeni_ip
        return kamera["ip"] if yeni_ip else None

    def kamera_sil(self, kamera_id: int) -> bool:
        """Kamerayı sil"""
        config = self.yukle()

        for i, k in enumerate(config["kameralar"]):
            if k["id"] == kamera_id:
                config["kameralar"].pop(i)

                # Varsayılan ayarını güncelle
                if config["varsayilan_kamera"] == kamera_id:
                    if config["kameralar"]:
                        config["varsayilan_kamera"] = config["kameralar"][0]["id"]
                    else:
                        config["varsayilan_kamera"] = None

                self.kaydet(config)
                return True

        return False

    def kamera_listele(self) -> list:
        """Tüm kameraları listele"""
        config = self.yukle()
        return config["kameralar"]

    def kamera_getir(self, kamera_id: int) -> Optional[dict]:
        """Belirli bir kamerayı getir"""
        config = self.yukle()
        for k in config["kameralar"]:
            if k["id"] == kamera_id:
                return k
        return None

    def rtsp_url_olustur(self, kamera_id: int) -> Optional[str]:
        """RTSP URL oluştur (Dahua formatı)"""
        kamera = self.kamera_getir(kamera_id)
        if not kamera:
            return None

        # rtsp://kullanici:sifre@ip:port/cam/realmonitor?channel=kanal&subtype=0
        return (
            f"rtsp://{kamera['kullanici']}:{kamera['sifre']}@"
            f"{kamera['ip']}:{kamera['port']}/cam/realmonitor"
            f"?channel={kamera['kanal']}&subtype=0"
        )

    def rtsp_url_maskeli(self, kamera_id: int) -> Optional[str]:
        """Şifre maskeli RTSP URL (gösterim için)"""
        kamera = self.kamera_getir(kamera_id)
        if not kamera:
            return None

        return (
            f"rtsp://{kamera['kullanici']}:***@"
            f"{kamera['ip']}:{kamera['port']}/cam/realmonitor"
            f"?channel={kamera['kanal']}"
        )

    def kamera_durumu_guncelle(self, kamera_id: int, aktif: bool):
        """Kamera aktif durumunu güncelle"""
        config = self.yukle()
        for k in config["kameralar"]:
            if k["id"] == kamera_id:
                k["aktif"] = aktif
                break
        self.kaydet(config)


# Wizard state yönetimi (kullanıcı bazlı)
user_kamera_wizard: Dict[int, Dict] = {}
# {
#   user_id: {
#     "adim": "ad" | "ip" | "port" | "kullanici" | "sifre" | "kanal",
#     "data": { "ad": "...", "ip": "...", ... }
#   }
# }

# Kullanıcı bazlı kamera thread yönetimi (çoklu kamera desteği)
user_kamera_threads: Dict[int, Dict[int, Dict]] = {}
# {
#   user_id: {
#     kamera_id: {
#       "thread": Thread,
#       "aktif": True/False,
#       "stop_flag": True/False
#     },
#     kamera_id_2: { ... }
#   }
# }

# ============== KAMERA SİSTEMİ (Multi-User) ==============

def kamera_izleme_baslat(user_id: int, chat_id: int, kamera_kaynak: str, kamera_id: int, kamera_ad: str):
    """Kamera izlemeyi arka planda başlat (kullanıcı bazlı)"""
    global user_kamera_threads

    import cv2
    import base64
    import requests
    import time
    from datetime import datetime
    from ultralytics import YOLO

    # RTSP TCP transport kullan (UDP yerine)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    KAYIT_KLASORU = f"user_data/user_{user_id}/kamera_kayitlar"
    os.makedirs(KAYIT_KLASORU, exist_ok=True)

    # YOLO
    yolo_model = YOLO("yolov8n.pt")
    INSAN_SINIF_ID = 0
    YOLO_GUVEN_ESIK = 0.5
    BILDIRIM_BEKLEME = 30

    def yolo_insan_tespit(frame):
        results = yolo_model(frame, verbose=False)
        insanlar = []
        for result in results:
            for box in result.boxes:
                sinif_id = int(box.cls[0])
                guven = float(box.conf[0])
                if sinif_id == INSAN_SINIF_ID and guven >= YOLO_GUVEN_ESIK:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    insanlar.append({"bbox": (x1, y1, x2, y2), "guven": guven})
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return insanlar, frame

    def llm_dogrula(foto_path):
        prompt = """Bu güvenlik kamerası görüntüsünde insan var mı?
SADECE: "EVET: [açıklama]" veya "HAYIR" yaz."""
        try:
            with open(foto_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'google/gemini-2.0-flash-001', 'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_base64}'}}
                ]}], 'max_tokens': 100},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
        except:
            pass
        return None

    def telegram_bildirim(foto_path, mesaj):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(foto_path, 'rb') as foto:
                requests.post(url, data={'chat_id': chat_id, 'caption': mesaj}, files={'photo': foto}, timeout=30)
        except:
            pass

    # Kamera aç (TCP transport ile)
    cap = cv2.VideoCapture(kamera_kaynak, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[HATA] Kamera acilamadi: {kamera_kaynak}")
        # Thread durumunu güncelle
        if user_id in user_kamera_threads and kamera_id in user_kamera_threads[user_id]:
            user_kamera_threads[user_id][kamera_id]["aktif"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"📹 Kamera izleme başladı - User: {user_id}, Kamera: {kamera_ad}")
    son_bildirim = 0

    # Thread'in aktif olduğunu işaretle
    if user_id in user_kamera_threads and kamera_id in user_kamera_threads[user_id]:
        user_kamera_threads[user_id][kamera_id]["aktif"] = True

    while True:
        # Durdurma kontrolü
        if user_id not in user_kamera_threads:
            break
        if kamera_id not in user_kamera_threads[user_id]:
            break
        if user_kamera_threads[user_id][kamera_id].get("stop_flag", False):
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        insanlar, frame_isaretli = yolo_insan_tespit(frame)

        if insanlar:
            simdi = time.time()
            if simdi - son_bildirim >= BILDIRIM_BEKLEME:
                son_bildirim = simdi
                tarih = datetime.now().strftime("%Y%m%d_%H%M%S")
                foto_path = f"{KAYIT_KLASORU}/tespit_{kamera_ad}_{tarih}.jpg"
                cv2.imwrite(foto_path, frame_isaretli)

                llm_cevap = llm_dogrula(foto_path)
                if llm_cevap and llm_cevap.upper().startswith("EVET"):
                    mesaj = f"🚨 İNSAN ALGILANDI!\n📷 {kamera_ad}\n📍 {datetime.now().strftime('%H:%M:%S')}\n🤖 {llm_cevap}"
                    telegram_bildirim(foto_path, mesaj)
                    print(f"  📤 [{kamera_ad}] Bildirim gönderildi: {llm_cevap}")
                else:
                    try:
                        os.remove(foto_path)
                    except:
                        pass

        time.sleep(0.1)

    cap.release()

    # Kamera durumunu güncelle
    if user_id in user_kamera_threads and kamera_id in user_kamera_threads[user_id]:
        user_kamera_threads[user_id][kamera_id]["aktif"] = False
        kamera_manager = KameraManager(user_id)
        kamera_manager.kamera_durumu_guncelle(kamera_id, False)

    print(f"📹 Kamera izleme durduruldu - User: {user_id}, Kamera: {kamera_ad}")


def kamera_test_baglanti(rtsp_url: str, kaydet_path: str = None) -> Tuple[bool, str, str]:
    """RTSP bağlantısını test et ve fotoğraf çek"""
    try:
        import cv2
        import os
        # RTSP TCP transport kullan (UDP yerine)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Timeout için 5 saniye
        import time
        start = time.time()
        while time.time() - start < 5:
            ret, frame = cap.read()
            if ret and frame is not None:
                foto_path = None
                # Fotoğraf kaydet
                if kaydet_path:
                    cv2.imwrite(kaydet_path, frame)
                    foto_path = kaydet_path
                cap.release()
                return True, "✅ Bağlantı başarılı!", foto_path

        cap.release()
        return False, "❌ Kamera yanıt vermedi.", None
    except Exception as e:
        return False, f"❌ Bağlantı hatası: {str(e)[:50]}", None


def temizle_cikti(text: str) -> str:
    """Yasak ifadeleri ve markdown formatlamalarını temizle"""

    # 1. Markdown temizle
    # **kalın** → kalın
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # *italik* → italik (tek yıldız, ama madde işareti değil)
    text = re.sub(r'(?<!\n)\*([^\*\n]+?)\*(?!\*)', r'\1', text)
    # Satır başı madde işaretleri: * veya -
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    # Numaralı liste: 1. 2. 3.
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # 2. Yasak ifadeleri temizle
    yasak_pattern = r',?\s*(ne dersin\??|değil mi\??|kim bilir\??|nasıl fikir\??|sence\??|vay canına\!?)\s*$'
    cumle_sonu = r'([.!?])\s*'
    cumleler = re.split(cumle_sonu, text)

    temiz = []
    for parca in cumleler:
        if parca in '.!?':
            temiz.append(parca)
            continue
        temiz_cumle = re.sub(yasak_pattern, '', parca, flags=re.IGNORECASE)
        temiz.append(temiz_cumle)

    sonuc = ''.join(temiz).strip()

    # 3. Çoklu boş satırları tek satıra indir
    sonuc = re.sub(r'\n{3,}', '\n\n', sonuc)

    if sonuc and sonuc[-1] not in '.!?':
        sonuc += '.'
    return sonuc

# ============================================================
# 📍 KONUM HİZMETLERİ
# ============================================================

async def adres_cozumle(lat: float, lon: float) -> Optional[str]:
    """
    Koordinattan adres çözümle (Reverse Geocoding - Nominatim).
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "accept-language": "tr"
        }
        headers = {"User-Agent": "PersonalAI-TelegramBot/1.0"}

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("display_name", "Adres bulunamadı")
    except Exception as e:
        print(f"Adres çözümleme hatası: {e}")
    return None


# Kullanıcı izolasyonu: Her kullanıcının kendi AI'ı
user_instances: Dict[int, Dict] = {}
TIMEOUT = 120


def get_user_ai(user_id: int) -> Dict:
    """Kullanıcı için HafizaAsistani + PersonalAI + BelgeAsistani al (izole)"""
    if user_id not in user_instances:
        user_str = str(user_id)  # Sadece ID (user_ prefix'i HafizaAsistani'da ekleniyor)

        # HafizaAsistani - prompt hazırlar, hafıza tutar
        hafiza = HafizaAsistani(user_id=user_str)

        # PersonalAI - cevap üretir
        ai = PersonalAI(user_id=user_str)

        # BelgeAsistani - belge yükleme ve arama (lazy init)
        belge = None  # İlk belge yüklemesinde oluşturulacak

        user_instances[user_id] = {
            "hafiza": hafiza,
            "ai": ai,
            "belge": belge
        }
        print(f"🆕 Yeni kullanıcı: {user_id}")

    return user_instances[user_id]


def get_belge_asistani(user_id: int) -> BelgeAsistani:
    """Kullanıcı için BelgeAsistani al (lazy init)"""
    user = get_user_ai(user_id)
    if user["belge"] is None:
        user_str = f"user_{user_id}"
        belge_dir = os.path.join("user_data", user_str, "belgeler")
        user["belge"] = BelgeAsistani(data_dir=belge_dir)
        print(f"📄 Belge asistanı oluşturuldu: {user_id}")
    return user["belge"]


def clear_active_modes(user_id: int, context=None):
    """Menü geçişlerinde tüm aktif modları temizle (çakışma önleme)"""
    global user_kamera_wizard

    if user_id not in user_instances:
        return

    user = user_instances[user_id]

    # 1. Pending not modunu kapat (context.user_data ile)
    if context is not None and hasattr(context, 'user_data'):
        context.user_data["not_bekliyor"] = False
        context.user_data.pop("hatirlatma_ozel_not_id", None)

    # 2. Aktif belgeyi kapat
    if user["belge"] is not None:
        user["belge"].clear_aktif()

    # 3. Kamera wizard'ı kapat
    if user_id in user_kamera_wizard:
        del user_kamera_wizard[user_id]

    print(f"🧹 Aktif modlar temizlendi: {user_id}")


# === KOMUTLAR ===

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start - Herkese açık"""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "Dostum"
    get_user_ai(user_id)

    # Veritabanına kullanıcıyı kaydet
    db = get_db()
    db.get_or_create_user(
        user_id,
        username=update.effective_user.username,
        first_name=update.effective_user.first_name,
        last_name=update.effective_user.last_name
    )

    welcome_text = f"""Merhaba {user_name}! 👋

*Özellikler:*
🤖 *Akıllı Sohbet* - Sorularına cevap, günlük sohbet
📝 *Not Defteri* - Not al + hatırlatma zamanlayıcısı
📄 *Çalışma Alanım* - PDF/DOCX/TXT yükle, soru sor
📍 *Konum Hizmetleri* - Yakındaki eczane, benzinlik, ATM, market
📷 *Güvenlik Kamerası* - İnsan tespiti, fotoğraflı bildirim

*Menü (/ butonu):*
/yeni - Yeni sohbet başlat
/konum - Konum paylaş
/notdefteri - Not Defteri
/belgelerim - Çalışma Alanım
/kameralarim - Kamera yönetimi
/limit - Günlük limitler
/bagis - Projeyi destekle

*Günlük Limitler (Beta):*
💬 30 mesaj | 📍 10 konum | 📷 1 kamera

_Limitler gece 00:00'da sıfırlanır._

Nasıl yardımcı olabilirim?
"""

    await update.message.reply_text(welcome_text, reply_markup=ReplyKeyboardRemove(), parse_mode="Markdown")


async def yeni_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/yeni - Hafızayı sıfırla (herkese açık)"""
    user_id = update.effective_user.id
    user = get_user_ai(user_id)
    user["hafiza"].clear()
    # Komut mesajını sil
    try:
        await update.message.delete()
    except:
        pass
    await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ Sohbet temizlendi!")


async def konum_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/konum - Konum paylaş butonu gönder"""
    chat_id = update.effective_chat.id
    # Komut mesajını sil
    try:
        await update.message.delete()
    except:
        pass
    # Konum paylaşma butonu gönder
    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("📍 Konum Paylaş", request_location=True)]],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await context.bot.send_message(
        chat_id=chat_id,
        text="📍 Aşağıdaki butona bas ve konumunu paylaş:",
        reply_markup=keyboard
    )


async def notdefteri_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/notdefteri - Not Defteri menüsü"""
    buttons = [
        [
            InlineKeyboardButton("📝 Yeni Not", callback_data="not_yeni"),
            InlineKeyboardButton("📋 Notlarım", callback_data="not_listele")
        ]
    ]
    await update.message.reply_text(
        "📝 *Not Defteri*\n\nNe yapmak istersin?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons)
    )


async def belgelerim_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/belgelerim - Çalışma Alanım - Doküman seç ve üzerine konuş"""
    user_id = update.effective_user.id

    # Belge asistanını al (lazy init - mevcut belgeleri yükler)
    belge_asistani = get_belge_asistani(user_id)
    belgeler = belge_asistani.listele()

    if not belgeler:
        await update.message.reply_text(
            "📄 *Çalışma Alanım*\n\n"
            "Henüz doküman yok.\n\n"
            "PDF, DOCX veya TXT dosyası gönder → otomatik yüklenir.",
            parse_mode="Markdown"
        )
        return

    # Doküman listesi - her biri tıklanabilir
    buttons = []
    for b in belgeler:
        buttons.append([InlineKeyboardButton(
            f"📄 {b['dosya_adi']}",
            callback_data=f"belge_gor:{b['belge_id']}"
        )])
    buttons.append([InlineKeyboardButton("🗑️ Doküman Sil", callback_data="belge_sil_menu")])

    await update.message.reply_text(
        f"📄 *Çalışma Alanım*\n\n"
        f"Bir doküman seç, içeriğini gör ve üzerine konuş:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons)
    )


async def limit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/limit - Kullanım limitini göster"""
    user_id = update.effective_user.id

    # Admin sınırsız
    if user_id in ADMIN_IDS:
        await update.message.reply_text(
            "👑 *ADMIN* - Tüm limitler sınırsız!",
            parse_mode="Markdown"
        )
        return

    db = get_db()
    rate_check = db.check_rate_limit(user_id)
    camera_check = db.check_camera_limit(user_id)
    location_check = db.check_location_limit(user_id)
    image_check = db.check_image_limit(user_id)
    usage = db.get_daily_usage(user_id)

    text = f"""📊 *Günlük Kullanım Durumun*

💬 Mesaj: *{rate_check['remaining']}/{rate_check['limit']}*
🎨 Görsel üretimi: *{image_check['remaining']}/{image_check['limit']}*
📷 Kamera bildirimi: *{camera_check['remaining']}/{camera_check['limit']}*
📍 Konum sorgusu: *{location_check['remaining']}/{location_check['limit']}*

📸 Gönderilen fotoğraf: {usage.get('photo_count', 0)}
🔍 Web arama: {usage.get('web_search_count', 0)}

_Limitler gece 00:00'da sıfırlanır._

💝 Projeyi desteklemek için: /bagis
"""

    await update.message.reply_text(text, parse_mode="Markdown")


async def bagis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/bagis - Bağış bilgilerini göster"""
    text = """💝 *Projeyi Destekle*

Bu bot beta aşamasında ve kısıtlı donanımda çalışıyor.
Beğendiysen ve gelişmeye devam etmesini istiyorsan destek olabilirsin.

📊 *Günlük Limitler (Ücretsiz):*
• 30 mesaj
• 5 kamera bildirimi
• 10 konum sorgusu

_Tüm özellikler açık, sadece günlük limit var._

⭐ *Telegram Stars ile Bağış:*
Aşağıdaki butona tıklayarak istediğin kadar Star gönderebilirsin.
"""

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ 10 Stars", callback_data="bagis_10")],
        [InlineKeyboardButton("⭐ 25 Stars", callback_data="bagis_25")],
        [InlineKeyboardButton("⭐ 50 Stars", callback_data="bagis_50")],
    ])

    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)


async def premium_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/premium - Bağış sayfasına yönlendir (eski komut uyumluluğu)"""
    await bagis_command(update, context)


# === KAMERA KOMUTLARI (Multi-User) ===

async def kamera_ekle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/kamera_ekle - Yeni kamera ekleme wizard'ı başlat"""
    global user_kamera_wizard

    user_id = update.effective_user.id

    # Wizard başlat
    user_kamera_wizard[user_id] = {
        "adim": "ad",
        "data": {}
    }

    await update.message.reply_text(
        "Yeni Kamera Ekleme\n\n"
        "Adım 1/6: Kamera adı gir",
        reply_markup=ForceReply(input_field_placeholder="Örn: Bahçe Kamerası")
    )
    await update.message.reply_text(
        "↩️",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ İptal", callback_data="kamera_wizard_iptal")]])
    )


async def kameralarim_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/kameralarim - Kullanıcının kameralarını listele"""
    user_id = update.effective_user.id

    kamera_manager = KameraManager(user_id)
    kameralar = kamera_manager.kamera_listele()

    if not kameralar:
        keyboard = [[InlineKeyboardButton("➕ Kamera Ekle", callback_data="kamera_ekle_wizard")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "📷 Henüz kamera eklememişsin.\n\n"
            "Menüden kamera ekleyebilirsin.",
            reply_markup=reply_markup
        )
        return

    # Aktif kameraları bul (çoklu kamera desteği)
    aktif_kamera_idleri = set()
    if user_id in user_kamera_threads:
        for kid, kdata in user_kamera_threads[user_id].items():
            if kdata.get("aktif"):
                aktif_kamera_idleri.add(kid)

    aktif_sayisi = len(aktif_kamera_idleri)
    mesaj = f"📷 Kameralarım ({len(kameralar)} adet)"
    if aktif_sayisi > 0:
        mesaj += f" - 🟢 {aktif_sayisi} aktif"
    mesaj += "\n\n"

    keyboard = []
    for k in kameralar:
        kamera_aktif = k["id"] in aktif_kamera_idleri
        durum = "🟢 AKTİF" if kamera_aktif else "⚫"
        mesaj += f"{k['id']}. {k['ad']} - {k['ip']}:{k['kanal']} {durum}\n"

        if kamera_aktif:
            # Aktif kamera için durdur butonu
            keyboard.append([InlineKeyboardButton(
                f"⏹️ {k['ad']} Durdur",
                callback_data=f"kamera_durdur:{k['id']}"
            )])
        else:
            # İnaktif kamera için başlat, test ve sil butonları
            keyboard.append([
                InlineKeyboardButton(f"▶️ Başlat", callback_data=f"kamera_baslat:{k['id']}"),
                InlineKeyboardButton(f"🔍 Test", callback_data=f"kamera_test:{k['id']}"),
                InlineKeyboardButton(f"🗑️ Sil", callback_data=f"kamera_sil:{k['id']}")
            ])

    # Tümünü Başlat / Tümünü Durdur butonları
    if len(kameralar) > 1:
        if aktif_sayisi < len(kameralar):
            keyboard.append([InlineKeyboardButton("▶️ Tümünü Başlat", callback_data="kamera_tumunu_baslat")])
        if aktif_sayisi > 0:
            keyboard.append([InlineKeyboardButton("⏹️ Tümünü Durdur", callback_data="kamera_tumunu_durdur")])

    # Yeni kamera ekle butonu
    keyboard.append([InlineKeyboardButton("➕ Yeni Kamera Ekle", callback_data="kamera_ekle_wizard")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(mesaj, reply_markup=reply_markup)


async def kamera_baslat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/kamera [id] - Kamera izlemeyi başlat"""
    global user_kamera_threads

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Argüman kontrolü
    args = context.args
    kamera_manager = KameraManager(user_id)
    kameralar = kamera_manager.kamera_listele()

    if not kameralar:
        await update.message.reply_text(
            "📷 Henüz kamera eklememişsin.\n"
            "Menüden kamera ekleyebilirsin."
        )
        return

    # ID belirtilmemişse listeyi göster
    if not args:
        await kameralarim_command(update, context)
        return

    try:
        kamera_id = int(args[0])
    except:
        await update.message.reply_text("⚠️ Geçersiz kamera ID.")
        return

    # Kamera kontrolü
    kamera = kamera_manager.kamera_getir(kamera_id)
    if not kamera:
        await update.message.reply_text(f"⚠️ Kamera #{kamera_id} bulunamadı.")
        return

    # Zaten aktif mi?
    if user_id in user_kamera_threads and user_kamera_threads[user_id].get("aktif"):
        aktif_id = user_kamera_threads[user_id].get("kamera_id")
        if aktif_id == kamera_id:
            await update.message.reply_text(f"⚠️ {kamera['ad']} zaten aktif!")
            return
        else:
            keyboard = [[InlineKeyboardButton("⏹️ Durdur", callback_data=f"kamera_durdur:{aktif_id}")]]
            await update.message.reply_text(
                f"⚠️ Başka bir kamera aktif (#{aktif_id}).\n"
                "Önce onu durdur.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

    # RTSP URL oluştur
    rtsp_url = kamera_manager.rtsp_url_olustur(kamera_id)

    # Thread başlat
    user_kamera_threads[user_id] = {
        "thread": None,
        "aktif": False,
        "kamera_id": kamera_id,
        "stop_flag": False
    }

    thread = threading.Thread(
        target=kamera_izleme_baslat,
        args=(user_id, chat_id, rtsp_url, kamera_id, kamera["ad"]),
        daemon=True
    )
    user_kamera_threads[user_id]["thread"] = thread
    thread.start()

    # Kamera durumunu güncelle
    kamera_manager.kamera_durumu_guncelle(kamera_id, True)

    keyboard = [[InlineKeyboardButton("⏹️ Durdur", callback_data=f"kamera_durdur:{kamera_id}")]]
    await update.message.reply_text(
        f"📹 {kamera['ad']} başlatıldı!\n\n"
        f"🔗 {kamera['ip']}:{kamera['port']} (Kanal {kamera['kanal']})\n\n"
        "Hareket algılandığında bildirim alacaksın.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def kamera_durdur_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/kamerakapat - Aktif kamerayı durdur"""
    global user_kamera_threads

    user_id = update.effective_user.id

    # Aktif kamera kontrolü
    if user_id not in user_kamera_threads or not user_kamera_threads[user_id].get("aktif"):
        await update.message.reply_text("⚠️ Aktif kamera yok!")
        return

    # Durdurma flag'i ayarla
    user_kamera_threads[user_id]["stop_flag"] = True

    kamera_id = user_kamera_threads[user_id].get("kamera_id")
    kamera_manager = KameraManager(user_id)
    kamera = kamera_manager.kamera_getir(kamera_id)
    kamera_ad = kamera["ad"] if kamera else f"#{kamera_id}"

    await update.message.reply_text(f"⏹️ {kamera_ad} durduruluyor...")


# === KONUM HANDLER ===

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📍 Konum mesajı handler - INLINE BUTONLU"""
    try:
        user_id = update.effective_user.id
        location = update.message.location

        if location is None:
            await update.message.reply_text("❌ Konum bilgisi alınamadı.")
            return

        lat = location.latitude
        lon = location.longitude
        print(f"[KONUM] Alinan: {lat:.4f}, {lon:.4f}")

        user = get_user_ai(user_id)

        # Adres çözümle
        try:
            adres = await adres_cozumle(lat, lon)
            if not adres:
                adres = f"{lat:.4f}, {lon:.4f}"
        except:
            adres = f"{lat:.4f}, {lon:.4f}"

        # Hafıza asistanına konumu kaydet
        asistan = user["hafiza"]
        asistan.set_location(lat, lon, adres)

        # Kısa adres oluştur
        kisa_adres = asistan.konum_adres if hasattr(asistan, 'konum_adres') and asistan.konum_adres else adres[:50]

        # 2'li sıralar halinde inline keyboard oluştur
        keyboard = []
        for i in range(0, len(KONUM_KATEGORILERI), 2):
            row = []
            row.append(InlineKeyboardButton(KONUM_KATEGORILERI[i][0], callback_data=f"konum_ara:{KONUM_KATEGORILERI[i][1]}"))
            if i + 1 < len(KONUM_KATEGORILERI):
                row.append(InlineKeyboardButton(KONUM_KATEGORILERI[i+1][0], callback_data=f"konum_ara:{KONUM_KATEGORILERI[i+1][1]}"))
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Mesaj gönder
        await update.message.reply_text(
            f"📍 {kisa_adres}\n\nNe aramak istiyorsun?",
            reply_markup=reply_markup
        )

    except Exception as e:
        print(f"[HATA] Konum hatasi: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text("❌ Konum işlenirken hata oluştu.")


# === FOTOĞRAF HANDLER ===

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📷 Fotoğraf handler - Ana LLM'e multimodal olarak gider (vision bypass)"""
    try:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # Kullanıcıyı al/oluştur
        user = get_user_ai(user_id)

        # Düşünüyorum mesajı
        status = await context.bot.send_message(chat_id, "🔍 Fotoğrafı inceliyorum...")

        # En yüksek çözünürlüklü fotoğrafı al
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Fotoğrafı indir ve base64'e çevir
        import base64
        photo_bytes = await file.download_as_bytearray()
        img_base64 = base64.b64encode(photo_bytes).decode('utf-8')

        # Caption varsa kullan, yoksa genel fotoğraf mesajı
        caption = update.message.caption or ""
        if caption:
            user_input = f"[Fotoğraf gönderildi] {caption}"
        else:
            user_input = "[Fotoğraf gönderildi]"

        # HafizaAsistani ile prompt hazırla (fotoğraf dahil)
        asistan = user["hafiza"]
        ai = user["ai"]

        # Fırlamama modu kontrolü
        firlama_modu = context.user_data.get("firlama_modu", False)

        # prepare() çağır - image_base64 ile
        result = await asistan.prepare(user_input, chat_history=[], firlama_modu=firlama_modu, image_base64=img_base64)

        messages = result.get("messages", [])
        image_for_llm = result.get("image_base64")

        if not messages:
            await status.delete()
            await update.message.reply_text("Fotoğrafı işleyemedim, tekrar dener misin?")
            return

        # PersonalAI ile cevap üret (multimodal - fotoğraf dahil)
        response = await ai.generate(messages=messages, image_base64=image_for_llm)

        # Düşünüyorum mesajını sil
        await status.delete()

        # Cevabı gönder
        await update.message.reply_text(response)

        # Hafızaya kaydet
        # Kullanıcı mesajı: "[Kullanıcı fotoğraf gönderdi]" (vision cevabı YOK)
        # Asistan mesajı: Ana LLM'in gerçek cevabı
        if caption:
            foto_kayit = f"[Kullanıcı fotoğraf gönderdi: {caption}]"
        else:
            foto_kayit = "[Kullanıcı fotoğraf gönderdi]"
        asistan.save(foto_kayit, response, [])

    except Exception as e:
        print(f"[HATA] Fotograf hatasi: {e}")
        import traceback
        traceback.print_exc()
        try:
            await status.delete()
        except:
            pass
        await update.message.reply_text("Fotoğrafı işlerken bir sorun oluştu.")


# === BELGE/DOSYA HANDLER ===

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📄 Belge yükleme handler - PDF, DOCX, TXT"""
    try:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # Kullanıcıyı al/oluştur
        get_user_ai(user_id)

        document = update.message.document
        file_name = document.file_name.lower()

        # Desteklenen formatlar
        supported_ext = ['.pdf', '.docx', '.txt']
        ext = None
        for e in supported_ext:
            if file_name.endswith(e):
                ext = e
                break

        if not ext:
            await update.message.reply_text(
                "Bu dosya türünü desteklemiyorum.\n\n"
                "Desteklenen formatlar:\n"
                "• PDF (.pdf)\n"
                "• Word (.docx)\n"
                "• Metin (.txt)"
            )
            return

        # Dosya boyutu kontrolü (1 MB limit - laptop dostu)
        if document.file_size > 1 * 1024 * 1024:
            await update.message.reply_text(
                "📄 Dosya çok büyük!\n\n"
                "Maksimum 1 MB yükleyebilirsin.\n"
                "Daha küçük bir dosya dene."
            )
            return

        # Belge sayısı kontrolü (max 2 belge)
        belge_asistani = get_belge_asistani(user_id)
        mevcut_belgeler = belge_asistani.listele()
        if len(mevcut_belgeler) >= 2:
            await update.message.reply_text(
                "📄 Belge limitine ulaştın!\n\n"
                "Maksimum 2 belge yükleyebilirsin.\n"
                "/belgelerim yazıp eski belgeleri silebilirsin."
            )
            return

        # Düşünüyorum mesajı
        status = await context.bot.send_message(chat_id, "📄 Belgeyi işliyorum...")

        # Dosyayı indir
        file = await context.bot.get_file(document.file_id)
        temp_dir = os.path.join("temp_uploads", f"user_{user_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, document.file_name)

        await file.download_to_drive(temp_path)

        result = belge_asistani.belge_yukle(temp_path)

        # Temp dosyayı sil
        try:
            os.remove(temp_path)
        except:
            pass

        # Düşünüyorum mesajını sil
        await status.delete()

        if result['success']:
            # Özet varsa göster
            ozet_text = ""
            if result.get('ozet') and result['ozet'] != "Özet oluşturulamadı":
                ozet_text = f"\n\n📝 Özet:\n{result['ozet']}"

            # Yükleme sonrası butonlarla yönlendir
            buttons = [
                [InlineKeyboardButton("💬 Bu Belge Üzerine Konuş", callback_data=f"belge_aktif:{result['belge_id']}")],
                [InlineKeyboardButton("📄 Tüm Belgelerim", callback_data="belge_liste_geri")]
            ]
            await update.message.reply_text(
                f"✅ Belge yüklendi!\n\n"
                f"📄 {result['dosya_adi']}\n"
                f"📊 {result['chunk_sayisi']} parça\n"
                f"📝 {result['karakter_sayisi']:,} karakter"
                f"{ozet_text}",
                reply_markup=InlineKeyboardMarkup(buttons)
            )
        else:
            await update.message.reply_text(f"❌ Hata: {result['error']}")

    except Exception as e:
        print(f"[HATA] Belge hatasi: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text("Belgeyi işlerken bir sorun oluştu.")


# === KAMERA WIZARD HANDLER ===

async def handle_kamera_wizard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kamera ekleme wizard adımlarını işle"""
    global user_kamera_wizard

    user_id = update.effective_user.id
    user_input = update.message.text.strip()

    if user_id not in user_kamera_wizard:
        return

    wizard = user_kamera_wizard[user_id]
    adim = wizard["adim"]
    data = wizard["data"]

    # İptal butonu
    iptal_btn = InlineKeyboardMarkup([[InlineKeyboardButton("❌ İptal", callback_data="kamera_wizard_iptal")]])

    # Adım: Kamera adı
    if adim == "ad":
        if len(user_input) < 2:
            await update.message.reply_text(
                "Kamera adı en az 2 karakter olmalı.",
                reply_markup=ForceReply(input_field_placeholder="Örn: Bahçe Kamerası")
            )
            await update.message.reply_text("↩️", reply_markup=iptal_btn)
            return

        data["ad"] = user_input
        wizard["adim"] = "ip"
        await update.message.reply_text(
            f"Kamera adı: {user_input}\n\n"
            "Adım 2/6: DVR/Kamera IP adresi",
            reply_markup=ForceReply(input_field_placeholder="Örn: 192.168.1.4")
        )
        await update.message.reply_text("↩️", reply_markup=iptal_btn)

    # Adım: IP adresi
    elif adim == "ip":
        # Basit IP validasyonu
        import re
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(ip_pattern, user_input):
            await update.message.reply_text(
                "Geçersiz IP adresi formatı.",
                reply_markup=ForceReply(input_field_placeholder="Örn: 192.168.1.4")
            )
            await update.message.reply_text("↩️", reply_markup=iptal_btn)
            return

        data["ip"] = user_input
        wizard["adim"] = "port"
        # Port seçimi için butonlar
        keyboard = [
            [InlineKeyboardButton("554 (Standart)", callback_data="kamera_port:554")],
            [InlineKeyboardButton("8554", callback_data="kamera_port:8554")],
            [InlineKeyboardButton("Farklı Port Gir", callback_data="kamera_port:custom")],
            [InlineKeyboardButton("İptal", callback_data="kamera_wizard_iptal")]
        ]
        await update.message.reply_text(
            f"IP: {user_input}\n\n"
            "Adım 3/6: RTSP Port seç",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # Adım: Port (custom port girişi için)
    elif adim == "port":
        try:
            port = int(user_input)
            if port < 1 or port > 65535:
                raise ValueError
        except:
            await update.message.reply_text(
                "Geçersiz port. 1-65535 arası olmalı.",
                reply_markup=ForceReply(input_field_placeholder="Port numarası girin")
            )
            await update.message.reply_text("↩️", reply_markup=iptal_btn)
            return

        data["port"] = port
        wizard["adim"] = "kullanici"
        await update.message.reply_text(
            f"Port: {port}\n\n"
            "Adım 4/6: Kullanıcı adı",
            reply_markup=ForceReply(input_field_placeholder="Örn: admin")
        )
        await update.message.reply_text("↩️", reply_markup=iptal_btn)

    # Adım: Kullanıcı adı
    elif adim == "kullanici":
        if len(user_input) < 1:
            await update.message.reply_text(
                "Kullanıcı adı boş olamaz.",
                reply_markup=ForceReply(input_field_placeholder="Kullanıcı adı girin")
            )
            await update.message.reply_text("↩️", reply_markup=iptal_btn)
            return

        data["kullanici"] = user_input
        wizard["adim"] = "sifre"
        await update.message.reply_text(
            f"Kullanıcı: {user_input}\n\n"
            "Adım 5/6: Şifre gir\n"
            "(mesajın güvenlik için silinecek)",
            reply_markup=ForceReply(input_field_placeholder="Şifre girin")
        )
        await update.message.reply_text("↩️", reply_markup=iptal_btn)

    # Adım: Şifre
    elif adim == "sifre":
        # Şifre mesajını sil (güvenlik)
        try:
            await update.message.delete()
        except:
            pass

        if len(user_input) < 1:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Şifre boş olamaz.",
                reply_markup=ForceReply(input_field_placeholder="Şifre girin")
            )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="↩️",
                reply_markup=iptal_btn
            )
            return

        data["sifre"] = user_input
        wizard["adim"] = "kanal"
        # Kanal seçimi için butonlar (4x4 grid)
        keyboard = [
            [
                InlineKeyboardButton("1", callback_data="kamera_kanal:1"),
                InlineKeyboardButton("2", callback_data="kamera_kanal:2"),
                InlineKeyboardButton("3", callback_data="kamera_kanal:3"),
                InlineKeyboardButton("4", callback_data="kamera_kanal:4")
            ],
            [
                InlineKeyboardButton("5", callback_data="kamera_kanal:5"),
                InlineKeyboardButton("6", callback_data="kamera_kanal:6"),
                InlineKeyboardButton("7", callback_data="kamera_kanal:7"),
                InlineKeyboardButton("8", callback_data="kamera_kanal:8")
            ],
            [
                InlineKeyboardButton("9", callback_data="kamera_kanal:9"),
                InlineKeyboardButton("10", callback_data="kamera_kanal:10"),
                InlineKeyboardButton("11", callback_data="kamera_kanal:11"),
                InlineKeyboardButton("12", callback_data="kamera_kanal:12")
            ],
            [
                InlineKeyboardButton("13", callback_data="kamera_kanal:13"),
                InlineKeyboardButton("14", callback_data="kamera_kanal:14"),
                InlineKeyboardButton("15", callback_data="kamera_kanal:15"),
                InlineKeyboardButton("16", callback_data="kamera_kanal:16")
            ],
            [InlineKeyboardButton("İptal", callback_data="kamera_wizard_iptal")]
        ]
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Şifre kaydedildi.\n\n"
                 "Adım 6/6: DVR kanal numarası seç",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # Adım: Kanal (artık butonlarla seçiliyor, bu kod gereksiz ama yedek olarak kalsın)
    elif adim == "kanal":
        # Butonlar kullanıldığı için buraya normalde gelmemeli
        await update.message.reply_text("Lütfen yukarıdaki butonlardan kanal seçin.")


# === MESAJ HANDLER ===

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ana akış:
    1. Telegram mesaj alır
    2. HafizaAsistani.prepare() → messages hazırlar
    3. Asistan.prepare() → messages hazırlar
    4. PersonalAI.generate() → cevap üretir
    5. Asistan.save() → hafızaya kaydeder
    6. Telegram'a cevap gönderir
    """
    global user_kamera_wizard

    user_id = update.effective_user.id
    user_input = update.message.text
    chat_id = update.effective_chat.id

    # 🔒 RATE LIMIT KONTROLU (Admin muaf)
    db = get_db()
    user_info = update.effective_user
    db.get_or_create_user(
        user_id,
        username=user_info.username,
        first_name=user_info.first_name,
        last_name=user_info.last_name
    )

    # 🔒 RATE LIMIT - Beta: Günlük 30 mesaj limiti
    if user_id not in ADMIN_IDS:
        rate_check = db.check_rate_limit(user_id)
        if not rate_check["allowed"]:
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💝 Bağış Yap", callback_data="bagis_menu")],
                [InlineKeyboardButton("📊 Limitlerimi Gör", callback_data="limit_info")]
            ])
            await update.message.reply_text(
                f"📊 *Bugünlük {rate_check['limit']} mesaj hakkın doldu!*\n\n"
                "Yarın sıfırlanır, görüşürüz!\n\n"
                "💝 Bu bot beta aşamasında ve kısıtlı donanımda çalışıyor.\n"
                "Beğendiysen projeyi destekleyebilirsin.",
                reply_markup=keyboard,
                parse_mode="Markdown"
            )
            return
        db.increment_usage(user_id, "message_count")

    # 📷 KAMERA WIZARD - Aktifse önce bunu işle
    if user_id in user_kamera_wizard:
        await handle_kamera_wizard(update, context)
        return

    # 📍 KONUM SİSTEMİ - Artık sadece /konum komutu ve butonlarla çalışıyor
    # Mesaj içeriğinden otomatik tetikleme kaldırıldı
    user_lower = user_input.lower().strip()

    # 📝 NOT KAYDETME - Butonla not yazdıysa direkt kaydet (LLM'e gitmesin)
    if context.user_data.get("not_bekliyor"):
        context.user_data["not_bekliyor"] = False

        # Direkt notu kaydet
        user = get_user_ai(user_id)
        user["hafiza"].not_manager.not_al(user_input)

        # Son kaydedilen notu al (tarih/saat için)
        not_id = None
        not_tarih = ""
        not_saat = ""
        if user["hafiza"].not_manager.notes:
            son_not = user["hafiza"].not_manager.notes[-1]
            not_id = son_not.get('id')
            not_tarih = son_not.get('tarih', '')
            not_saat = son_not.get('saat', '')

        # Hatırlatma seçeneği ile yanıt
        buttons = [
            [InlineKeyboardButton("🕐 Hatırlatma Ekle", callback_data=f"hatirlatma_ozel:{not_id}")],
            [InlineKeyboardButton("✅ Tamam", callback_data=f"hatirlatma_iptal:{not_id}")]
        ]

        await update.message.reply_text(
            f"✅ Not kaydedildi:\n\n"
            f"📝 {user_input}\n"
            f"📅 {not_tarih} - {not_saat}\n\n"
            f"⏰ Hatırlatma eklemek ister misin?",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    # ⏰ ÖZEL HATIRLATMA SÜRESİ - Kullanıcı süre girdiyse
    if context.user_data.get("hatirlatma_ozel_not_id"):
        not_id = context.user_data.pop("hatirlatma_ozel_not_id")

        # Süreyi parse et
        dakika = _parse_ozel_sure(user_input)

        if dakika is None or dakika <= 0:
            await update.message.reply_text(
                "❌ Süreyi anlayamadım.\n\n"
                "Örnekler: `40`, `40 dk`, `2 saat`, `1 saat 30 dk`",
                parse_mode="Markdown"
            )
            return

        # Kullanıcının AI'larını al
        user = get_user_ai(user_id)
        asistan = user["hafiza"]

        # Not'u bul ve hatırlatma ekle
        from datetime import datetime, timedelta
        hatirlatma_zamani = datetime.now() + timedelta(minutes=dakika)

        not_bulundu = False
        for n in asistan.not_manager.notes:
            if n.get('id') == not_id:
                n['hatirlatma'] = hatirlatma_zamani.isoformat()
                n['hatirlatma_gonderildi'] = False
                asistan.not_manager._save_notes()
                not_bulundu = True

                # JobQueue'ya ekle
                hatirlatma_job_ekle(context.application, user_id, n)

                # Süre formatla
                if dakika < 60:
                    sure_str = f"{dakika} dakika"
                elif dakika % 60 == 0:
                    sure_str = f"{dakika // 60} saat"
                else:
                    saat = dakika // 60
                    dk = dakika % 60
                    sure_str = f"{saat} saat {dk} dakika"

                buttons = [[InlineKeyboardButton("↩️ Geri Al", callback_data=f"hatirlatma_geri_al:{not_id}")]]
                await update.message.reply_text(
                    f"✅ Hatırlatma eklendi!\n\n"
                    f"📝 {n['icerik']}\n\n"
                    f"⏰ {sure_str} sonra\n"
                    f"🕐 {hatirlatma_zamani.strftime('%H:%M')}",
                    reply_markup=InlineKeyboardMarkup(buttons)
                )
                return

        if not not_bulundu:
            await update.message.reply_text("❌ Not bulunamadı.")
        return

    # Kullanıcının AI'larını al
    user = get_user_ai(user_id)

    # Düşünüyorum mesajı (aktif belge varsa göster)
    belge_asistani_check = get_belge_asistani(user_id)
    aktif_belge_check = belge_asistani_check.get_aktif()
    if aktif_belge_check:
        status_text = f"💭 Düşünüyorum... 📄 {aktif_belge_check['dosya_adi']}"
    else:
        status_text = "💭 Düşünüyorum..."
    status = await context.bot.send_message(chat_id, status_text)

    try:
        ai = user["ai"]
        asistan = user["hafiza"]

        # 📄 AKTİF BELGE CONTEXT - Belge üzerine konuşma modu
        belge_asistani = get_belge_asistani(user_id)
        aktif_belge = belge_asistani.get_aktif()

        if aktif_belge:
            # Timeout kontrolü
            timeout = belge_asistani.increment_mesaj()

            if timeout:
                if timeout.startswith("sure_limit:"):
                    # 30 dakika doldu - otomatik kapatıldı
                    dosya_adi = timeout.split(":")[1]
                    asistan.belge_context = None
                    try:
                        await context.bot.delete_message(chat_id, status.message_id)
                    except:
                        pass
                    await update.message.reply_text(
                        f"⏱️ *{dosya_adi}* 30 dakika dolduğu için otomatik kapatıldı.\n\n"
                        f"Normal sohbete dönüldü.",
                        parse_mode="Markdown"
                    )
                    return

                elif timeout == "mesaj_limit_sor":
                    # 15 mesaj doldu - soru sor
                    dosya_adi = aktif_belge['dosya_adi']
                    buttons = [
                        [InlineKeyboardButton("✅ Devam Et", callback_data="belge_devam")],
                        [InlineKeyboardButton("❌ Kapat", callback_data="belge_aktif_kapat")]
                    ]
                    try:
                        await context.bot.delete_message(chat_id, status.message_id)
                    except:
                        pass
                    await update.message.reply_text(
                        f"📄 *{dosya_adi}* üzerinde 15 mesaj konuştunuz.\n\n"
                        f"Devam etmek ister misiniz?",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(buttons)
                    )
                    return

            # Belge context'i ekle
            belge_context = belge_asistani.get_context(user_input, k=3)
            if belge_context:
                asistan.belge_context = belge_context
                print(f"[BELGE] Aktif belge context eklendi: {aktif_belge['dosya_adi']}")
        else:
            asistan.belge_context = None

        result = await asyncio.wait_for(
            asistan.prepare(user_input, []),
            timeout=TIMEOUT
        )

        # 📝 Paket kontrolü
        paket = result.get("paket", {})

        # 📍 KONUM GÖNDERME - Telegram location mesajı
        if paket.get("send_location"):
            loc = paket["send_location"]
            # Status mesajını sil
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass
            # Konum mesajı gönder
            await context.bot.send_location(
                chat_id=chat_id,
                latitude=loc["lat"],
                longitude=loc["lon"]
            )
            # Bilgi mesajı
            await update.message.reply_text(
                f"📍 {loc['ad']}\n📏 {loc['mesafe']}m uzaklıkta"
            )
            # History'e kaydet
            asistan.save(user_input, f"[Konum gönderildi: {loc['ad']}]", [])
            return

        # 📍 YAKIN YERLER LİSTESİ - Inline butonlarla göster
        if paket.get("yakin_yerler"):
            data = paket["yakin_yerler"]
            emoji = data["emoji"]
            kategori = data["kategori"]
            yerler = data["yerler"]

            # Status mesajını sil
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass

            # Mesaj oluştur
            mesaj = f"{emoji} Yakınındaki {kategori}ler:\n\n"
            buttons = []
            for i, yer in enumerate(yerler, 1):
                mesaj += f"{i}. {yer['ad']} ({yer['mesafe']}m)\n"
                buttons.append([InlineKeyboardButton(
                    f"{i}. {yer['ad'][:25]}{'...' if len(yer['ad']) > 25 else ''} ({yer['mesafe']}m)",
                    callback_data=f"konum_gonder:{i-1}"
                )])

            reply_markup = InlineKeyboardMarkup(buttons)
            await update.message.reply_text(mesaj, reply_markup=reply_markup)

            # History'e kaydet
            asistan.save(user_input, mesaj, [])
            return

        # ⏰ HATIRLATMA SEÇİMİ - Not kaydedildi, zaman seçimi butonları göster
        if paket.get("hatirlatma_secimi"):
            data = paket["hatirlatma_secimi"]
            mesaj = data["mesaj"]
            not_id = data["not_id"]

            # Status mesajını sil
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass

            # Zaman seçimi butonları
            buttons = [
                [
                    InlineKeyboardButton("1 dk", callback_data=f"hatirlatma_ekle:{not_id}:1"),
                    InlineKeyboardButton("5 dk", callback_data=f"hatirlatma_ekle:{not_id}:5"),
                    InlineKeyboardButton("15 dk", callback_data=f"hatirlatma_ekle:{not_id}:15"),
                    InlineKeyboardButton("30 dk", callback_data=f"hatirlatma_ekle:{not_id}:30"),
                ],
                [
                    InlineKeyboardButton("1 sa", callback_data=f"hatirlatma_ekle:{not_id}:60"),
                    InlineKeyboardButton("2 sa", callback_data=f"hatirlatma_ekle:{not_id}:120"),
                    InlineKeyboardButton("6 sa", callback_data=f"hatirlatma_ekle:{not_id}:360"),
                    InlineKeyboardButton("12 sa", callback_data=f"hatirlatma_ekle:{not_id}:720"),
                ],
                [
                    InlineKeyboardButton("24 sa", callback_data=f"hatirlatma_ekle:{not_id}:1440"),
                    InlineKeyboardButton("⌨️ Özel Süre", callback_data=f"hatirlatma_ozel:{not_id}"),
                ],
                [
                    InlineKeyboardButton("❌ Hatırlatma Yok", callback_data="hatirlatma_iptal"),
                ],
            ]

            await update.message.reply_text(mesaj, reply_markup=InlineKeyboardMarkup(buttons))
            asistan.save(user_input, mesaj, [])
            return

        # 📝 NOTLAR LİSTESİ - Inline butonlarla göster
        if paket.get("notlar_listesi"):
            data = paket["notlar_listesi"]
            baslik = data["baslik"]
            notlar = data["notlar"]

            # Status mesajını sil
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass

            # Mesaj oluştur
            mesaj = f"{baslik}\n\n"
            buttons = []
            for n in notlar:
                gun = n.get('gun', '')
                gun_str = f" {gun}" if gun else ""
                hatirlatma_str = ""
                if n.get('hatirlatma') and not n.get('hatirlatma_gonderildi', False):
                    hatirlatma_str = " ⏰"
                mesaj += f"{n['id']}. [{n['tarih']}{gun_str} - {n['saat']}]{hatirlatma_str}\n"
                mesaj += f"   {n['icerik']}\n\n"
                # Silme butonu
                buttons.append([InlineKeyboardButton(
                    f"🗑️ {n['id']}. sil",
                    callback_data=f"not_sil:{n['id']}"
                )])

            # Kapat butonu
            buttons.append([InlineKeyboardButton("✅ Tamam", callback_data="notlar_kapat")])

            reply_markup = InlineKeyboardMarkup(buttons)
            await update.message.reply_text(mesaj.strip(), reply_markup=reply_markup)
            return

        # 📝 Direct response kontrolü (not sistemi, konum araçları vs.)
        if paket.get("direct_response"):
            response = paket["direct_response"]
            # Araç sonucunu history'e kaydet (LLM bağlamı korusun)
            tool_used = paket.get("tool_used", "")
            if tool_used in ["konum_hizmeti", "not_sistemi"]:
                asistan.save(user_input, response, [])

        # 🎨 GÖRSEL - Gemma BYPASS (halüsinasyon önleme)
        elif paket.get("gorsel_bytes") and paket.get("gemma_bypass"):
            # Status sil
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass

            # Sabit cevap (Gemma çağrılmadı)
            response = paket.get("sabit_cevap", "İşte sana hazırladığım görsel! 😊")
            await update.message.reply_text(response)

            # Görsel gönder
            await update.message.reply_photo(
                photo=io.BytesIO(paket["gorsel_bytes"])
            )

            # 🔢 Kullanımı artır (admin hariç)
            if user_id not in ADMIN_IDS:
                db = get_db()
                db.increment_usage(user_id, "image_count")

            # Kaydet (history'e Gemma söylemiş gibi ekle)
            asistan.save(user_input, response, [])
            return

        # 🔒 Görsel limiti aşıldıysa bildir
        elif paket.get("gorsel_limit_asıldı"):
            try:
                await context.bot.delete_message(chat_id, status.message_id)
            except:
                pass
            await update.message.reply_text(f"😔 {paket.get('gorsel_limit_mesaj', 'Görsel limitin doldu.')}")
            return

        else:
            messages = result["messages"]

            # Cevap üret
            response = await asyncio.wait_for(
                ai.generate(messages=messages),
                timeout=TIMEOUT
            )

            # Çıktıyı temizle (markdown + yasak ifadeler)
            response = temizle_cikti(response)

            # Kaydet
            asistan.save(user_input, response, [])

    except asyncio.TimeoutError:
        response = "⏱️ Zaman aşımı, tekrar dene."
    except Exception as e:
        print(f"[HATA]: {e}")
        response = "❌ Bir sorun oluştu."

    # Status mesajını sil
    try:
        await context.bot.delete_message(chat_id, status.message_id)
    except:
        pass

    # Cevabı gönder (aktif belge varsa butonlu)
    belge_asistani_cevap = get_belge_asistani(user_id)
    aktif_belge_cevap = belge_asistani_cevap.get_aktif()

    if aktif_belge_cevap:
        # Aktif belge var - "Bitir" butonu ekle
        buttons = [[InlineKeyboardButton("❌ Bitir", callback_data="belge_aktif_kapat")]]
        await update.message.reply_text(
            response,
            reply_markup=InlineKeyboardMarkup(buttons)
        )
    else:
        # Normal cevap
        await update.message.reply_text(response)


# === CALLBACK HANDLER (Inline butonlar için) ===

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline buton tıklamalarını işle"""
    global user_instances

    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    chat_id = query.message.chat_id
    data = query.data

    print(f"[CALLBACK] {data} (user: {user_id})")

    # Konum arama callback'i: konum_ara:kategori
    if data.startswith("konum_ara:"):
        kategori = data.split(":")[1]

        # Kullanıcıyı kontrol et
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Konum kontrolü
        if not asistan.user_location:
            await query.edit_message_text("❌ Önce konum paylaş.")
            return

        lat, lon = asistan.user_location

        # Arama yap
        try:
            # Nöbetçi eczane için il/ilçe seçeneği göster
            if kategori == "nobetci_eczane":
                # İl/ilçe bilgisini al
                if asistan.konum_adres:
                    parcalar = [p.strip() for p in asistan.konum_adres.split(",")]
                    if len(parcalar) >= 2:
                        il = parcalar[-1]
                        ilce = parcalar[-2]
                        buttons = [
                            [InlineKeyboardButton(f"🏘️ {ilce} (ilçe)", callback_data=f"nobetci_ara:ilce:{ilce}:{il}")],
                            [InlineKeyboardButton(f"🏙️ {il} (tüm il)", callback_data=f"nobetci_ara:il:{il}")],
                            [InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]
                        ]
                        reply_markup = InlineKeyboardMarkup(buttons)
                        await query.edit_message_text(
                            f"🌙 Nöbetçi Eczane\n\nNerede arayalım?",
                            reply_markup=reply_markup
                        )
                        return
                # Adres yoksa direkt il için ara
                result = await asistan._get_nobetci_eczane(lat, lon)

            # Yakıt fiyatları için özel işlem
            elif kategori == "yakit_fiyat":
                if asistan.konum_adres:
                    parcalar = [p.strip() for p in asistan.konum_adres.split(",")]
                    if len(parcalar) >= 1:
                        il = parcalar[-1]
                        result = await asistan._get_yakit_fiyatlari(il)

                        if isinstance(result, str):
                            # Hata mesajı
                            geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
                            await query.edit_message_text(result, reply_markup=geri_btn)
                        else:
                            # Başarılı sonuç
                            geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
                            await query.edit_message_text(result, reply_markup=geri_btn, parse_mode="Markdown")
                        return
                result = "❌ Konum bilgisi bulunamadı."
            else:
                result = await asistan._get_yakin_yerler(lat, lon, kategori)

            # Dict döndüyse inline butonlarla göster
            if isinstance(result, dict) and result.get("type") == "yakin_yerler_listesi":
                yerler = result["yerler"]

                mesaj = f"Yakınındaki {kategori}ler:\n\n"
                buttons = []
                for i, yer in enumerate(yerler, 1):
                    # 99999m = koordinat yok
                    has_konum = yer['mesafe'] < 99999
                    mesafe_str = f"{yer['mesafe']}m" if has_konum else "📍yok"

                    # Mesajda adres/tel varsa göster
                    mesaj += f"{i}. {yer['ad']} ({mesafe_str})"
                    if not has_konum and yer.get('adres'):
                        mesaj += f"\n   📫 {yer['adres'][:40]}"
                    if not has_konum and yer.get('telefon'):
                        mesaj += f"\n   📞 {yer['telefon']}"
                    mesaj += "\n"

                    # Buton metni
                    btn_text = f"{i}. {yer['ad'][:20]}{'...' if len(yer['ad']) > 20 else ''}"
                    if has_konum:
                        btn_text += f" ({mesafe_str})"
                    else:
                        btn_text += " 📍yok"

                    buttons.append([InlineKeyboardButton(
                        btn_text,
                        callback_data=f"konum_gonder:{i-1}"
                    )])

                # Geri butonu ekle
                buttons.append([InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")])

                reply_markup = InlineKeyboardMarkup(buttons)
                await query.edit_message_text(mesaj, reply_markup=reply_markup)
            else:
                # String döndüyse (hata mesajı vs.) - geri butonuyla göster
                geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
                await query.edit_message_text(result if result else f"{kategori} bulunamadı.", reply_markup=geri_btn)
        except Exception as e:
            print(f"Callback hata: {e}")
            geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
            await query.edit_message_text(f"{kategori} araması başarısız.", reply_markup=geri_btn)

    # Nöbetçi eczane il/ilçe seçimi: nobetci_ara:tip:ilce:il veya nobetci_ara:il:il
    elif data.startswith("nobetci_ara:"):
        parts = data.split(":")
        tip = parts[1]  # "ilce" veya "il"

        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        if not asistan.user_location:
            await query.edit_message_text("❌ Önce konum paylaş.")
            return

        lat, lon = asistan.user_location

        try:
            if tip == "ilce":
                ilce = parts[2]
                il = parts[3]
                result = await asistan._get_nobetci_eczane(lat, lon, ilce=ilce, il=il)
            else:  # tip == "il"
                il = parts[2]
                result = await asistan._get_nobetci_eczane(lat, lon, il=il)

            # Sonuçları göster
            if isinstance(result, dict) and result.get("type") == "yakin_yerler_listesi":
                yerler = result["yerler"]
                kategori = "nöbetçi eczane"

                mesaj = f"🌙 Nöbetçi Eczaneler:\n\n"
                buttons = []
                for i, yer in enumerate(yerler, 1):
                    has_konum = yer['mesafe'] < 99999
                    mesafe_str = f"{yer['mesafe']}m" if has_konum else "📍yok"
                    mesaj += f"{i}. {yer['ad']} ({mesafe_str})\n"

                    btn_text = f"{i}. {yer['ad'][:20]}{'...' if len(yer['ad']) > 20 else ''}"
                    if has_konum:
                        btn_text += f" ({mesafe_str})"
                    buttons.append([InlineKeyboardButton(btn_text, callback_data=f"konum_gonder:{i-1}")])

                buttons.append([InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")])
                reply_markup = InlineKeyboardMarkup(buttons)
                await query.edit_message_text(mesaj, reply_markup=reply_markup)
            else:
                geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
                await query.edit_message_text(result if result else "Nöbetçi eczane bulunamadı.", reply_markup=geri_btn)
        except Exception as e:
            print(f"Nöbetçi eczane hata: {e}")
            geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
            await query.edit_message_text(f"Nöbetçi eczane araması başarısız.", reply_markup=geri_btn)

    # Konum gönderme callback'i: konum_gonder:index
    elif data.startswith("konum_gonder:"):
        index = int(data.split(":")[1])

        # Kullanıcıyı kontrol et
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Son arama sonuçları kontrolü
        if not asistan.son_yakin_yerler:
            await query.edit_message_text("❌ Önce yakın yer araması yap.")
            return

        if index < 0 or index >= len(asistan.son_yakin_yerler):
            await query.edit_message_text("❌ Geçersiz seçim.")
            return

        yer = asistan.son_yakin_yerler[index]

        # Koordinat kontrolü
        if not yer.get("lat") or not yer.get("lon"):
            geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
            # Adres/telefon bilgisi varsa göster
            mesaj = f"📍 {yer['ad']}\n\n❌ Koordinat bilgisi yok."
            if yer.get("adres"):
                mesaj += f"\n📫 Adres: {yer['adres']}"
            if yer.get("telefon"):
                mesaj += f"\n📞 Tel: {yer['telefon']}"
            await query.edit_message_text(mesaj, reply_markup=geri_btn)
            return

        # Mesajı güncelle
        await query.edit_message_text(f"📍 {yer['ad']} konumu gönderiliyor...")

        # Konum mesajı gönder
        await context.bot.send_location(
            chat_id=chat_id,
            latitude=yer["lat"],
            longitude=yer["lon"]
        )

        # Bilgi mesajı + geri butonu
        geri_btn = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Kategoriler", callback_data="konum_menu")]])
        info_text = f"📍 {yer['ad']}\n📏 {yer['mesafe']}m uzaklıkta"
        if yer.get("adres"):
            info_text += f"\n📫 {yer['adres']}"
        if yer.get("telefon"):
            info_text += f"\n📞 {yer['telefon']}"
        await context.bot.send_message(
            chat_id=chat_id,
            text=info_text,
            reply_markup=geri_btn
        )

    # 📝 NOT SİL callback'i: not_sil:id
    elif data.startswith("not_sil:"):
        not_id = int(data.split(":")[1])

        # Kullanıcıyı kontrol et
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Notu sil
        silme_sonuc = asistan.not_manager.not_sil(not_id)

        # İlgili hatırlatma job'ını da sil
        job_name = f"hatirlatma_{user_id}_{not_id}"
        jobs = context.job_queue.get_jobs_by_name(job_name)
        for job in jobs:
            job.schedule_removal()
            print(f"[HATIRLATMA] Job silindi (not silme): {job_name}")

        # Güncel notları al
        notlar_result = asistan.not_manager.notlari_getir()

        # Eğer not kalmadıysa sadece mesaj göster
        if isinstance(notlar_result, str):
            await query.edit_message_text(f"{silme_sonuc}\n\n{notlar_result}")
            return

        # Not varsa listeyi güncelle
        notlar = notlar_result["notlar"]
        baslik = notlar_result["baslik"]

        mesaj = f"✅ {silme_sonuc}\n\n{baslik}\n\n"
        buttons = []
        for n in notlar:
            gun = n.get('gun', '')
            gun_str = f" {gun}" if gun else ""
            hatirlatma_str = ""
            if n.get('hatirlatma') and not n.get('hatirlatma_gonderildi', False):
                hatirlatma_str = " ⏰"
            mesaj += f"{n['id']}. [{n['tarih']}{gun_str} - {n['saat']}]{hatirlatma_str}\n"
            mesaj += f"   {n['icerik']}\n\n"
            # Silme butonu
            buttons.append([InlineKeyboardButton(
                f"🗑️ {n['id']}. sil",
                callback_data=f"not_sil:{n['id']}"
            )])

        # Kapat butonu ekle
        buttons.append([InlineKeyboardButton("✅ Tamam", callback_data="notlar_kapat")])

        await query.edit_message_text(mesaj, reply_markup=InlineKeyboardMarkup(buttons))

    # 📝 NOTLAR LİSTESİNİ KAPAT
    elif data == "notlar_kapat":
        await query.edit_message_text("📝 Notlar kapatıldı.")

    # 📄 DOKÜMAN GÖR - İçeriği göster
    elif data.startswith("belge_gor:"):
        belge_id = data.split(":")[1]

        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle (not vs.)
        clear_active_modes(user_id, context)

        belge_asistani = get_belge_asistani(user_id)
        result = belge_asistani.get_icerik(belge_id)

        if not result["success"]:
            await query.edit_message_text(f"❌ {result['error']}")
            return

        # Özeti al (Markdown karakterlerini escape et)
        ozet = belge_asistani.get_ozet(belge_id)
        if ozet and ozet != "Özet oluşturulamadı":
            ozet_safe = ozet.replace('*', '').replace('_', '').replace('`', '').replace('[', '').replace(']', '')
            ozet_text = f"📝 *Özet:*\n{ozet_safe}"
        else:
            # Özet yoksa içeriğin başını göster
            icerik = result["icerik"][:400]
            if len(result["icerik"]) > 400:
                icerik += "..."
            icerik_safe = icerik.replace('*', '').replace('_', '').replace('`', '').replace('[', '').replace(']', '')
            ozet_text = f"📄 *İçerik:*\n{icerik_safe}"

        buttons = [
            [InlineKeyboardButton("💬 Bu Belge Üzerine Konuş", callback_data=f"belge_aktif:{belge_id}")],
            [InlineKeyboardButton("🔙 Geri", callback_data="belge_liste_geri")]
        ]

        await query.edit_message_text(
            f"📄 *{result['dosya_adi']}*\n\n"
            f"{ozet_text}\n\n"
            f"💬 Sorularına bu belgeden cevap alırsın.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 📄 LİSTEYE GERİ DÖN
    elif data == "belge_liste_geri":
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        belge_asistani = get_belge_asistani(user_id)
        belgeler = belge_asistani.listele()

        if not belgeler:
            await query.edit_message_text("📄 Henüz doküman yok.")
            return

        buttons = []
        for b in belgeler:
            buttons.append([InlineKeyboardButton(
                f"📄 {b['dosya_adi']}",
                callback_data=f"belge_gor:{b['belge_id']}"
            )])
        buttons.append([InlineKeyboardButton("🗑️ Doküman Sil", callback_data="belge_sil_menu")])

        await query.edit_message_text(
            f"📄 *Çalışma Alanım*\n\n"
            f"Bir doküman seç, içeriğini gör ve üzerine konuş:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 📄 DOKÜMAN SİL MENÜ
    elif data == "belge_sil_menu":
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle (not vs.)
        clear_active_modes(user_id, context)

        # Belge asistanını al (lazy init)
        belge_asistani = get_belge_asistani(user_id)
        belgeler = belge_asistani.listele()
        if not belgeler:
            await query.edit_message_text("📄 Silinecek doküman yok.")
            return

        buttons = []
        for b in belgeler:
            buttons.append([InlineKeyboardButton(
                f"🗑️ {b['dosya_adi'][:30]}",
                callback_data=f"belge_sil:{b['belge_id']}"
            )])
        buttons.append([InlineKeyboardButton("❌ İptal", callback_data="belge_sil_iptal")])

        await query.edit_message_text(
            "🗑️ *Hangi dokümanı silmek istiyorsun?*",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 📄 BELGE SİL
    elif data.startswith("belge_sil:"):
        belge_id = data.split(":")[1]

        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # Belge asistanını al (lazy init)
        belge_asistani = get_belge_asistani(user_id)

        # Silinen belge aktif ise temizle
        if belge_asistani.aktif_belge_id == belge_id:
            belge_asistani.clear_aktif()

        result = belge_asistani.sil(belge_id)
        if result["success"]:
            # Kalan belgeleri göster
            belgeler = belge_asistani.listele()
            if not belgeler:
                await query.edit_message_text(f"✅ {result['dosya_adi']} silindi.\n\nTüm dokümanlar silindi.")
            else:
                buttons = []
                for b in belgeler:
                    buttons.append([InlineKeyboardButton(
                        f"📄 {b['dosya_adi']}",
                        callback_data=f"belge_gor:{b['belge_id']}"
                    )])
                buttons.append([InlineKeyboardButton("🗑️ Doküman Sil", callback_data="belge_sil_menu")])

                await query.edit_message_text(
                    f"✅ {result['dosya_adi']} silindi.\n\n"
                    f"📄 *Kalan Dokümanlar:*",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(buttons)
                )
        else:
            await query.edit_message_text(f"❌ {result['error']}")

    # 📄 BELGE SİL İPTAL - Listeye geri dön
    elif data == "belge_sil_iptal":
        belge_asistani = get_belge_asistani(user_id)
        belgeler = belge_asistani.listele()

        if not belgeler:
            await query.edit_message_text("📄 Henüz doküman yok.")
            return

        buttons = []
        for b in belgeler:
            buttons.append([InlineKeyboardButton(
                f"📄 {b['dosya_adi']}",
                callback_data=f"belge_gor:{b['belge_id']}"
            )])
        buttons.append([InlineKeyboardButton("🗑️ Doküman Sil", callback_data="belge_sil_menu")])

        await query.edit_message_text(
            f"📄 *Çalışma Alanım*\n\n"
            f"Bir doküman seç, içeriğini gör ve üzerine konuş:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 💬 BELGE AKTİF - Bu belge üzerine konuş modunu başlat
    elif data.startswith("belge_aktif:"):
        belge_id = data.split(":")[1]

        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle (not bekleme vs.)
        clear_active_modes(user_id, context)

        belge_asistani = get_belge_asistani(user_id)
        result = belge_asistani.set_aktif(belge_id)

        if not result["success"]:
            await query.edit_message_text(f"❌ {result['error']}")
            return

        buttons = [
            [InlineKeyboardButton("❌ Konuşmayı Bitir", callback_data="belge_aktif_kapat")]
        ]

        await query.edit_message_text(
            f"💬 *{result['dosya_adi']}* üzerine konuşma başladı!\n\n"
            f"Artık sorularına bu belgeden cevap vereceğim.\n\n"
            f"_Örnek: \"Bu belgede ne anlatılıyor?\", \"Özet çıkar\", \"X konusu nerede geçiyor?\"_",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 💬 BELGE AKTİF KAPAT - Konuşma modunu bitir
    elif data == "belge_aktif_kapat":
        belge_asistani = get_belge_asistani(user_id)
        belge_asistani.clear_aktif()

        # Belge listesine dön
        belgeler = belge_asistani.listele()
        if not belgeler:
            await query.edit_message_text("💬 Belge konuşması kapatıldı.\n\n📄 Henüz doküman yok.")
            return

        buttons = []
        for b in belgeler:
            buttons.append([InlineKeyboardButton(
                f"📄 {b['dosya_adi']}",
                callback_data=f"belge_gor:{b['belge_id']}"
            )])
        buttons.append([InlineKeyboardButton("🗑️ Doküman Sil", callback_data="belge_sil_menu")])

        await query.edit_message_text(
            f"💬 Belge konuşması kapatıldı.\n\n"
            f"📄 *Çalışma Alanım*\n\n"
            f"Bir doküman seç:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons)
        )

    # 💬 BELGE DEVAM - 15 mesaj sonrası devam et
    elif data == "belge_devam":
        belge_asistani = get_belge_asistani(user_id)
        aktif_belge = belge_asistani.get_aktif()

        if not aktif_belge:
            await query.edit_message_text("❌ Aktif belge bulunamadı.")
            return

        # Mesaj sayacını sıfırla
        belge_asistani.reset_mesaj_sayaci()

        await query.edit_message_text(
            f"✅ *{aktif_belge['dosya_adi']}* üzerinde konuşmaya devam ediyorsunuz.\n\n"
            f"_15 mesaj daha sorabilirsiniz._",
            parse_mode="Markdown"
        )

    # 📝 YENİ NOT - Not ekleme moduna geç
    elif data == "not_yeni":
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle (belge vs.)
        clear_active_modes(user_id, context)

        # Pending not moduna geç (context.user_data ile - message handler'da kontrol ediliyor)
        context.user_data["not_bekliyor"] = True

        # Eski mesajı sil
        try:
            await query.message.delete()
        except:
            pass

        # Yeni mesaj ile ForceReply gönder
        await context.bot.send_message(
            chat_id=chat_id,
            text="📝 *Not içeriğini yaz:*\n\n_Örnek: yarın toplantı var_",
            parse_mode="Markdown",
            reply_markup=ForceReply(selective=True)
        )

    # 📋 NOTLARIMI LİSTELE
    elif data == "not_listele":
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle (belge vs.)
        clear_active_modes(user_id, context)

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Notları getir
        notlar_result = asistan.not_manager.notlari_getir()

        # Not yoksa mesaj göster
        if isinstance(notlar_result, str):
            await query.edit_message_text(notlar_result)
            return

        # Notları listele
        notlar = notlar_result["notlar"]
        baslik = notlar_result["baslik"]

        mesaj = f"{baslik}\n\n"
        buttons = []
        for n in notlar:
            gun = n.get('gun', '')
            gun_str = f" {gun}" if gun else ""
            hatirlatma_str = ""
            if n.get('hatirlatma') and not n.get('hatirlatma_gonderildi', False):
                hatirlatma_str = " ⏰"
            mesaj += f"{n['id']}. [{n['tarih']}{gun_str} - {n['saat']}]{hatirlatma_str}\n"
            mesaj += f"   {n['icerik']}\n\n"
            buttons.append([InlineKeyboardButton(
                f"🗑️ {n['id']}. sil",
                callback_data=f"not_sil:{n['id']}"
            )])

        buttons.append([InlineKeyboardButton("✅ Tamam", callback_data="notlar_kapat")])
        await query.edit_message_text(mesaj, reply_markup=InlineKeyboardMarkup(buttons))

    # ⏰ HATIRLATMA EKLEME CALLBACK'İ
    elif data.startswith("hatirlatma_ekle:"):
        parts = data.split(":")
        not_id = int(parts[1])
        dakika = int(parts[2])

        # Kullanıcıyı kontrol et
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Not'u bul ve hatırlatma ekle
        from datetime import datetime, timedelta
        hatirlatma_zamani = datetime.now() + timedelta(minutes=dakika)

        # Not'a hatırlatma bilgisi ekle
        not_bulundu = False
        for n in asistan.not_manager.notes:
            if n.get('id') == not_id:
                n['hatirlatma'] = hatirlatma_zamani.isoformat()
                n['hatirlatma_gonderildi'] = False
                asistan.not_manager._save_notes()
                not_bulundu = True

                # JobQueue'ya ekle
                hatirlatma_job_ekle(context.application, user_id, n)

                # Süre formatla
                if dakika < 60:
                    sure_str = f"{dakika} dakika"
                else:
                    saat = dakika // 60
                    sure_str = f"{saat} saat"

                await query.edit_message_text(
                    f"✅ Not kaydedildi!\n\n"
                    f"📝 {n['icerik']}\n\n"
                    f"⏰ {sure_str} sonra hatırlatılacak\n"
                    f"🕐 {hatirlatma_zamani.strftime('%H:%M')}"
                )
                break

        if not not_bulundu:
            await query.edit_message_text("❌ Not bulunamadı.")

    # ⏰ HATIRLATMA İPTAL - Sadece not olarak kalsın
    elif data.startswith("hatirlatma_iptal"):
        # Not içeriğini göster
        not_icerik = ""
        not_tarih = ""
        not_saat = ""
        if ":" in data:
            not_id = int(data.split(":")[1])
            if user_id in user_instances:
                user = user_instances[user_id]
                for n in user["hafiza"].not_manager.notes:
                    if n.get('id') == not_id:
                        not_icerik = n['icerik']
                        not_tarih = n.get('tarih', '')
                        not_saat = n.get('saat', '')
                        break

        if not_icerik:
            await query.edit_message_text(f"✅ Not kaydedildi:\n\n📝 {not_icerik}\n📅 {not_tarih} - {not_saat}")
        else:
            await query.edit_message_text("✅ Not kaydedildi")

    # ↩️ HATIRLATMA GERİ AL
    elif data.startswith("hatirlatma_geri_al:"):
        not_id = int(data.split(":")[1])

        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Not'u bul ve hatırlatmayı kaldır
        for n in asistan.not_manager.notes:
            if n.get('id') == not_id:
                n.pop('hatirlatma', None)
                n.pop('hatirlatma_gonderildi', None)
                asistan.not_manager._save_notes()

                # JobQueue'dan da kaldır
                job_name = f"hatirlatma_{user_id}_{not_id}"
                jobs = context.job_queue.get_jobs_by_name(job_name)
                for job in jobs:
                    job.schedule_removal()

                # Yeniden süre girme seçeneği
                buttons = [[InlineKeyboardButton("🕐 Yeniden Hatırlatma Ekle", callback_data=f"hatirlatma_ozel:{not_id}")]]
                await query.edit_message_text(
                    f"↩️ Hatırlatma geri alındı.\n\n📝 {n['icerik']}",
                    reply_markup=InlineKeyboardMarkup(buttons)
                )
                return

        await query.edit_message_text("❌ Not bulunamadı.")

    # ⏰ ÖZEL SÜRE - Kullanıcıdan süre iste
    elif data.startswith("hatirlatma_ozel:"):
        not_id = int(data.split(":")[1])

        # Not ID'yi kaydet
        context.user_data["hatirlatma_ozel_not_id"] = not_id

        await query.edit_message_text(
            "⏰ *Özel süre gir:*\n\n"
            "Örnekler:\n"
            "• `40` veya `40 dk` → 40 dakika\n"
            "• `2 saat` veya `2 sa` → 2 saat\n"
            "• `1 saat 30 dk` → 1.5 saat\n"
            "• `90` → 90 dakika",
            parse_mode="Markdown"
        )

    # 📷 KAMERA CALLBACK'LERİ

    # Kamera ekle wizard başlat
    elif data == "kamera_ekle_wizard":
        # ⚠️ Diğer aktif modları temizle (belge, not vs.)
        clear_active_modes(user_id, context)

        user_kamera_wizard[user_id] = {
            "adim": "ad",
            "data": {}
        }

        # Önce mevcut mesajı güncelle
        await query.edit_message_text("Yeni Kamera Ekleme Başlatıldı")

        # Sonra ForceReply ile input iste
        await query.message.reply_text(
            "Adım 1/6: Kamera adı gir",
            reply_markup=ForceReply(input_field_placeholder="Örn: Bahçe Kamerası")
        )
        iptal_btn = InlineKeyboardMarkup([[InlineKeyboardButton("❌ İptal", callback_data="kamera_wizard_iptal")]])
        await query.message.reply_text("↩️", reply_markup=iptal_btn)

    # Kamera wizard iptal
    elif data == "kamera_wizard_iptal":
        if user_id in user_kamera_wizard:
            del user_kamera_wizard[user_id]
        await query.edit_message_text("Kamera ekleme iptal edildi.")

    # Kamera port seçimi
    elif data.startswith("kamera_port:"):
        if user_id not in user_kamera_wizard:
            await query.answer("Oturum sonlandı, tekrar başlat.")
            return

        port_val = data.split(":")[1]
        wizard = user_kamera_wizard[user_id]

        iptal_btn = InlineKeyboardMarkup([[InlineKeyboardButton("❌ İptal", callback_data="kamera_wizard_iptal")]])
        if port_val == "custom":
            # Kullanıcıdan custom port iste
            wizard["adim"] = "port"
            await query.message.reply_text(
                "Port numarasını gir:",
                reply_markup=ForceReply(input_field_placeholder="Örn: 554, 8554")
            )
            await query.message.reply_text("↩️", reply_markup=iptal_btn)
            await query.answer()
        else:
            # Seçilen portu kaydet
            wizard["data"]["port"] = int(port_val)
            wizard["adim"] = "kullanici"
            await query.edit_message_text(
                f"Port: {port_val}\n\n"
                "Adım 4/6: Kullanıcı adı",
            )
            await query.message.reply_text(
                "Kullanıcı adını gir:",
                reply_markup=ForceReply(input_field_placeholder="Örn: admin")
            )
            await query.message.reply_text("↩️", reply_markup=iptal_btn)

    # Kamera kanal seçimi
    elif data.startswith("kamera_kanal:"):
        if user_id not in user_kamera_wizard:
            await query.answer("Oturum sonlandı, tekrar başlat.")
            return

        kanal = int(data.split(":")[1])
        wizard = user_kamera_wizard[user_id]
        wizard_data = wizard["data"]
        wizard_data["kanal"] = kanal

        # Wizard tamamlandı - kamerayı kaydet
        kamera_manager = KameraManager(user_id)
        yeni_id = kamera_manager.kamera_ekle(
            ad=wizard_data["ad"],
            ip=wizard_data["ip"],
            port=wizard_data["port"],
            kullanici=wizard_data["kullanici"],
            sifre=wizard_data["sifre"],
            kanal=kanal
        )

        # Wizard'ı temizle
        del user_kamera_wizard[user_id]

        # RTSP URL (maskeli)
        rtsp_maskeli = kamera_manager.rtsp_url_maskeli(yeni_id)

        # Onay butonları
        keyboard = [
            [InlineKeyboardButton("Bağlantıyı Test Et", callback_data=f"kamera_test:{yeni_id}")],
            [InlineKeyboardButton("Şimdi Başlat", callback_data=f"kamera_baslat:{yeni_id}")],
            [InlineKeyboardButton("Kameralarım", callback_data="kameralarim")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"Kamera Eklendi!\n\n"
            f"Ad: {wizard_data['ad']}\n"
            f"IP: {wizard_data['ip']}\n"
            f"Kanal: {kanal}\n"
            f"URL: {rtsp_maskeli}",
            reply_markup=reply_markup
        )

    # Kameralarım listesi
    elif data == "kameralarim":
        # ⚠️ Diğer aktif modları temizle (belge, not vs.)
        clear_active_modes(user_id, context)

        kamera_manager = KameraManager(user_id)
        kameralar = kamera_manager.kamera_listele()

        if not kameralar:
            keyboard = [[InlineKeyboardButton("➕ Kamera Ekle", callback_data="kamera_ekle_wizard")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            mesaj_text = "📷 Henüz kamera eklememişsin.\n\nKamera eklemek için butona tıkla."

            if query.message.photo:
                await query.message.delete()
                await context.bot.send_message(chat_id=query.message.chat_id, text=mesaj_text, reply_markup=reply_markup)
            else:
                await query.edit_message_text(mesaj_text, reply_markup=reply_markup)
            return

        # Aktif kamera kontrolü
        aktif_kamera_id = None
        # Aktif kameraları bul (çoklu kamera desteği)
        aktif_kamera_idleri = set()
        if user_id in user_kamera_threads:
            for kid, kdata in user_kamera_threads[user_id].items():
                if kdata.get("aktif"):
                    aktif_kamera_idleri.add(kid)

        aktif_sayisi = len(aktif_kamera_idleri)
        mesaj = f"📷 Kameralarım ({len(kameralar)} adet)"
        if aktif_sayisi > 0:
            mesaj += f" - 🟢 {aktif_sayisi} aktif"
        mesaj += "\n\n"

        keyboard = []
        for k in kameralar:
            kamera_aktif = k["id"] in aktif_kamera_idleri
            durum = "🟢 AKTİF" if kamera_aktif else "⚫"
            mesaj += f"{k['id']}. {k['ad']} - {k['ip']}:{k['kanal']} {durum}\n"

            if kamera_aktif:
                keyboard.append([InlineKeyboardButton(
                    f"⏹️ {k['ad']} Durdur",
                    callback_data=f"kamera_durdur:{k['id']}"
                )])
            else:
                keyboard.append([
                    InlineKeyboardButton(f"▶️ Başlat", callback_data=f"kamera_baslat:{k['id']}"),
                    InlineKeyboardButton(f"🔍 Test", callback_data=f"kamera_test:{k['id']}"),
                    InlineKeyboardButton(f"🗑️ Sil", callback_data=f"kamera_sil:{k['id']}")
                ])

        # Tümünü Başlat / Tümünü Durdur butonları
        if len(kameralar) > 1:
            if aktif_sayisi < len(kameralar):
                keyboard.append([InlineKeyboardButton("▶️ Tümünü Başlat", callback_data="kamera_tumunu_baslat")])
            if aktif_sayisi > 0:
                keyboard.append([InlineKeyboardButton("⏹️ Tümünü Durdur", callback_data="kamera_tumunu_durdur")])

        keyboard.append([InlineKeyboardButton("➕ Yeni Kamera Ekle", callback_data="kamera_ekle_wizard")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Fotoğraflı mesajdan geliyorsa sil ve yeni mesaj gönder
        if query.message.photo:
            await query.message.delete()
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=mesaj,
                reply_markup=reply_markup
            )
        else:
            await query.edit_message_text(mesaj, reply_markup=reply_markup)

    # Kamera başlat
    elif data.startswith("kamera_baslat:"):
        kamera_id = int(data.split(":")[1])
        kamera_manager = KameraManager(user_id)
        kamera = kamera_manager.kamera_getir(kamera_id)

        if not kamera:
            await query.answer("⚠️ Kamera bulunamadı.", show_alert=True)
            return

        # Kullanıcı dict'i yoksa oluştur
        if user_id not in user_kamera_threads:
            user_kamera_threads[user_id] = {}

        # Bu kamera zaten aktif mi?
        if kamera_id in user_kamera_threads[user_id] and user_kamera_threads[user_id][kamera_id].get("aktif"):
            await query.answer("⚠️ Bu kamera zaten aktif!", show_alert=True)
            return

        # RTSP URL
        rtsp_url = kamera_manager.rtsp_url_olustur(kamera_id)

        # Thread başlat
        user_kamera_threads[user_id][kamera_id] = {
            "thread": None,
            "aktif": False,
            "stop_flag": False
        }

        thread = threading.Thread(
            target=kamera_izleme_baslat,
            args=(user_id, chat_id, rtsp_url, kamera_id, kamera["ad"]),
            daemon=True
        )
        user_kamera_threads[user_id][kamera_id]["thread"] = thread
        thread.start()

        # Durumu güncelle
        kamera_manager.kamera_durumu_guncelle(kamera_id, True)

        await query.answer(f"▶️ {kamera['ad']} başlatılıyor...")

        # Aktif kamera sayısı
        aktif_sayisi = sum(1 for k, v in user_kamera_threads[user_id].items() if v.get("aktif"))

        # Mesajı güncelle
        keyboard = [
            [InlineKeyboardButton(f"⏹️ {kamera['ad']} Durdur", callback_data=f"kamera_durdur:{kamera_id}")],
            [InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"📹 {kamera['ad']} başlatıldı!\n\n"
            f"🔗 {kamera['ip']}:{kamera['port']} (Kanal {kamera['kanal']})\n\n"
            "Hareket algılandığında bildirim alacaksın.",
            reply_markup=reply_markup
        )

    # Kamera durdur
    elif data.startswith("kamera_durdur:"):
        kamera_id = int(data.split(":")[1])

        if user_id not in user_kamera_threads or kamera_id not in user_kamera_threads[user_id]:
            await query.answer("⚠️ Bu kamera aktif değil.", show_alert=True)
            return

        # Durdur
        user_kamera_threads[user_id][kamera_id]["stop_flag"] = True

        kamera_manager = KameraManager(user_id)
        kamera = kamera_manager.kamera_getir(kamera_id)
        kamera_ad = kamera["ad"] if kamera else f"#{kamera_id}"

        await query.answer(f"⏹️ {kamera_ad} durduruluyor...")

        # Kameralarım listesine geri dön
        keyboard = [[InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"⏹️ {kamera_ad} durduruldu.",
            reply_markup=reply_markup
        )

    # Tümünü Başlat
    elif data == "kamera_tumunu_baslat":
        kamera_manager = KameraManager(user_id)
        kameralar = kamera_manager.kamera_listele()

        if not kameralar:
            await query.answer("⚠️ Kamera bulunamadı.", show_alert=True)
            return

        # Kullanıcı dict'i yoksa oluştur
        if user_id not in user_kamera_threads:
            user_kamera_threads[user_id] = {}

        baslatilanlar = []
        for kamera in kameralar:
            kamera_id = kamera["id"]

            # Zaten aktif mi?
            if kamera_id in user_kamera_threads[user_id] and user_kamera_threads[user_id][kamera_id].get("aktif"):
                continue

            rtsp_url = kamera_manager.rtsp_url_olustur(kamera_id)

            # Thread başlat
            user_kamera_threads[user_id][kamera_id] = {
                "thread": None,
                "aktif": False,
                "stop_flag": False
            }

            thread = threading.Thread(
                target=kamera_izleme_baslat,
                args=(user_id, chat_id, rtsp_url, kamera_id, kamera["ad"]),
                daemon=True
            )
            user_kamera_threads[user_id][kamera_id]["thread"] = thread
            thread.start()

            kamera_manager.kamera_durumu_guncelle(kamera_id, True)
            baslatilanlar.append(kamera["ad"])

        await query.answer(f"▶️ {len(baslatilanlar)} kamera başlatılıyor...")

        keyboard = [[InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"📹 Tüm kameralar başlatıldı!\n\n" +
            "\n".join([f"✅ {ad}" for ad in baslatilanlar]) +
            "\n\nHareket algılandığında bildirim alacaksın.",
            reply_markup=reply_markup
        )

    # Tümünü Durdur
    elif data == "kamera_tumunu_durdur":
        if user_id not in user_kamera_threads:
            await query.answer("⚠️ Aktif kamera yok.", show_alert=True)
            return

        durdurulanlar = []
        kamera_manager = KameraManager(user_id)

        for kamera_id, kdata in user_kamera_threads[user_id].items():
            if kdata.get("aktif"):
                kdata["stop_flag"] = True
                kamera = kamera_manager.kamera_getir(kamera_id)
                if kamera:
                    durdurulanlar.append(kamera["ad"])

        await query.answer(f"⏹️ {len(durdurulanlar)} kamera durduruluyor...")

        keyboard = [[InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"⏹️ Tüm kameralar durduruldu.\n\n" +
            "\n".join([f"⏹️ {ad}" for ad in durdurulanlar]),
            reply_markup=reply_markup
        )

    # Kamera sil
    elif data.startswith("kamera_sil:"):
        kamera_id = int(data.split(":")[1])
        kamera_manager = KameraManager(user_id)
        kamera = kamera_manager.kamera_getir(kamera_id)

        if not kamera:
            await query.answer("⚠️ Kamera bulunamadı.", show_alert=True)
            return

        # Aktif mi kontrol et
        if user_id in user_kamera_threads and kamera_id in user_kamera_threads[user_id]:
            if user_kamera_threads[user_id][kamera_id].get("aktif"):
                await query.answer("⚠️ Aktif kamera silinemez. Önce durdur.", show_alert=True)
                return

        # Onay iste
        keyboard = [
            [InlineKeyboardButton(f"✅ Evet, Sil", callback_data=f"kamera_sil_onayla:{kamera_id}")],
            [InlineKeyboardButton("❌ İptal", callback_data="kameralarim")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"🗑️ {kamera['ad']} silinecek.\n\n"
            f"🔗 {kamera['ip']}:{kamera['kanal']}\n\n"
            "Emin misin?",
            reply_markup=reply_markup
        )

    # Kamera sil onay
    elif data.startswith("kamera_sil_onayla:"):
        kamera_id = int(data.split(":")[1])
        kamera_manager = KameraManager(user_id)
        kamera = kamera_manager.kamera_getir(kamera_id)
        kamera_ad = kamera["ad"] if kamera else f"#{kamera_id}"

        # Sil
        if kamera_manager.kamera_sil(kamera_id):
            await query.answer(f"🗑️ {kamera_ad} silindi.")

            # Kameralarım listesine geri dön
            keyboard = [[InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"🗑️ {kamera_ad} silindi.",
                reply_markup=reply_markup
            )
        else:
            await query.answer("⚠️ Silme başarısız.", show_alert=True)

    # Kamera bağlantı testi
    elif data.startswith("kamera_test:"):
        kamera_id = int(data.split(":")[1])
        kamera_manager = KameraManager(user_id)
        kamera = kamera_manager.kamera_getir(kamera_id)

        if not kamera:
            await query.answer("⚠️ Kamera bulunamadı.", show_alert=True)
            return

        await query.answer("🔗 Test ediliyor...")
        await query.edit_message_text(f"🔗 {kamera['ad']} test ediliyor...\n\nBu işlem birkaç saniye sürebilir.")

        # RTSP URL
        rtsp_url = kamera_manager.rtsp_url_olustur(kamera_id)

        # Test fotoğrafı için path
        test_foto_path = f"user_data/user_{user_id}/kamera_test_{kamera_id}.jpg"
        os.makedirs(f"user_data/user_{user_id}", exist_ok=True)

        # Test et
        basarili, mesaj, foto_path = kamera_test_baglanti(rtsp_url, test_foto_path)

        # Başarısız olursa IP değişmiş olabilir - MAC ile ara
        ip_degisti = False
        if not basarili and kamera.get("mac"):
            await query.edit_message_text(
                f"🔗 {kamera['ad']} bağlantı başarısız.\n\n🔍 IP değişmiş olabilir, ağda aranıyor..."
            )
            yeni_ip = kamera_manager.ip_otomatik_bul(kamera_id)
            if yeni_ip and yeni_ip != kamera["ip"]:
                ip_degisti = True
                # Yeni IP ile tekrar dene
                rtsp_url = kamera_manager.rtsp_url_olustur(kamera_id)
                basarili, mesaj, foto_path = kamera_test_baglanti(rtsp_url, test_foto_path)
                if basarili:
                    mesaj = f"✅ Bağlantı başarılı!\n\n📍 IP güncellendi: {kamera['ip']} → {yeni_ip}"

        # Sonuç butonları
        if basarili:
            keyboard = [
                [InlineKeyboardButton("▶️ Şimdi Başlat", callback_data=f"kamera_baslat:{kamera_id}")],
                [InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]
            ]
        else:
            keyboard = [[InlineKeyboardButton("📋 Kameralarım", callback_data="kameralarim")]]

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Fotoğraf varsa gönder
        if basarili and foto_path and os.path.exists(foto_path):
            with open(foto_path, 'rb') as foto:
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=foto,
                    caption=f"📸 {kamera['ad']} - Test Görüntüsü\n\n{mesaj}",
                    reply_markup=reply_markup
                )
            # Eski mesajı sil
            await query.delete_message()
            # Test fotoğrafını sil
            os.remove(foto_path)
        else:
            await query.edit_message_text(
                f"🔗 {kamera['ad']} Bağlantı Testi\n\n{mesaj}",
                reply_markup=reply_markup
            )

    # 📍 KONUM MENU callback'i: kategorilere geri dön
    elif data == "konum_menu":
        # Kullanıcıyı kontrol et
        if user_id not in user_instances:
            await query.edit_message_text("❌ Önce botu başlat.")
            return

        # ⚠️ Diğer aktif modları temizle
        clear_active_modes(user_id, context)

        user = user_instances[user_id]
        asistan = user["hafiza"]

        # Konum kontrolü
        if not asistan.user_location:
            await query.edit_message_text("Konum bulunamadı. Tekrar konum paylaş.")
            return

        # Kısa adres
        kisa_adres = asistan.konum_adres if hasattr(asistan, 'konum_adres') and asistan.konum_adres else "Konumun"

        keyboard = []
        for i in range(0, len(KONUM_KATEGORILERI), 2):
            row = []
            row.append(InlineKeyboardButton(KONUM_KATEGORILERI[i][0], callback_data=f"konum_ara:{KONUM_KATEGORILERI[i][1]}"))
            if i + 1 < len(KONUM_KATEGORILERI):
                row.append(InlineKeyboardButton(KONUM_KATEGORILERI[i+1][0], callback_data=f"konum_ara:{KONUM_KATEGORILERI[i+1][1]}"))
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"📍 {kisa_adres}\n\nNe aramak istiyorsun?",
            reply_markup=reply_markup
        )

    # 💝 BAĞIŞ callback'leri
    elif data == "bagis_menu":
        text = """💝 *Projeyi Destekle*

Bu bot beta aşamasında ve kısıtlı donanımda çalışıyor.
Beğendiysen ve devam etmesini istiyorsan, sunucu altyapısı için bağış yapabilirsin.

⭐ Telegram Stars ile bağış yapabilirsin.
"""
        keyboard = [
            [InlineKeyboardButton("⭐ 10 Stars", callback_data="bagis_10")],
            [InlineKeyboardButton("⭐ 25 Stars", callback_data="bagis_25")],
            [InlineKeyboardButton("⭐ 50 Stars", callback_data="bagis_50")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

    elif data.startswith("bagis_"):
        # Telegram Stars ile bağış gönder
        star_miktari = data.split("_")[1]
        if star_miktari == "custom":
            await query.edit_message_text(
                "⭐ Özel miktar için /bagis komutunu kullan.",
                parse_mode="Markdown"
            )
            return

        miktar = int(star_miktari)

        # Telegram Stars invoice gönder
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Proje Desteği",
                description=f"Bot geliştirme ve sunucu altyapısı için {miktar} Stars bağış",
                payload=f"bagis_{user_id}_{miktar}",
                provider_token="",  # Telegram Stars için boş
                currency="XTR",     # Telegram Stars para birimi
                prices=[{"label": "Bağış", "amount": miktar}],
            )
            await query.edit_message_text(
                f"⭐ *{miktar} Stars bağış faturası gönderildi!*\n\n"
                "Ödeme butonuna tıklayarak bağışını tamamlayabilirsin.\n\n"
                "💝 Desteğin için şimdiden teşekkürler!",
                parse_mode="Markdown"
            )
        except Exception as e:
            print(f"[HATA] Bağış invoice hatası: {e}")
            await query.edit_message_text(
                "❌ Bağış sistemi şu anda kullanılamıyor.\n"
                "Lütfen daha sonra tekrar dene.",
                parse_mode="Markdown"
            )

    elif data == "limit_info":
        # Admin sınırsız
        if user_id in ADMIN_IDS:
            await query.edit_message_text("👑 *ADMIN* - Tüm limitler sınırsız!", parse_mode="Markdown")
            return

        # Limit bilgilerini göster
        db = get_db()
        rate_check = db.check_rate_limit(user_id)
        camera_check = db.check_camera_limit(user_id)
        location_check = db.check_location_limit(user_id)
        image_check = db.check_image_limit(user_id)

        text = f"""📊 *Günlük Limitler*

💬 Mesaj: *{rate_check['remaining']}/{rate_check['limit']}*
🎨 Görsel üretimi: *{image_check['remaining']}/{image_check['limit']}*
📷 Kamera bildirimi: *{camera_check['remaining']}/{camera_check['limit']}*
📍 Konum sorgusu: *{location_check['remaining']}/{location_check['limit']}*

_Limitler gece 00:00'da sıfırlanır._
"""
        keyboard = [[InlineKeyboardButton("💝 Bağış Yap", callback_data="bagis_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")


# === TELEGRAM STARS ÖDEME HANDLERLARİ ===

async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ödeme öncesi doğrulama - Telegram Stars için"""
    query = update.pre_checkout_query

    # Bağış payload'ını kontrol et
    if query.invoice_payload.startswith("bagis_"):
        # Bağışı kabul et
        await query.answer(ok=True)
    else:
        # Bilinmeyen payload
        await query.answer(ok=False, error_message="Geçersiz ödeme.")


async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Başarılı ödeme sonrası işlem"""
    payment = update.message.successful_payment
    user_id = update.effective_user.id

    # Payload'dan bilgileri al
    payload = payment.invoice_payload  # bagis_userid_miktar
    parts = payload.split("_")

    if len(parts) >= 3 and parts[0] == "bagis":
        miktar = parts[2]

        # Ödemeyi kaydet
        db = get_db()
        db.record_payment(
            user_id=user_id,
            plan=PlanType.FREE,  # Bağış, plan değil
            amount_tl=float(miktar),  # Stars miktarı
            payment_method="telegram_stars",
            transaction_id=payment.telegram_payment_charge_id
        )

        await update.message.reply_text(
            f"💝 *Teşekkürler!*\n\n"
            f"⭐ {miktar} Stars bağışın başarıyla alındı!\n\n"
            f"Desteğin sayesinde bu proje gelişmeye devam edecek. 🙏\n\n"
            f"_İşlem ID: {payment.telegram_payment_charge_id[:20]}..._",
            parse_mode="Markdown"
        )

        # Admin'e bildir
        for admin_id in ADMIN_IDS:
            try:
                await context.bot.send_message(
                    admin_id,
                    f"💝 *Yeni Bağış!*\n\n"
                    f"👤 Kullanıcı: {user_id}\n"
                    f"⭐ Miktar: {miktar} Stars\n"
                    f"🆔 İşlem: {payment.telegram_payment_charge_id}",
                    parse_mode="Markdown"
                )
            except:
                pass


# === MAIN ===

def main():
    print("=" * 50)
    print("Telegram Bot Baslatiliyor...")
    print("=" * 50)

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("[HATA] TELEGRAM_TOKEN bulunamadi!")
        return

    # Telegram menüsüne komutları ekle
    async def post_init(application):
        try:
            # Menüyü ayarla
            komutlar = [
                BotCommand("yeni", "🔄 Yeni sohbet"),
                BotCommand("konum", "📍 Konum paylaş"),
                BotCommand("notdefteri", "📝 Not Defteri"),
                BotCommand("belgelerim", "📄 Çalışma Alanım"),
                BotCommand("kameralarim", "📷 Kamera yönetimi"),
                BotCommand("limit", "📊 Günlük limitler"),
                BotCommand("bagis", "💝 Projeyi destekle")
            ]
            await application.bot.set_my_commands(komutlar)
            print("[OK] Telegram menusu ayarlandi!")

            # Mevcut hatırlatmaları yükle
            await mevcut_hatirlatmalari_yukle(application)
        except Exception as e:
            print(f"[HATA] Menu hatasi: {e}")

    # HTTPXRequest ile timeout ayarları (default 5sn çok kısa)
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=10.0
    )
    app = Application.builder().token(token).request(request).post_init(post_init).build()

    # GLOBAL ERROR HANDLER
    async def error_handler(update, context):
        print("=" * 50)
        print("[HATA] GLOBAL HATA YAKALANDI!")
        print(f"   Hata: {context.error}")
        print(f"   Update: {update}")
        import traceback
        traceback.print_exc()
        print("=" * 50)

    app.add_error_handler(error_handler)

    # Komutlar
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("yeni", yeni_command))
    app.add_handler(CommandHandler("konum", konum_command))
    app.add_handler(CommandHandler("notdefteri", notdefteri_command))
    app.add_handler(CommandHandler("belgelerim", belgelerim_command))
    app.add_handler(CommandHandler("limit", limit_command))
    app.add_handler(CommandHandler("bagis", bagis_command))
    app.add_handler(CommandHandler("premium", premium_command))  # Eski uyumluluk

    # Kamera komutları (multi-user)
    app.add_handler(CommandHandler("kamera_ekle", kamera_ekle_command))
    app.add_handler(CommandHandler("kameralarim", kameralarim_command))
    app.add_handler(CommandHandler("kamera", kamera_baslat_command))
    app.add_handler(CommandHandler("kamerakapat", kamera_durdur_command))

    # Mesaj
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Konum
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    # Fotograf
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Belge/Dosya (PDF, DOCX, TXT)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Callback (inline butonlar)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Telegram Stars ödeme handler'ları
    app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))

    print("[OK] Bot hazir!")
    print("=" * 50)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
