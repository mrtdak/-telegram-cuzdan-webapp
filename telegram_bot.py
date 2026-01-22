"""
Telegram Bot - Arayüz

Akış:
Telegram → HafizaAsistani.prepare() → PersonalAI.generate() → HafizaAsistani.save() → Telegram
"""

import os
import asyncio
import math
import aiohttp
from dotenv import load_dotenv
from telegram import Update, BotCommand, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Dict, Tuple, Optional

from hafiza_asistani import HafizaAsistani
from yazar_asistani import YazarAsistani
from personal_ai import PersonalAI
import re

load_dotenv()


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

# Kabe koordinatları
KABE_LAT = 21.4225
KABE_LON = 39.8262

def hesapla_kible_yonu(lat: float, lon: float) -> Tuple[float, str]:
    """
    Verilen koordinattan Kabe'ye kıble yönünü hesapla.

    Returns:
        (açı_derece, yön_metni)
    """
    # Radyana çevir
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.radians(KABE_LAT)
    lon2 = math.radians(KABE_LON)

    # Kıble açısı hesaplama (bearing formula)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360  # 0-360 arası normalize

    # Yön metni
    yonler = [
        (0, "Kuzey"), (45, "Kuzeydoğu"), (90, "Doğu"), (135, "Güneydoğu"),
        (180, "Güney"), (225, "Güneybatı"), (270, "Batı"), (315, "Kuzeybatı"), (360, "Kuzey")
    ]

    yon_metni = "Kuzey"
    for aci, yon in yonler:
        if bearing >= aci - 22.5 and bearing < aci + 22.5:
            yon_metni = yon
            break

    return bearing, yon_metni


def hesapla_mesafe(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    İki koordinat arası mesafe (Haversine formülü).

    Returns:
        Mesafe (km)
    """
    R = 6371  # Dünya yarıçapı (km)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


async def adres_cozumle(lat: float, lon: float) -> Optional[str]:
    """
    Koordinattan adres çözümle (Reverse Geocoding - Nominatim).
    Cadde/sokak bilgisi dahil detaylı adres döndürür.
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "accept-language": "tr",
            "addressdetails": 1
        }
        headers = {"User-Agent": "PersonalAI-TelegramBot/1.0"}

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Detaylı adres bilgisi varsa, özel format oluştur
                    address = data.get("address", {})
                    if address:
                        parts = []

                        # Cadde/Sokak
                        road = address.get("road") or address.get("street") or address.get("pedestrian")
                        if road:
                            parts.append(road)

                        # Mahalle
                        mahalle = address.get("suburb") or address.get("neighbourhood") or address.get("quarter")
                        if mahalle:
                            parts.append(mahalle)

                        # İlçe
                        ilce = address.get("town") or address.get("district") or address.get("county")
                        if ilce:
                            parts.append(ilce)

                        # İl
                        il = address.get("city") or address.get("province") or address.get("state")
                        if il:
                            parts.append(il)

                        parts.append("Türkiye")

                        if parts:
                            return ", ".join(parts)

                    # Fallback: display_name
                    return data.get("display_name", "Adres bulunamadı")
    except Exception as e:
        print(f"Adres çözümleme hatası: {e}")
    return None


async def hava_durumu_koordinat(lat: float, lon: float) -> str:
    """Koordinata göre hava durumu (wttr.in)"""
    try:
        url = f"https://wttr.in/{lat},{lon}?format=j1&lang=tr"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return "❌ Hava durumu alınamadı."
                data = await resp.json()

        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        sehir = area.get("areaName", [{}])[0].get("value", "Bilinmeyen")

        desc_list = current.get("lang_tr", [])
        if desc_list:
            description = desc_list[0].get("value", current["weatherDesc"][0]["value"])
        else:
            description = current["weatherDesc"][0]["value"]

        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]

        return (
            f"🌤️ {sehir} Hava Durumu\n"
            f"{'─' * 28}\n"
            f"☁️ Durum: {description}\n"
            f"🌡️ Sıcaklık: {temp}°C\n"
            f"🤚 Hissedilen: {feels}°C\n"
            f"💧 Nem: {humidity}%"
        )
    except Exception as e:
        print(f"Hava durumu hatası: {e}")
        return "❌ Hava durumu alınamadı."


async def namaz_vakti_koordinat(lat: float, lon: float) -> str:
    """Koordinata göre namaz vakitleri (Aladhan API)"""
    try:
        url = "http://api.aladhan.com/v1/timings"
        params = {
            "latitude": lat,
            "longitude": lon,
            "method": 13  # Diyanet metodu
        }

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return "❌ Namaz vakitleri alınamadı."
                data = await resp.json()

        if data.get("code") != 200:
            return "❌ Namaz vakitleri alınamadı."

        timings = data["data"]["timings"]

        prayer_names = {
            "Fajr": ("İmsak", "🌙"),
            "Sunrise": ("Güneş", "☀️"),
            "Dhuhr": ("Öğle", "🌤️"),
            "Asr": ("İkindi", "🌅"),
            "Maghrib": ("Akşam", "🌆"),
            "Isha": ("Yatsı", "🌃"),
        }

        result = f"🕌 Namaz Vakitleri\n{'─' * 28}\n\n"
        for eng_name, (turkish_name, emoji) in prayer_names.items():
            time_value = timings[eng_name]
            result += f"{emoji} {turkish_name:<8} {time_value}\n"

        return result.strip()
    except Exception as e:
        print(f"Namaz vakti hatası: {e}")
        return "❌ Namaz vakitleri alınamadı."


# Kullanıcı son konumları (mesafe hesaplama için)
user_last_location: Dict[int, Tuple[float, float]] = {}

# Kullanıcı izolasyonu: Her kullanıcının kendi AI'ı
user_instances: Dict[int, Dict] = {}
TIMEOUT = 120

# 🔒 İZİNLİ KULLANICILAR (tüm özelliklere erişim)
ALLOWED_USERS = [6505503887, 5007922833]  # Murat + Eşi


def is_allowed(user_id: int) -> bool:
    """Kullanıcının botu kullanma izni var mı?"""
    return user_id in ALLOWED_USERS


def get_user_ai(user_id: int) -> Dict:
    """Kullanıcı için HafizaAsistani + YazarAsistani + PersonalAI al (izole)"""
    if user_id not in user_instances:
        user_str = f"user_{user_id}"

        # HafizaAsistani - Sohbet modu (prompt hazırlar, hafıza tutar)
        hafiza = HafizaAsistani(user_id=user_str)

        # YazarAsistani - Yazar modu (QuantumTree karakteri)
        yazar = YazarAsistani(user_id=user_str)

        # PersonalAI - Ağız (cevap üretir)
        ai = PersonalAI(user_id=user_str)

        user_instances[user_id] = {
            "hafiza": hafiza,
            "yazar": yazar,
            "ai": ai,
            "aktif_mod": "normal",  # "normal" veya "yazar"
            "firlama_modu": False   # 🚀 Fırlama modu (kapalı başlar)
        }
        print(f"🆕 Yeni kullanıcı: {user_id}")

    return user_instances[user_id]


# === KOMUTLAR ===

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start - Herkese açık"""
    user_id = update.effective_user.id
    get_user_ai(user_id)

    # İzinli kullanıcıya tam menü, diğerlerine sadece sohbet
    if is_allowed(user_id):
        await update.message.reply_text(
            "🤖 Merhaba!\n\n"
            "📌 Modlar:\n"
            "/normal - 💬 Sohbet modu\n"
            "/yazar - ✍️ QuantumTree yazar modu\n"
            "/komedi - 😂 Komedi yazarı modu\n\n"
            "📍 Konum:\n"
            "/konum - Konum hizmetleri rehberi\n"
            "📎 → Konum gönder = Hava + Namaz + Kıble\n\n"
            "⚙️ Ayarlar:\n"
            "/yeni - Hafızayı sıfırla\n"
            "/firlama - 🚀 Fırlama modu aç/kapat"
        )
    else:
        await update.message.reply_text(
            "🤖 Merhaba!\n\n"
            "💬 Sohbet modundasın. Bana bir şey sor!\n\n"
            "/yeni - Hafızayı sıfırla"
        )


async def yeni_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/yeni - Hafızayı sıfırla (herkese açık)"""
    user_id = update.effective_user.id
    user = get_user_ai(user_id)
    user["hafiza"].clear()
    user["yazar"].clear()
    await update.message.reply_text("✅ Hafıza sıfırlandı!")


async def firlama_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/firlama - Fırlama modunu aç/kapat"""
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return

    print(f"🚀 /firlama komutu alındı! User: {user_id}")
    user = get_user_ai(user_id)

    # Toggle
    user["firlama_modu"] = not user["firlama_modu"]
    print(f"   Fırlama modu: {user['firlama_modu']}")

    if user["firlama_modu"]:
        await update.message.reply_text("🚀 FIRLAMA MODU AKTİF!\nEnerjik, şakacı, rekabetçi mod açıldı!")
    else:
        await update.message.reply_text("😌 Fırlama modu kapatıldı.\nNormal moda dönüldü.")


async def yazar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/yazar - QuantumTree yazar moduna geç"""
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return

    print(f"✍️ /yazar komutu alındı! User: {user_id}")
    user = get_user_ai(user_id)
    user["aktif_mod"] = "yazar"

    await update.message.reply_text(
        "✍️ YAZAR MODU: QuantumTree\n\n"
        "Bilim kurgu ve gerilim yazarı aktif.\n"
        "Bana bir konu, karakter veya sahne ver - yazayım.\n\n"
        "Normal moda dönmek için: /normal"
    )


async def normal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/normal - Normal sohbet moduna dön (herkese açık)"""
    user_id = update.effective_user.id
    print(f"💬 /normal komutu alındı! User: {user_id}")

    user = get_user_ai(user_id)
    user["aktif_mod"] = "normal"

    if is_allowed(user_id):
        await update.message.reply_text(
            "💬 NORMAL MOD\n\n"
            "Sohbet asistanı aktif.\n"
            "Yazar moduna geçmek için: /yazar"
        )
    else:
        await update.message.reply_text(
            "💬 NORMAL MOD\n\n"
            "Sohbet asistanı aktif."
        )


async def komedi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/komedi - Yazar modunda komedi türünü aktifle"""
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return

    print(f"😂 /komedi komutu alındı! User: {user_id}")
    user = get_user_ai(user_id)

    # Yazar moduna geç ve komedi türünü aktifle
    user["aktif_mod"] = "yazar"
    user["yazar"].set_tur("komedi")

    await update.message.reply_text(
        "😂 KOMEDİ MODU AKTİF!\n\n"
        "QuantumTree şimdi komedi yazarı.\n"
        "Kahkaha bol, eğlence dolu hikayeler!\n\n"
        "Normal moda dönmek için: /normal"
    )


async def konum_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/konum - Konum paylaşma rehberi"""
    await update.message.reply_text(
        "📍 KONUM HİZMETLERİ\n"
        "═" * 28 + "\n\n"
        "Konum paylaşınca şunları alırsın:\n\n"
        "🌤️ Hava durumu\n"
        "🕌 Namaz vakitleri\n"
        "🧭 Kıble yönü ve açısı\n"
        "🕋 Kabe'ye mesafe\n"
        "🗺️ Adres bilgisi\n"
        "📏 Önceki konumdan mesafe\n\n"
        "📎 Konum göndermek için:\n"
        "Ataç simgesi → Konum → Gönder"
    )


# === KONUM HANDLER ===

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📍 Konum mesajı handler - LLM ENTEGRASYONLU"""
    try:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        location = update.message.location

        if location is None:
            await update.message.reply_text("❌ Konum bilgisi alınamadı.")
            return

        lat = location.latitude
        lon = location.longitude
        print(f"📍 Konum alındı: {lat:.4f}, {lon:.4f}")

        # Kaydet
        user_last_location[user_id] = (lat, lon)
        user = get_user_ai(user_id)

        # Düşünüyorum mesajı
        status = await context.bot.send_message(chat_id, "📍 Konumunu alıyorum...")

        # Adres çözümle
        try:
            adres = await adres_cozumle(lat, lon)
            if not adres:
                adres = f"{lat:.4f}, {lon:.4f}"
        except:
            adres = f"{lat:.4f}, {lon:.4f}"

        try:
            # LLM'e gönder
            asistan = user["hafiza"]
            ai = user["ai"]

            # Konum alındı mesajı hazırla
            result = await asistan.prepare_konum_alindi(lat, lon, adres)
            messages = result["messages"]

            # LLM'den cevap al
            response = await asyncio.wait_for(
                ai.generate(messages=messages),
                timeout=TIMEOUT
            )

            # Temizle
            response = temizle_cikti(response)

            # Kaydet (konum bilgisi olarak)
            asistan.save(f"[Konum paylaşıldı: {adres}]", response, [])

            # Düşünüyorum mesajını sil
            await status.delete()

            # Cevabı gönder
            await update.message.reply_text(
                response,
                reply_markup=ReplyKeyboardRemove()
            )

        except asyncio.TimeoutError:
            await status.delete()
            await update.message.reply_text(
                f"📍 Konum alındı: {adres}\n\n"
                "⏱️ Cevap zaman aşımına uğradı. Ne bilmek istersen sor!",
                reply_markup=ReplyKeyboardRemove()
            )

    except Exception as e:
        print(f"❌ Konum hatası: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text("❌ Konum işlenirken hata oluştu.")


# === MESAJ HANDLER ===

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ana akış:
    1. Telegram mesaj alır
    2. Aktif moda göre asistan seç (normal/yazar)
    3. Asistan.prepare() → messages hazırlar
    4. PersonalAI.generate() → cevap üretir
    5. Asistan.save() → hafızaya kaydeder
    6. Telegram'a cevap gönderir
    """
    user_id = update.effective_user.id
    user_input = update.message.text
    chat_id = update.effective_chat.id

    # 📍 KONUM İSTE - "konum gönder" pattern'i algıla (yazım hatası toleranslı)
    user_lower = user_input.lower().strip()

    # Yazım hatası toleranslı konum kontrolü
    def konum_istegi_mi(text):
        words = text.split()
        if len(words) < 2:
            return False
        konum_var = any(w.startswith('konum') for w in words)
        aksiyon_patterns = ['gön', 'gon', 'payla', 'iste', ' at', ' ver']
        aksiyon_var = any(p in text for p in aksiyon_patterns)
        return konum_var and aksiyon_var

    if konum_istegi_mi(user_lower):
        print(f"📍 Konum butonu gönderiliyor: '{user_input}'")
        keyboard = ReplyKeyboardMarkup(
            [[KeyboardButton("📍 Konumumu Paylaş", request_location=True)]],
            resize_keyboard=True,
            one_time_keyboard=True
        )
        await update.message.reply_text(
            "📍 Konum paylaşmak için aşağıdaki butona bas:",
            reply_markup=keyboard
        )
        return

    # Kullanıcının AI'larını al
    user = get_user_ai(user_id)
    aktif_mod = user.get("aktif_mod", "normal")

    # 🔒 Yazar modu sadece izinli kullanıcılara
    if aktif_mod == "yazar" and not is_allowed(user_id):
        user["aktif_mod"] = "normal"  # Normal moda zorla
        aktif_mod = "normal"

    # Düşünüyorum mesajı
    if aktif_mod == "yazar":
        status = await context.bot.send_message(chat_id, "✍️ Yazıyorum...")
    else:
        status = await context.bot.send_message(chat_id, "💭 Düşünüyorum...")

    try:
        # Kullanıcının AI'larını al
        ai = user["ai"]
        firlama_modu = user.get("firlama_modu", False)

        # Aktif moda göre asistan seç
        if aktif_mod == "yazar":
            # YAZAR MODU - YazarAsistani kullan
            asistan = user["yazar"]
            result = asistan.prepare(user_input)
            messages = result["messages"]

            # Cevap üret
            response = await asyncio.wait_for(
                ai.generate(messages=messages),
                timeout=TIMEOUT
            )

            # Yazar modunda temizleme yapma - yaratıcı yazı olduğu gibi kalsın
            # response = temizle_cikti(response)

            # Kaydet
            asistan.save(user_input, response)

        else:
            # NORMAL MOD - HafizaAsistani kullan
            asistan = user["hafiza"]
            result = await asyncio.wait_for(
                asistan.prepare(user_input, [], firlama_modu=firlama_modu),
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

            # 📝 Direct response kontrolü (not sistemi, konum araçları vs.)
            if paket.get("direct_response"):
                response = paket["direct_response"]
                # Araç sonucunu history'e kaydet (LLM bağlamı korusun)
                tool_used = paket.get("tool_used", "")
                if tool_used in ["konum_hizmeti", "not_sistemi"]:
                    asistan.save(user_input, response, [])
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
        print(f"❌ Hata: {e}")
        response = "❌ Bir sorun oluştu."

    # Status mesajını sil
    try:
        await context.bot.delete_message(chat_id, status.message_id)
    except:
        pass

    # Cevabı gönder
    await update.message.reply_text(response)


# === MAIN ===

def main():
    print("=" * 50)
    print("🚀 Telegram Bot")
    print("=" * 50)

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ TELEGRAM_TOKEN bulunamadı!")
        return

    app = Application.builder().token(token).build()

    # 🔴 GLOBAL ERROR HANDLER
    async def error_handler(update, context):
        print("=" * 50)
        print("🔴 GLOBAL HATA YAKALANDI!")
        print(f"   Hata: {context.error}")
        print(f"   Update: {update}")
        import traceback
        traceback.print_exc()
        print("=" * 50)

    app.add_error_handler(error_handler)

    # Komutlar
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("yeni", yeni_command))
    app.add_handler(CommandHandler("firlama", firlama_command))
    app.add_handler(CommandHandler("yazar", yazar_command))
    app.add_handler(CommandHandler("normal", normal_command))
    app.add_handler(CommandHandler("komedi", komedi_command))
    app.add_handler(CommandHandler("konum", konum_command))

    # Mesaj
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 📍 Konum
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    # Telegram menüsüne komutları ekle
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start", "Botu başlat"),
            BotCommand("yeni", "Hafızayı sıfırla"),
            BotCommand("konum", "📍 Konum hizmetleri"),
            BotCommand("firlama", "🚀 Fırlama modunu aç/kapat"),
            BotCommand("yazar", "✍️ QuantumTree yazar modu"),
            BotCommand("normal", "💬 Normal sohbet modu"),
            BotCommand("komedi", "😂 Komedi yazarı modu")
        ])
        print("✅ Telegram menüsü güncellendi!")

    app.post_init = post_init

    print("✅ Bot hazır!")
    print("=" * 50)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
