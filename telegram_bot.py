"""
Telegram Bot - Arayüz

Akış:
Telegram → HafizaAsistani.prepare() → PersonalAI.generate() → HafizaAsistani.save() → Telegram
"""

import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Dict

from hafiza_asistani import HafizaAsistani
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
    yasak_pattern = r',?\s*(ne dersin\??|değil mi\??|kim bilir\??|nasıl fikir\??|sence\??)\s*$'
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

# Kullanıcı izolasyonu: Her kullanıcının kendi AI'ı
user_instances: Dict[int, Dict] = {}
TIMEOUT = 120


def get_user_ai(user_id: int) -> Dict:
    """Kullanıcı için HafizaAsistani + PersonalAI al (izole)"""
    if user_id not in user_instances:
        user_str = f"user_{user_id}"

        # HafizaAsistani - Beyin (prompt hazırlar, hafıza tutar)
        hafiza = HafizaAsistani(user_id=user_str)

        # PersonalAI - Ağız (cevap üretir)
        ai = PersonalAI(user_id=user_str)

        user_instances[user_id] = {
            "hafiza": hafiza,
            "ai": ai
        }
        print(f"🆕 Yeni kullanıcı: {user_id}")

    return user_instances[user_id]


# === KOMUTLAR ===

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start"""
    user_id = update.effective_user.id
    get_user_ai(user_id)

    await update.message.reply_text(
        "🤖 Merhaba!\n\n"
        "Komutlar:\n"
        "/yeni - Hafızayı sıfırla"
    )


async def yeni_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/yeni - Hafızayı sıfırla"""
    user_id = update.effective_user.id
    user = get_user_ai(user_id)
    user["hafiza"].clear()
    await update.message.reply_text("✅ Hafıza sıfırlandı!")


# === MESAJ HANDLER ===

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ana akış:
    1. Telegram mesaj alır
    2. HafizaAsistani.prepare() → messages hazırlar
    3. PersonalAI.generate() → cevap üretir
    4. HafizaAsistani.save() → hafızaya kaydeder
    5. Telegram'a cevap gönderir
    """
    user_input = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Düşünüyorum mesajı
    status = await context.bot.send_message(chat_id, "💭 Düşünüyorum...")

    try:
        # Kullanıcının AI'larını al
        user = get_user_ai(user_id)
        hafiza = user["hafiza"]
        ai = user["ai"]

        # 1. HafizaAsistani prompt hazırlasın
        result = await asyncio.wait_for(
            hafiza.prepare(user_input, []),
            timeout=TIMEOUT
        )
        messages = result["messages"]

        # 2. PersonalAI cevap üretsin
        response = await asyncio.wait_for(
            ai.generate(messages=messages),
            timeout=TIMEOUT
        )

        # 3. Çıktıyı temizle (markdown + yasak ifadeler)
        response = temizle_cikti(response)

        # 4. HafizaAsistani hafızaya kaydetsin
        hafiza.save(user_input, response, [])

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

    # Komutlar
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("yeni", yeni_command))

    # Mesaj
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot hazır!")
    print("=" * 50)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
