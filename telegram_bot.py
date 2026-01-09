"""
Telegram Bot - Basit Arayüz
Kullanıcı mesaj yazar → HafizaAsistani/QuantumTree → Cevap
"""

import sys
import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Dict, Optional

# Modülleri yükle
try:
    from personal_ai import LocalLLM
    from hafiza_asistani import HafizaAsistani
    print("✅ HafizaAsistani yüklendi")
except ImportError as e:
    print(f"❌ HafizaAsistani yüklenemedi: {e}")
    sys.exit(1)

try:
    from quantum_agac import QuantumTree
    quantum_available = True
    print("✅ QuantumTree yüklendi")
except ImportError:
    quantum_available = False
    print("⚠️ QuantumTree yüklenemedi - sadece basit mod aktif")

load_dotenv()

# Global
ai_instances: Dict[int, "AIWrapper"] = {}
TIMEOUT = 120


class AIWrapper:
    """Basit AI Wrapper - İki mod: basit ve derin"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mode = "basit"  # varsayılan

        # Basit mod: HafizaAsistani + LLM
        self.llm = LocalLLM(user_id)
        self.hafiza = HafizaAsistani(
            user_id=user_id,  # Dinamik kullanıcı ID
            saat_limiti=48,
            esik=0.50,
            max_mesaj=20,
            model_adi="BAAI/bge-m3",
            use_decision_llm=True,
            decision_model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        )
        self.hafiza.set_llm(self.llm)

        # Derin mod: QuantumTree
        self.quantum = None
        if quantum_available:
            try:
                self.quantum = QuantumTree(
                    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
                    neo4j_password=os.getenv("NEO4J_PASS", ""),
                    thinking_framework_path="thinking_framework.json"
                )
            except Exception as e:
                print(f"⚠️ QuantumTree başlatılamadı: {e}")

        print(f"✅ AIWrapper hazır (user: {user_id})")

    async def process(self, user_input: str) -> str:
        """Mesajı işle - moda göre"""
        try:
            if self.mode == "derin" and self.quantum:
                # Derin mod: QuantumTree
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.quantum.truth_filter, user_input
                )
                return result.get("final_response", "QuantumTree yanıt üretemedi.")
            else:
                # Basit mod: HafizaAsistani
                response = await asyncio.wait_for(
                    self.hafiza.process(user_input, []),
                    timeout=TIMEOUT
                )
                return response

        except asyncio.TimeoutError:
            return "⏱️ Zaman aşımı, tekrar dene."
        except Exception as e:
            print(f"❌ Hata: {e}")
            return "❌ Bir sorun oluştu."

    def reset(self):
        """Hafızayı sıfırla"""
        if hasattr(self.hafiza, 'hafiza'):
            self.hafiza.hafiza = []
        return "✅ Sıfırlandı"

    async def summarize_and_save(self):
        """Sohbeti özetle ve profile kaydet"""
        if not hasattr(self.hafiza, 'hafiza') or len(self.hafiza.hafiza) < 2:
            return  # Yeterli sohbet yok

        # Sohbet geçmişini al
        chat_text = ""
        for msg in self.hafiza.hafiza[-20:]:  # Son 20 mesaj
            rol = "Kullanıcı" if msg.get("rol") == "user" else "AI"
            chat_text += f"{rol}: {msg.get('mesaj', '')}\n"

        if not chat_text.strip():
            return

        # LLM'e özet sorgusu
        summary_prompt = f"""Bu sohbeti analiz et ve şu bilgileri çıkar:

1. ÖZET: Sohbetin 1-2 cümlelik özeti
2. YENİ BİLGİLER: Kullanıcı hakkında öğrenilen yeni bilgiler (isim, meslek, ilgi alanı, önemli gerçekler)

Sohbet:
{chat_text}

Sadece JSON formatında cevap ver:
{{"ozet": "...", "yeni_bilgiler": ["bilgi1", "bilgi2"]}}"""

        try:
            response = await self.llm.generate(summary_prompt)

            # JSON parse et
            import json
            import re

            # JSON kısmını bul
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Profile kaydet
                if hasattr(self.hafiza, 'profile_manager'):
                    pm = self.hafiza.profile_manager

                    # Özeti kaydet
                    if data.get('ozet'):
                        pm.update_last_session(data['ozet'])
                        print(f"📝 Sohbet özeti kaydedildi: {data['ozet'][:50]}...")

                    # Yeni bilgileri ekle
                    for bilgi in data.get('yeni_bilgiler', []):
                        if bilgi and len(bilgi) > 3:
                            pm.add_important_fact(bilgi)
                            print(f"💡 Yeni bilgi eklendi: {bilgi}")

        except Exception as e:
            print(f"⚠️ Özet çıkarma hatası: {e}")


def get_ai(user_id: int) -> AIWrapper:
    """Kullanıcı için AI instance al"""
    if user_id not in ai_instances:
        ai_instances[user_id] = AIWrapper(f"user_{user_id}")
    return ai_instances[user_id]


# === KOMUTLAR ===

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start"""
    user_id = update.effective_user.id
    ai = get_ai(user_id)

    mode_text = "🧠 Derin (QuantumTree)" if ai.mode == "derin" else "⚡ Basit (HafizaAsistani)"
    quantum_status = "✅" if ai.quantum else "❌"

    await update.message.reply_text(
        f"🤖 Bot Hazır!\n\n"
        f"Mod: {mode_text}\n"
        f"QuantumTree: {quantum_status}\n\n"
        f"Komutlar:\n"
        f"/basit - Hızlı mod\n"
        f"/derin - Derin düşünme\n"
        f"/yeni - Sıfırla"
    )


async def basit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/basit"""
    user_id = update.effective_user.id
    ai = get_ai(user_id)
    ai.mode = "basit"
    await update.message.reply_text("⚡ Basit mod aktif (HafizaAsistani)")


async def derin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/derin"""
    user_id = update.effective_user.id
    ai = get_ai(user_id)

    if not ai.quantum:
        await update.message.reply_text("❌ QuantumTree mevcut değil, basit modda kalınıyor.")
        return

    ai.mode = "derin"
    await update.message.reply_text("🧠 Derin mod aktif (QuantumTree)")


async def yeni_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/yeni"""
    user_id = update.effective_user.id
    ai = get_ai(user_id)
    result = ai.reset()
    await update.message.reply_text(result)


# === MESAJ HANDLER ===

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Text mesaj"""
    user_input = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Düşünüyorum mesajı
    status = await context.bot.send_message(chat_id, "💭 Düşünüyorum...")

    # İşle
    ai = get_ai(user_id)
    response = await ai.process(user_input)

    # Status mesajını sil
    try:
        await context.bot.delete_message(chat_id, status.message_id)
    except:
        pass

    # Cevabı gönder
    await update.message.reply_text(response)


# === MAIN ===

async def shutdown_handler():
    """Bot kapanırken tüm kullanıcıların sohbetlerini özetle"""
    print("\n🛑 Bot kapatılıyor...")
    print("📝 Sohbetler özetleniyor...")

    for user_id, ai in ai_instances.items():
        try:
            await ai.summarize_and_save()
        except Exception as e:
            print(f"⚠️ User {user_id} özet hatası: {e}")

    print("✅ Tüm sohbetler kaydedildi!")


def main():
    print("=" * 50)
    print("🚀 Telegram Bot Başlatılıyor...")
    print("=" * 50)

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ TELEGRAM_TOKEN bulunamadı!")
        sys.exit(1)

    app = Application.builder().token(token).build()

    # Komutlar
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("basit", basit_command))
    app.add_handler(CommandHandler("derin", derin_command))
    app.add_handler(CommandHandler("yeni", yeni_command))

    # Mesaj
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Shutdown handler
    app.post_shutdown = lambda _: asyncio.get_event_loop().run_until_complete(shutdown_handler())

    print("✅ Bot hazır!")
    print("🛑 Durdurmak için Ctrl+C")
    print("=" * 50)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
