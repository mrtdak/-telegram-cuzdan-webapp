# -*- coding: utf-8 -*-
import sys
import os
import io

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass  # Console encoding değiştirilemedi, varsayılan kullanılacak
import asyncio
import logging
import json
import re
import time
import threading
import base64
from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from typing import List, Dict, Optional, Tuple, Any
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np 
from neo4j import GraphDatabase

# =========================================================
# PERSONAI IMPORT - YENİ TEK DOSYA YAPISI
# =========================================================
try:
    from personal_ai import PersonalAI, ResponseCodes, SystemConfig

    # Alias for compatibility
    AIResponseCodes = ResponseCodes

    logger = logging.getLogger(__name__)
    print("✅ PersonalAI modülü yüklendi")
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    print(f"❌ PersonalAI modülü yüklenemedi: {e}")
    
    # Fallback classes
    class PersonalAI:
        def __init__(self, user_id="murat"):
            self.user_id = user_id
            
        async def process(self, user_input, chat_history, image_data=None):
            return "PersonalAI bağlantı hatası", user_input, "Sistem yüklenemedi"
            
        def set_mode(self, mode): pass
        def reset_conversation(self): pass
        def close(self): pass
        def get_system_stats(self): return {}
    
    class ResponseCodes:
        NO_DATA = "NO_DATA"
        API_ERROR = "API_ERROR"
        SEARCH_FAILED = "SEARCH_FAILED"
        
    class SystemConfig:
        DEFAULT_USER_ID = "murat"
    
    AIResponseCodes = ResponseCodes

# =========================================================
# QUANTUM TREE IMPORT (OPTIONAL)
# =========================================================
try:
    from quantum_agac import QuantumTree
    quantum_available = True
except ImportError:
    quantum_available = False
    class QuantumTree:
        def __init__(self, *args, **kwargs): pass
        def truth_filter(self, query): return {"final_response": "QuantumTree mevcut değil"}
        def stop_background(self): pass

# =========================================================
# RISALE ARAMA SİSTEMİ IMPORT
# =========================================================
try:
    from rısale import RisaleSearchEngine, RisaleTelegramInterface
    risale_available = True
    print("✅ Risale modülü yüklendi")
except ImportError as e:
    risale_available = False
    print(f"⚠️ Risale modülü yüklenemedi: {e}")

# =========================================================
# SES İŞLEME
# =========================================================
import speech_recognition as sr
from pydub import AudioSegment
import gtts
import io

# =========================================================
# TEMEL YAPILANDIRMA
# =========================================================
load_dotenv()
logging.basicConfig(
    level=logging.WARNING,  # Sadece uyarı ve hatalar
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Harici kütüphanelerin loglarını kapat
logging.getLogger("telegram").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Global değişkenler
ai_instances: Dict[int, "PersonalAIWrapperEnhanced"] = {}
quantum_tree: Optional[QuantumTree] = None
chat_manager: Optional["ChatHistoryManager"] = None
chat_analyzer: Optional["ChatDataAnalyzer"] = None
risale_engine = None
risale_interface = None

# Kullanıcı ayarları
user_voice_modes = {}
user_voice_speeds = {}

# =========================================================
# ⚠️ TIMEOUT AYARLARI - YENİ!
# =========================================================
PROCESS_TIMEOUT = 120  # PersonalAI için maksimum bekleme süresi (saniye)

# =========================================================
# YAPILANDIRMA MESAJLARI
# =========================================================
USER_MESSAGES_TR = {
    "error": "Bir sorun oluştu, lütfen tekrar deneyin",
    "no_data": "Bu konuda bilgi bulamadım, başka nasıl yardımcı olabilirim?",
    "processing_error": "Şu anda teknik bir sorun var, biraz sonra tekrar dener misin?",
    "image_error": "Görseli analiz edemedim, tekrar yükler misin?",
    "search_failed": "Arama yaparken sorun yaşadım, tekrar deneyebilir misin?",
    "cant_understand": "Anlayamadım, farklı bir şekilde sorabilirim misin?",
    "timeout_error": "Yanıt süresi aşıldı, tekrar dener misin? 🔄"
}

MESSAGE_FORMATTING = {
    "max_chunk_length": 3800,
    "preferred_chunk_length": 2000,
    "min_chunk_length": 500,
    "chunk_delay": 1.5,
    "add_formatting": True,
    "smart_breaking": True,
}

VOICE_CONFIG = {
    "speech_recognition": {
        "language": "tr-TR",
        "timeout": 10,
        "phrase_time_limit": 30,
        "energy_threshold": 300
    },
    "text_to_speech": {
        "language": "tr",
        "slow": False,
        "tld": "com.tr",
        "speed_settings": {
            "yavaş": {"slow": True, "speed": 0.8},
            "normal": {"slow": False, "speed": 1.0},
            "hızlı": {"slow": False, "speed": 1.25},
            "çok_hızlı": {"slow": False, "speed": 1.5}
        }
    },
    "voice_response": {
        "auto_voice_reply": True,
        "voice_command": "sesli",
        "max_tts_length": 500
    }
}

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def get_system_user_id(telegram_user_id: int) -> str:
    """Telegram ID'den PersonalAI için sistem user ID'si oluştur"""
    return f"user_{telegram_user_id}"

# =========================================================
# TOKEN YÖNETİMİ - SLIDING WINDOW SİSTEMİ
# =========================================================
def calculate_tokens(history: List[Dict]) -> int:
    """Chat history'nin yaklaşık token sayısını hesapla"""
    if not history:
        return 0
    
    total_tokens = 0
    for msg in history:
        content = msg.get('content', '')
        # Türkçe için: 1 token ≈ 4 karakter
        total_tokens += len(content) // 4
    
    return total_tokens

def manage_history_tokens(context) -> bool:
    """
    Token limiti kontrolü ve otomatik temizleme (Sliding Window)
    
    Returns:
        bool: Temizlik yapıldı mı?
    """
    history = context.chat_data.get('history', [])
    
    if not history:
        return False
    
    # Token sayısını hesapla
    current_tokens = calculate_tokens(history)
    
    # Limitler
    MAX_TOKENS = 5000  # Maksimum token eşiği
    REMOVE_COUNT = 10  # Kaç mesaj silinecek (5 soru-cevap çifti)
    
    if current_tokens > MAX_TOKENS:
        # Token limiti aşıldı - eski mesajları sil
        context.chat_data['history'] = history[REMOVE_COUNT:]
        new_tokens = calculate_tokens(context.chat_data['history'])
        print(f"🔄 Token optimizasyonu: {current_tokens} → {new_tokens} ({len(history)} → {len(context.chat_data['history'])} mesaj)")
        return True
    
    return False

def smart_text_splitter(text: str, max_length: int = 3800, is_deep_mode: bool = False) -> list:
    """Metni akıllıca parçalara böler"""
    if not text or len(text) <= max_length:
        return [text] if text else ["Yanıt boş"]
    
    chunks = []
    current_chunk = ""
    sentences = text.split('. ')
    
    for sentence in sentences:
        if len(current_chunk + sentence + '. ') > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks if chunks else [text]

def clean_markdown(text: str) -> str:
    """Telegram Markdown'ı KORUYARAK sadece tehlikeli karakterleri temizle"""
    if not text:
        return text

    # Markdown'ı KORUYORUZ artık! Sadece çok uzun satırları böl
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if len(line) > 4000:
            while len(line) > 4000:
                split_point = line.rfind(' ', 0, 4000)
                if split_point == -1:
                    split_point = 4000
                cleaned_lines.append(line[:split_point])
                line = line[split_point:].strip()
            if line:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def clean_response_for_user(response):
    """Yanıtı kullanıcı dostu hale getir"""
    if not response or not isinstance(response, str) or len(response.strip()) < 3:
        return USER_MESSAGES_TR["cant_understand"]
    
    error_mappings = {
        "API_ERROR": USER_MESSAGES_TR["processing_error"],
        "NO_DATA": USER_MESSAGES_TR["no_data"],
        "SEARCH_FAILED": USER_MESSAGES_TR["search_failed"],
        "IMAGE_PROCESSING_ERROR": USER_MESSAGES_TR["image_error"],
        "processing_error": USER_MESSAGES_TR["processing_error"]
    }
    
    for error_code, user_message in error_mappings.items():
        if error_code in response:
            return user_message
    
    return response.strip()

def format_response_for_telegram(text: str) -> str:
    """LLM yanıtını Telegram için paragraflara böl"""
    if not text:
        return text
    
    sentences = re.split(r'([.!?])\s+', text.strip())
    formatted_text = ""
    sentence_count = 0
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        
        formatted_text += sentence + " "
        sentence_count += 1
        
        # Her 3 cümleden sonra paragraf boşluğu
        if sentence_count >= 3 and i + 2 < len(sentences):
            formatted_text += "\n\n"
            sentence_count = 0
    
    return formatted_text.strip()

def should_send_voice_response(user_id: int, user_input: str) -> bool:
    """Sesli yanıt verilmeli mi?"""
    # Kullanıcı kaydettiği modu kullanır.
    user_voice_mode = user_voice_modes.get(user_id, False)
    if user_voice_mode:
        return True
    # Sadece "sesli" komutu ile tetiklenir ("ses" kelimesi kaldırıldı)
    return "sesli" in user_input.lower()

def should_send_text_response(user_id: int) -> bool:
    """Metin yanıtı verilmeli mi? (Şimdilik hep True, ileride seçenek olabilir)"""
    return True

# =========================================================
# NEO4J CHAT HAFIZA SİSTEMİ
# =========================================================
class ChatHistoryManager:
    """Neo4j tabanlı chat hafıza yöneticisi"""

    def __init__(self, neo4j_uri="bolt://localhost:7687", user="neo4j", password="senegal5454", enabled=False):
        self.driver = None
        if not enabled:
            print("ℹ️ Neo4j chat hafızası devre dışı")
            return
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Neo4j chat hafızası bağlandı")
        except Exception as e:
            print(f"❌ Neo4j bağlantı hatası: {e}")
            self.driver = None
    
    def get_most_relevant_chat(self, user_id: str, keyword: str) -> Optional[Tuple[str, str]]:
        """Anahtar kelimeye en alakalı sohbeti getir"""
        if not self.driver:
            return None
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Conversation {user_id: $user_id})
                    WHERE toLower(c.user_input) CONTAINS toLower($keyword) OR 
                          toLower(c.ai_response) CONTAINS toLower($keyword)
                    RETURN c.user_input, c.ai_response
                    ORDER BY c.timestamp DESC LIMIT 1
                """, user_id=str(user_id), keyword=keyword)
                
                record = result.single()
                if record:
                    return (record["c.user_input"], record["c.ai_response"])
                return None
        except Exception:
            return None
    
    def get_recent_context(self, user_id, limit=10):
        """Son konuşmaları getir"""
        if not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Conversation {user_id: $user_id}) "
                    "RETURN c.user_input, c.ai_response, c.timestamp "
                    "ORDER BY c.timestamp DESC LIMIT $limit",
                    user_id=str(user_id), limit=limit
                )
                contexts = []
                for record in result:
                    contexts.append((
                        record["c.user_input"],
                        record["c.ai_response"],
                        record["c.timestamp"]
                    ))
                return contexts[::-1]
        except Exception:
            return []
    
    def get_chat_summary(self, user_id, days=7):
        """Chat özetini getir"""
        if not self.driver:
            return "Chat hafızası mevcut değil"
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Conversation {user_id: $user_id}) "
                    "WHERE c.date >= date() - duration({days: $days}) "
                    "RETURN count(c) as total_chats, "
                    "collect(c.user_input)[..3] as recent_topics "
                    "ORDER BY c.timestamp DESC",
                    user_id=str(user_id), days=days
                )
                record = result.single()
                if record:
                    total = record["total_chats"]
                    topics = record["recent_topics"]
                    return f"Son {days} günde {total} sohbet. Son konular: {', '.join(topics[:3])}"
                return "Henüz sohbet yok"
        except Exception:
            return "Chat özeti alınamadı"
    
    def search_chats(self, user_id, keyword, limit=3):
        """Konuşmalarda arama yap"""
        if not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Conversation {user_id: $user_id}) "
                    "WHERE toLower(c.user_input) CONTAINS toLower($keyword) OR "
                    "toLower(c.ai_response) CONTAINS toLower($keyword) "
                    "RETURN c.user_input, c.ai_response, c.timestamp "
                    "ORDER BY c.timestamp DESC LIMIT $limit",
                    user_id=str(user_id), keyword=keyword, limit=limit
                )
                return [
                    (record["c.user_input"], record["c.ai_response"], record["c.timestamp"])
                    for record in result
                ]
        except Exception:
            return []

class ChatDataAnalyzer:
    """Neo4j tabanlı chat veri analizi"""

    def __init__(self, neo4j_uri="bolt://localhost:7687", user="neo4j", password="senegal5454", enabled=False):
        self.driver = None
        if not enabled:
            return
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            print(f"⚠️ Chat analiz Neo4j hatası: {e}")
            self.driver = None
    
    def get_user_chat_stats(self, user_id: str) -> Dict[str, Any]:
        """Kullanucu chat istatistikleri"""
        if not self.driver:
            return {"error": "Neo4j bağlantısı yok"}
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Conversation {user_id: $user_id}) "
                    "RETURN count(c) as total_chats, "
                    "min(c.timestamp) as first_chat, "
                    "max(c.timestamp) as last_chat",
                    user_id=str(user_id)
                )
                record = result.single()
                
                if not record:
                    return {
                        "total_chats": 0,
                        "first_chat_date": "Henüz yok",
                        "last_chat_date": "Henüz yok",
                        "daily_average": 0,
                        "recent_chats_7days": 0
                    }
                
                total_chats = record["total_chats"] or 0
                first_chat = record["first_chat"]
                last_chat = record["last_chat"]
                
                daily_avg = 0
                if first_chat and last_chat and total_chats > 0:
                    days_diff = max(1, (last_chat - first_chat) / (1000 * 86400))
                    daily_avg = total_chats / days_diff
                
                seven_days_ago = datetime.now() - timedelta(days=7)
                result = session.run(
                    "MATCH (c:Conversation {user_id: $user_id}) "
                    "WHERE c.date >= date($seven_days_ago) "
                    "RETURN count(c) as recent_chats",
                    user_id=str(user_id),
                    seven_days_ago=seven_days_ago.strftime("%Y-%m-%d")
                )
                recent_record = result.single()
                recent_chats = recent_record["recent_chats"] if recent_record else 0
                
                return {
                    "total_chats": total_chats,
                    "first_chat_date": datetime.fromtimestamp(first_chat/1000).strftime("%Y-%m-%d %H:%M") if first_chat else "Henüz yok",
                    "last_chat_date": datetime.fromtimestamp(last_chat/1000).strftime("%Y-%m-%d %H:%M") if last_chat else "Henüz yok",
                    "daily_average": round(daily_avg, 2),
                    "recent_chats_7days": recent_chats
                }
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Neo4j bağlantısını kapat"""
        if self.driver:
            self.driver.close()

# =========================================================
# SES İŞLEME SINIFI
# =========================================================
class VoiceProcessor:
    """Ses tanıma ve sentezleme"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = VOICE_CONFIG["speech_recognition"]["energy_threshold"]
    
    async def speech_to_text(self, voice_data: bytes) -> str:
        """Sesi metne çevir"""
        try:
            audio_segment = AudioSegment.from_ogg(io.BytesIO(voice_data))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            with sr.AudioFile(wav_buffer) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(
                audio_data,
                language=VOICE_CONFIG["speech_recognition"]["language"]
            )
            return text.strip()
            
        except sr.UnknownValueError:
            return "❌ Ses anlaşılamadı"
        except sr.RequestError:
            return "❌ Ses tanıma servisi hatası"
        except Exception:
            return "❌ Ses işlenirken hata oluştu"
    
    async def text_to_speech(self, text: str, user_id: int) -> io.BytesIO:
        """Metni sese çevir"""
        try:
            if len(text) > VOICE_CONFIG["voice_response"]["max_tts_length"]:
                text = text[:VOICE_CONFIG["voice_response"]["max_tts_length"]] + "..."
            
            user_speed = user_voice_speeds.get(user_id, "normal")
            speed_settings = VOICE_CONFIG["text_to_speech"]["speed_settings"][user_speed]
            
            tts = gtts.gTTS(
                text=text,
                lang=VOICE_CONFIG["text_to_speech"]["language"],
                slow=speed_settings["slow"],
                tld=VOICE_CONFIG["text_to_speech"]["tld"]
            )
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            if speed_settings["speed"] != 1.0:
                try:
                    audio_segment = AudioSegment.from_mp3(audio_buffer)
                    
                    if speed_settings["speed"] > 1.0:
                        faster_audio = audio_segment.speedup(
                            playback_speed=speed_settings["speed"]
                        )
                    else:
                        slower_speed = 1.0 / speed_settings["speed"]
                        faster_audio = audio_segment.speedup(
                            playback_speed=1.0 / slower_speed
                        )
                    
                    final_buffer = io.BytesIO()
                    faster_audio.export(final_buffer, format="mp3") 
                    final_buffer.seek(0)
                    return final_buffer
                    
                except Exception:
                    audio_buffer.seek(0)
                    return audio_buffer

            return audio_buffer

        except Exception:
            return None

voice_processor = VoiceProcessor()

# =========================================================
# PERSONAI WRAPPER - YENİ TEK DOSYA YAPISI
# =========================================================
class PersonalAIWrapper:
    """PersonalAI sistemine direkt bağlanan wrapper"""
    
    def __init__(self, user_id="murat"):
        self.user_id = user_id
        
        # PersonalAI'ı direkt başlat
        self.ai_core_instance = PersonalAI(user_id=user_id)
        # KULLANICI TALİMATI: Her zaman basit modda başla
        self.current_mode = "simple"
    
    def validate_and_format_chat_history(self, telegram_history: List[Dict]) -> List[Dict[str, Any]]:
        """Telegram chat history'yi PersonalAI formatına çevir"""
        if not telegram_history:
            return []

        validated_history = []
        for entry in telegram_history:
            if not isinstance(entry, dict):
                continue

            role = entry.get('role', '')
            content = (
                entry.get('content_en') or
                entry.get('content') or
                entry.get('message') or
                ""
            ).strip()

            if not role or not content:
                continue

            validated_history.append({
                'role': role,
                'content': content,
                'timestamp': entry.get('timestamp', time.time())
            })

        return validated_history
    
    async def process(self, user_input: str, chat_history: List[Dict],
                      image_data: Optional[bytes] = None) -> Tuple[Any, str, str]:
        """Ana process metodu - PersonalAI'a direkt bağlı"""
        try:
            formatted_history = self.validate_and_format_chat_history(chat_history)

            # Deep mode kontrolü
            if self.current_mode == "deep" and quantum_tree:
                
                try:
                    english_text = GoogleTranslator(source='auto', target='en').translate(user_input)
                except Exception as e:
                    print(f"Çeviri hatası: {e}")
                    english_text = user_input
                
                loop = asyncio.get_event_loop()
                quantum_result = await loop.run_in_executor(None, quantum_tree.truth_filter, english_text)
                final_response_text = quantum_result.get("final_response") or "QuantumTree yanıt üretemedi."
                
                return {
                    "chat_response": final_response_text,
                    "quantum_result": quantum_result,
                    "mode": "deep"
                }, user_input, final_response_text

            else:
                # TIMEOUT İLE PERSONAI ÇAĞIR
                ai_response_tr, processed_input, final_response = await asyncio.wait_for(
                    self.ai_core_instance.process(
                        user_input=user_input,
                        chat_history=formatted_history,
                        image_data=image_data
                    ),
                    timeout=PROCESS_TIMEOUT
                )
                
                return {
                    "chat_response": ai_response_tr,
                    "processed_input": processed_input,
                    "mode": self.current_mode,
                    "personai_used": True
                }, user_input, ai_response_tr

        except asyncio.TimeoutError:
            error_msg = USER_MESSAGES_TR["timeout_error"]
            return {"chat_response": error_msg, "error": "timeout"}, user_input, error_msg

        except Exception as e:
            error_msg = USER_MESSAGES_TR["processing_error"]
            return {"chat_response": error_msg, "error": str(e)}, user_input, error_msg
    
    def set_mode(self, mode: str):
        """Mode değiştir"""
        self.current_mode = mode
        if hasattr(self.ai_core_instance, 'set_mode'):
            self.ai_core_instance.set_mode(mode)

    def reset_conversation(self):
        """Konuşmayı sıfırla - DeepThinker dahil!"""
        if hasattr(self.ai_core_instance, 'memory') and self.ai_core_instance.memory:
            self.ai_core_instance.memory.clear()

        if hasattr(self.ai_core_instance, 'thinker') and self.ai_core_instance.thinker:
            self.ai_core_instance.thinker.clear_context()

        return "✅ Konuşma sıfırlandı"

class PersonalAIWrapperEnhanced(PersonalAIWrapper):
    """Enhanced wrapper with usage tracking"""
    
    def __init__(self, user_id="murat"):
        super().__init__(user_id)
        self.last_used = time.time()
        self.message_count = 0
    
    async def process(self, user_input: str, chat_history: List[Dict], 
                      image_data: Optional[bytes] = None):
        """Enhanced process with tracking"""
        self.last_used = time.time()
        self.message_count += 1
        
        return await super().process(user_input, chat_history, image_data)

# =========================================================
# TELEGRAM BOT FONKSİYONLARI
# =========================================================
def get_user_ai(telegram_user_id):
    """Her Telegram kullanıcısı için ayrı PersonalAI instance oluştur"""
    telegram_id = int(telegram_user_id)

    if telegram_id not in ai_instances:
        system_user_id = get_system_user_id(telegram_id)
        ai_instances[telegram_id] = PersonalAIWrapperEnhanced(user_id=system_user_id)

    return ai_instances[telegram_id]

def initialize_chat_history(context):
    """Chat history'yi initialize et"""
    if 'history' not in context.chat_data:
        context.chat_data['history'] = []
    if 'mode_asked' not in context.user_data:
        context.user_data['mode_asked'] = False
    return context.chat_data['history']

def add_to_chat_history(context, user_message: str, ai_response: str):
    """Chat history'e mesaj ekle - PersonalAI formatında"""
    current_time = time.time()
    history = context.chat_data.get('history', [])
    
    # User message ekle (PersonalAI formatı)
    history.append({
        'role': 'user',
        'content': user_message,
        'timestamp': current_time
    })
    
    # AI response ekle (PersonalAI formatı)
    history.append({
        'role': 'ai',
        'content': ai_response,
        'timestamp': current_time
    })
    
    # History limitini koru (son 30 mesaj = 15 çift)
    if len(history) > 30:
        context.chat_data['history'] = history[-30:]
    else:
        context.chat_data['history'] = history

def initialize_quantum_tree():
    """QuantumTree'yi başlat"""
    global quantum_tree

    if not quantum_available:
        return False

    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PASSWORD") or ""

        quantum_tree = QuantumTree(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            thinking_framework_path="thinking_framework.json"
        )
        return True

    except Exception as e:
        print(f"❌ QuantumTree başlatma hatası: {e}")
        return False

def cleanup_inactive_users():
    """Pasif kullanıcıları temizle (24 saat)"""
    current_time = time.time()
    inactive_users = []

    for telegram_id, ai_instance in list(ai_instances.items()):
        if hasattr(ai_instance, 'last_used'):
            if current_time - ai_instance.last_used > 86400:
                inactive_users.append(telegram_id)

    for telegram_id in inactive_users:
        try:
            if hasattr(ai_instances[telegram_id], 'ai_core_instance'):
                ai_core = ai_instances[telegram_id].ai_core_instance
                if hasattr(ai_core, 'close'):
                    ai_core.close()
            del ai_instances[telegram_id]
        except Exception:
            pass

def start_cleanup_timer():
    """Her 6 saatte bir cleanup çalıştır"""
    def cleanup_timer():
        time.sleep(300)
        while True:
            time.sleep(21600)
            try:
                cleanup_inactive_users()
            except Exception:
                pass

    cleanup_thread = threading.Thread(target=cleanup_timer, daemon=True)
    cleanup_thread.start()

# =========================================================
# TELEGRAM KOMUT HANDLERs
# =========================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start komutu"""
    user_id = update.effective_user.id
    initialize_chat_history(context)
    
    try:
        user_ai = get_user_ai(user_id)
        user_ai.set_mode("simple") 
        ai_mode = getattr(user_ai, 'current_mode', 'simple')
        voice_mode = user_voice_modes.get(user_id, False)
        
        ai_icon = "🔍" if ai_mode == "deep" else "⚡"
        ai_text = "Derin" if ai_mode == "deep" else "Hızlı"
        voice_icon = "🔊" if voice_mode else "📝"
        voice_text = "Sesli" if voice_mode else "Yazılı"
        
        risale_status = "✅ Aktif" if risale_interface else "❌ Kapalı"
        
        response_text = (
            f"🤖 **PersonalAI Telegram Bot Hazır!**\n\n"
            f"{ai_icon} **{ai_text} AI** | {voice_icon} **{voice_text} Mod**\n\n"
            f"🧠 Kalıcı Hafıza Aktif!\n"
            f"📖 Risale Arama: {risale_status}\n\n"
            f"Komutlar:\n"
            f"• /yazili - Yazılı mod\n"
            f"• /sesli - Sesli mod\n"
            f"• /hizli_ai - Hızlı AI\n"
            f"• /risale - Risale-i Nur arama\n"
            f"• /yeni - Sohbeti sıfırla\n"
            f"• /gecmis - Chat özeti\n"
            f"• /ara [kelime] - Sohbet arama\n\n"
            f"💬 Mesaj yaz veya sesli mesaj gönder!"
        )
        
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
        if not context.user_data.get('mode_asked', False):
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Sohbeti Yeniden Başlat", callback_data="reset_chat")]
            ])
            await update.message.reply_text(
                "🔄 **Sohbeti sıfırlamak için butona bas:**",
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            context.user_data['mode_asked'] = True

    except Exception:
        await update.message.reply_text("❌ Başlatırken sorun oluştu, tekrar deneyin.")

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sohbeti sıfırla - DeepThinker dahil!"""
    user_id = update.effective_user.id
    try:
        if 'history' in context.chat_data:
            context.chat_data['history'].clear()

        user_ai = get_user_ai(user_id)
        user_ai.reset_conversation()

        await update.message.reply_text(
            "✅ **Sohbet Tamamen Sıfırlandı!**\n\n"
            "🆕 Yeni başlangıç yapabiliriz\n"
            "🧠 PersonalAI hafızası temizlendi\n"
            "🔄 DeepThinker bağlamı temizlendi\n"
            "📊 Neo4j kalıcı hafıza korunuyor",
            parse_mode='Markdown'
        )

    except Exception:
        await update.message.reply_text("❌ Sıfırlarken sorun oluştu, tekrar dener misin?")

async def gecmis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chat geçmişi özeti"""
    user_id = update.effective_user.id
    
    if chat_manager and chat_manager.driver:
        summary = chat_manager.get_chat_summary(get_system_user_id(user_id), 7)
    else:
        summary = "❌ Chat hafızası aktif değil"
    
    history_count = len(context.chat_data.get('history', []))
    
    await update.message.reply_text(
        f"📊 **Sohbet Özeti (Son 7 Gün)**\n\n"
        f"{summary}\n\n"
        f"📱 Telegram Oturum: {history_count} mesaj\n"
        f"🧠 PersonalAI Hafıza: Aktif",
        parse_mode='Markdown'
    )

async def ara_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sohbet geçmişinde arama"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "🔍 Nasıl kullanılır:\n"
            "/ara [arama kelimesi]\n\n"
            "Örnek: **/ara python**",
            parse_mode='Markdown'
        )
        return
    
    keyword = " ".join(context.args)
    
    if chat_manager and chat_manager.driver:
        results = chat_manager.search_chats(get_system_user_id(user_id), keyword, 3)
        
        if results:
            response = f"🔍 **'{keyword}' araması sonuçları:**\n\n"
            for i, (message, response_text, timestamp) in enumerate(results, 1):
                response += f"**{i}. Kullanıcı:** {message[:50]}...\n💬 **AI Yanıtı:** {response_text[:100]}...\n\n"
        else:
            response = f"❌ **'{keyword}' için sonuç bulunamadı**"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    else:
        await update.message.reply_text("❌ Chat hafızası aktif değil")

async def yazili_mod_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Yazılı mod"""
    user_id = update.effective_user.id
    user_voice_modes[user_id] = False
    await update.message.reply_text(
        "📝 **Yazılı Mod Aktif!**\n"
        "Sadece metin yanıtlar vereceğim.",
        parse_mode='Markdown'
    )

async def sesli_mod_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sesli mod"""
    user_id = update.effective_user.id
    user_voice_modes[user_id] = True
    await update.message.reply_text(
        "🔊 **Sesli Mod Aktif!**\n"
        "Her mesajına sesli yanıt vereceğim!",
        parse_mode='Markdown'
    )

async def hizli_ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hızlı AI modu"""
    user_id = update.effective_user.id
    user_ai = get_user_ai(user_id)
    user_ai.set_mode("simple")
    await update.message.reply_text(
        "⚡ **Hızlı AI Aktif!**\n"
        "PersonalAI hızlı modda çalışıyor.",
        parse_mode='Markdown'
    )

async def derin_ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Derin AI modu - devre dışı, her zaman hızlı mod aktif"""
    # Her zaman hızlı modda kal
    await update.message.reply_text(
        "⚡ **Hızlı AI aktif!**\n"
        "Sistem her zaman hızlı modda çalışıyor.",
        parse_mode='Markdown'
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sistem istatistikleri (Admin)"""
    if update.effective_user.id != 6505503887:
        await update.message.reply_text("❌ Bu komut sadece admin içindir.", parse_mode=None)
        return

    stats = f"""📊 **Sistem İstatistikleri**

👥 **Aktif Kullanıcı Instance'ları**: {len(ai_instances)}

**Kullanıcı Detayları**:"""

    for telegram_id, ai_instance in ai_instances.items():
        system_user_id = ai_instance.user_id
        message_count = getattr(ai_instance, 'message_count', 0)
        last_used = getattr(ai_instance, 'last_used', 0)

        if last_used:
            last_used_str = datetime.fromtimestamp(last_used).strftime("%H:%M")
        else:
            last_used_str = "Bilinmiyor"

        stats += f"\n• User {telegram_id} ({system_user_id.split('_')[-1]}): **{message_count}** mesaj, son: {last_used_str}"

    await update.message.reply_text(stats, parse_mode='Markdown')

# =========================================================
# MENÜ VE BİLGİSAYAR KONTROL KOMUTLARI
# =========================================================
ADMIN_USER_ID = 6505503887  # Sadece admin kullanabilir

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kontrol menüsü - Sadece admin için"""
    user_id = update.effective_user.id

    if user_id != ADMIN_USER_ID:
        await update.message.reply_text("❌ Bu menü sadece admin içindir.")
        return

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Bilgisayarı Yeniden Başlat", callback_data="pc_restart")],
        [InlineKeyboardButton("⏻ Bilgisayarı Kapat", callback_data="pc_shutdown")],
    ])

    await update.message.reply_text(
        "📋 **Kontrol Menüsü**\n\n"
        "Aşağıdaki işlemlerden birini seçin:",
        reply_markup=keyboard,
        parse_mode='Markdown'
    )

async def handle_pc_control_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bilgisayar kontrol callback handler"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    # Admin kontrolü
    if user_id != ADMIN_USER_ID:
        await query.edit_message_text("❌ Bu işlem sadece admin tarafından yapılabilir.")
        return

    if callback_data == "pc_shutdown":
        # Onay iste
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Evet, Kapat", callback_data="pc_shutdown_confirm"),
                InlineKeyboardButton("❌ İptal", callback_data="pc_cancel")
            ]
        ])
        await query.edit_message_text(
            "⚠️ **Bilgisayarı kapatmak istediğinize emin misiniz?**\n\n"
            "Bu işlem geri alınamaz!",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )

    elif callback_data == "pc_restart":
        # Onay iste
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Evet, Yeniden Başlat", callback_data="pc_restart_confirm"),
                InlineKeyboardButton("❌ İptal", callback_data="pc_cancel")
            ]
        ])
        await query.edit_message_text(
            "⚠️ **Bilgisayarı yeniden başlatmak istediğinize emin misiniz?**\n\n"
            "Bot otomatik olarak tekrar başlayacak.",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )

    elif callback_data == "pc_shutdown_confirm":
        await query.edit_message_text("⏻ **Bilgisayar 10 saniye içinde kapanacak...**", parse_mode='Markdown')
        # Windows shutdown komutu (10 saniye gecikme)
        os.system("shutdown /s /t 10")

    elif callback_data == "pc_restart_confirm":
        await query.edit_message_text("🔁 **Bilgisayar 10 saniye içinde yeniden başlayacak...**\n\nBot otomatik olarak açılacak.", parse_mode='Markdown')
        # Windows restart komutu (10 saniye gecikme)
        os.system("shutdown /r /t 10")

    elif callback_data == "pc_cancel":
        await query.edit_message_text("❌ İşlem iptal edildi.")

# =========================================================
# RISALE ARAMA HANDLERS
# =========================================================
async def risale_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ana Risale arama komutu"""
    if not risale_interface:
        await update.message.reply_text(
            "❌ **Risale arama sistemi şu anda kullanılamıyor.**\n"
            "Sistem yöneticisi ile iletişime geçin.",
            parse_mode='Markdown'
        )
        return
    
    try:
        menu_data = risale_interface.create_main_menu()
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
            for row in menu_data["keyboard"]
        ])
        
        await update.message.reply_text(
            menu_data["text"],
            reply_markup=keyboard,
            parse_mode=None
        )

    except Exception:
        await update.message.reply_text("❌ Risale menüsü yüklenirken hata oluştu.")

async def handle_risale_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risale arama callback handler'ı"""
    query = update.callback_query
    await query.answer()
    
    if not risale_interface:
        await query.edit_message_text("❌ Risale arama sistemi kullanılamıyor.")
        return
    
    try:
        callback_data = query.data
        user_id = update.effective_user.id
        
        if callback_data == "set_mode_yazili":
            user_voice_modes[user_id] = False
            await query.edit_message_text("📝 **Yazılı Mod** tercih edildi. Yanıtlar bundan sonra metin olarak gelecek.")
            return

        if callback_data == "set_mode_sesli":
            user_voice_modes[user_id] = True
            await query.edit_message_text("🔊 **Sesli Mod** tercih edildi. Yanıtlar bundan sonra sesli olarak gelecek.")
            return

        if callback_data == "reset_chat":
            if 'history' in context.chat_data:
                context.chat_data['history'].clear()

            user_ai = get_user_ai(user_id)
            user_ai.reset_conversation()

            await query.edit_message_text(
                "✅ **Sohbet Sıfırlandı!**\n\n"
                "🆕 Yeni bir sohbet başlatabilirsin.\n"
                "🧠 Hafıza temizlendi.",
                parse_mode='Markdown'
            )
            return

        if callback_data == "main_menu":
            menu_data = risale_interface.create_main_menu()
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
                for row in menu_data["keyboard"]
            ])
            await query.edit_message_text(menu_data["text"], reply_markup=keyboard, parse_mode=None)
            
        elif callback_data == "select_source":
            menu_data = risale_interface.create_source_menu()
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
                for row in menu_data["keyboard"]
            ])
            await query.edit_message_text(menu_data["text"], reply_markup=keyboard, parse_mode=None)
            
        elif callback_data == "show_stats":
            stats_data = risale_interface.get_statistics()
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
                for row in stats_data["keyboard"]
            ])
            await query.edit_message_text(stats_data["text"], reply_markup=keyboard, parse_mode=None)
            
        elif callback_data == "random_content":
            random_data = risale_interface.get_random_content()
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
                for row in random_data["keyboard"]
            ])
            await query.edit_message_text(random_data["text"], reply_markup=keyboard, parse_mode=None)
            
        elif callback_data == "search_keyword":
            context.user_data['risale_search_mode'] = True
            context.user_data['risale_source_filter'] = None
            await query.edit_message_text(
                "🔍 **Arama Modu Aktif**\n\n"
                "Aramak istediğiniz kelimeyi yazın.\n"
                "Örnek: iman, namaz, tevhid\n\n"
                "❌ İptal etmek için /iptal yazın.",
                parse_mode='Markdown'
            )
            
        elif callback_data.startswith("source_"):
            source = callback_data.replace("source_", "")
            if source == "all":
                source_name = "Tüm Kaynaklar"
                source_filter = None
            else:
                source_name = risale_engine.source_mapping.get(source, source.title())
                source_filter = source
            
            context.user_data['risale_search_mode'] = True
            context.user_data['risale_source_filter'] = source_filter
            
            await query.edit_message_text(
                f"📚 **{source_name}** seçildi\n\n"
                f"🔍 **Arama Modu Aktif**\n\n"
                f"Bu kaynakta aramak istediğiniz kelimeyi yazın.\n"
                f"Örnek: iman, namaz, tevhid\n\n"
                f"❌ İptal etmek için /iptal yazın.",
                parse_mode='Markdown'
            )
            
        elif callback_data == "help":
            help_text = """📖 **Risale-i Nur Arama Sistemi Yardımı**

🔍 **Kelime Ara**: Risale'de belirli kelimeleri arayın
📚 **Kaynak Seç**: Sadece belirli kitaplarda arama yapın
📊 **İstatistikler**: Veritabanı bilgilerini görün
🎲 **Rastgele İçerik**: Örnek metinler görün

Desteklenen Kaynaklar:
- Sözler
- Mektubat
- Lemalar
- Şualar
- Lahikalar

Arama İpuçları:
- Tek kelime aramaları daha etkilidir
- Farklı çekim eklerini deneyin
- "Nerede geçiyor" sorusunu ekleyebilirsiniz"""
            
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Ana Menü", callback_data="main_menu")
            ]])
            await query.edit_message_text(help_text, reply_markup=keyboard, parse_mode='Markdown')
            
        else:
            await query.edit_message_text("❌ Bilinmeyen komut.")

    except Exception:
        await query.edit_message_text("❌ İşlem gerçekleştirilemedi.")

async def handle_risale_search(update: Update, context: ContextTypes.DEFAULT_TYPE, search_query: str):
    """Risale arama işlemi"""
    if not risale_interface:
        await update.message.reply_text("❌ Risale arama sistemi kullanılamıyor.")
        return
    
    try:
        user_id = update.effective_user.id
        source_filter = context.user_data.get('risale_source_filter')
        
        search_data = risale_interface.handle_search_request(user_id, search_query, source_filter)
        
        if search_data.get("keyboard"):
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton(row[0]["text"], callback_data=row[0]["callback"])]
                for row in search_data["keyboard"]
            ])
            await update.message.reply_text(
                search_data["text"],
                reply_markup=keyboard,
                parse_mode=search_data.get('parse_mode', None)
            )
        else:
            await update.message.reply_text(
                search_data["text"],
                parse_mode=search_data.get('parse_mode', None)
            )
        
        context.user_data['risale_search_mode'] = False
        context.user_data['risale_source_filter'] = None

    except Exception:
        await update.message.reply_text("❌ Arama sırasında hata oluştu.")

async def iptal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risale arama modunu iptal et"""
    if context.user_data.get('risale_search_mode'):
        context.user_data['risale_search_mode'] = False
        context.user_data['risale_source_filter'] = None
        await update.message.reply_text(
            "❌ **Risale arama modu iptal edildi.**\n\n"
            "📖 Yeniden başlatmak için /risale komutunu kullanın.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text("❌ Aktif bir arama modu bulunmuyor.")

# =========================================================
# MESAJ HANDLERs
# =========================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ana mesaj handler - PersonalAI full integration + TIMEOUT"""
    user_input = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if context.user_data.get('risale_search_mode'):
        await handle_risale_search(update, context, user_input)
        return

    chat_history = initialize_chat_history(context)

    was_cleaned = manage_history_tokens(context)
    if was_cleaned:
        chat_history = context.chat_data.get('history', [])
    
    status_message = None
    
    try:
        user_ai = get_user_ai(user_id)
        current_mode = getattr(user_ai, 'current_mode', 'simple')
        is_deep_mode = current_mode == "deep"
        
        wants_voice_reply = should_send_voice_response(user_id, user_input)
        send_text = should_send_text_response(user_id)
        
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text="☁️ Düşünüyorum...",
            parse_mode=None
        )
        
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')

        ai_response, mode, status = await user_ai.process(
            user_input=user_input,
            chat_history=chat_history,
            image_data=None
        )
        
        if isinstance(ai_response, dict):
            processed_response = ai_response.get('chat_response', status)
        else:
            processed_response = ai_response
        
        # Status mesajını sil
        if status_message:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
            except Exception:
                pass
        
        raw_response = clean_response_for_user(processed_response)
        raw_response = format_response_for_telegram(raw_response)
        
        message_chunks = smart_text_splitter(raw_response, MESSAGE_FORMATTING["max_chunk_length"], is_deep_mode=False)
        
        add_to_chat_history(context, user_input, raw_response)
        
        voice_response = None
        if wants_voice_reply:
            voice_response = await voice_processor.text_to_speech(raw_response, user_id)
        
        for i, chunk in enumerate(message_chunks):
            cleaned_chunk = clean_markdown(chunk)

            if send_text:
                await context.bot.send_message(chat_id=chat_id, text=cleaned_chunk, parse_mode=None)

            if i == 0 and wants_voice_reply and voice_response:
                keyboard = [
                    [
                        InlineKeyboardButton("1x", callback_data=f"speed_play_normal_{user_id}"),
                        InlineKeyboardButton("1.25x", callback_data=f"speed_play_fast_{user_id}"),
                        InlineKeyboardButton("1.5x", callback_data=f"speed_play_faster_{user_id}"),
                        InlineKeyboardButton("2x", callback_data=f"speed_play_fastest_{user_id}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_voice(chat_id=chat_id, voice=voice_response, reply_markup=reply_markup)
            
            if i < len(message_chunks) - 1:
                delay = MESSAGE_FORMATTING["chunk_delay"]
                if is_deep_mode:
                    delay *= 1.5
                await asyncio.sleep(delay)

    except Exception:
        
        # Status mesajını sil
        if status_message:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
            except Exception:
                pass  # Mesaj zaten silinmiş olabilir

        await update.message.reply_text("❌ Bir sorun oluştu, tekrar dener misin? 🔄")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sesli mesaj handler"""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    chat_history = initialize_chat_history(context)
    status_message = None
    
    try:
        user_ai = get_user_ai(user_id)
        current_mode = getattr(user_ai, 'current_mode', 'simple')
        is_deep_mode = current_mode == "deep"
        
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text="☁️ Düşünüyorum...",
            parse_mode=None
        )
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        voice_data = await voice_file.download_as_bytearray()
        user_input = await voice_processor.speech_to_text(bytes(voice_data))
        
        if user_input.startswith("❌"):
            await context.bot.edit_message_text(
                text=user_input + "\n🔄 Tekrar dener misiniz?",
                chat_id=chat_id,
                message_id=status_message.message_id,
                parse_mode='Markdown'
            )
            return
        
        await context.bot.edit_message_text(
            text=f"🎧 Duyduğum: {user_input[:60]}...\n☁️ Düşünüyorum...",
            chat_id=chat_id,
            message_id=status_message.message_id,
            parse_mode=None
        )
        
        ai_response, mode, status = await user_ai.process(
            user_input=user_input,
            chat_history=chat_history,
            image_data=None
        )
        
        if isinstance(ai_response, dict):
            processed_response = ai_response.get('chat_response', status)
        else:
            processed_response = ai_response
        
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
        except Exception:
            pass
        
        raw_response = clean_response_for_user(processed_response)
        raw_response = format_response_for_telegram(raw_response)
        message_chunks = smart_text_splitter(raw_response, MESSAGE_FORMATTING["max_chunk_length"], is_deep_mode=False)
        
        add_to_chat_history(context, user_input, raw_response)
        
        voice_response = await voice_processor.text_to_speech(raw_response, user_id)
        
        for i, chunk in enumerate(message_chunks):
            cleaned_chunk = clean_markdown(chunk)
            await context.bot.send_message(chat_id=chat_id, text=cleaned_chunk, parse_mode=None)

            if i == 0 and voice_response:
                keyboard = [
                    [
                        InlineKeyboardButton("1x", callback_data=f"speed_play_normal_{user_id}"),
                        InlineKeyboardButton("1.25x", callback_data=f"speed_play_fast_{user_id}"),
                        InlineKeyboardButton("1.5x", callback_data=f"speed_play_faster_{user_id}"),
                        InlineKeyboardButton("2x", callback_data=f"speed_play_fastest_{user_id}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_voice(chat_id=chat_id, voice=voice_response, reply_markup=reply_markup)
            
            if i < len(message_chunks) - 1:
                delay = MESSAGE_FORMATTING["chunk_delay"]
                if is_deep_mode:
                    delay *= 1.5
                await asyncio.sleep(delay)

    except Exception:
        if status_message:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
            except Exception:
                pass  # Mesaj zaten silinmiş olabilir
        await update.message.reply_text("❌ Sesli mesajı işlerken sorun oluştu. 🔄")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fotoğraf handler"""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    chat_history = initialize_chat_history(context)
    status_message = None
    
    try:
        user_ai = get_user_ai(user_id)
        is_deep_mode = getattr(user_ai, 'current_mode', 'simple') == "deep"
        
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text="☁️ Düşünüyorum...",
            parse_mode=None
        )
        
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_data = await file.download_as_bytearray()
        user_input = update.message.caption or "Bu görseli detaylıca analiz et ve açıkla."

        image_base64 = base64.b64encode(bytes(image_data)).decode('utf-8')

        ai_response, mode, status = await user_ai.process(
            user_input=user_input,
            chat_history=chat_history,
            image_data=image_base64
        )
        
        if isinstance(ai_response, dict):
            processed_response = ai_response.get('chat_response', status)
        else:
            processed_response = ai_response 

        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
        except Exception:
            pass
        
        raw_response = clean_response_for_user(processed_response)
        raw_response = format_response_for_telegram(raw_response)
        message_chunks = smart_text_splitter(raw_response, MESSAGE_FORMATTING["max_chunk_length"], is_deep_mode=False)
        
        add_to_chat_history(context, f"[Görsel] {user_input}", raw_response)
        
        for i, chunk in enumerate(message_chunks):
            cleaned_chunk = clean_markdown(chunk)
            await context.bot.send_message(chat_id=chat_id, text=cleaned_chunk, parse_mode=None)

            if i < len(message_chunks) - 1:
                delay = MESSAGE_FORMATTING["chunk_delay"]
                if is_deep_mode:
                    delay *= 1.5
                await asyncio.sleep(delay)

    except Exception:
        if status_message:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
            except Exception:
                pass  # Mesaj zaten silinmiş olabilir
        await update.message.reply_text("❌ Görseli analiz ederken sorun oluştu. 🔄")

async def speed_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ses hızı callback handler"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("speed_play_"):
        parts = query.data.split("_")
        speed_type = parts[2]
        target_user_id = int(parts[3])
        
        speed_mapping = {
            "normal": "normal",
            "fast": "hızlı",
            "faster": "çok_hızlı",
            "fastest": "çok_hızlı"
        }
        user_voice_speeds[target_user_id] = speed_mapping.get(speed_type, "normal")
        
        speed_text = {
            "normal": "1x",
            "fast": "1.25x",
            "faster": "1.5x",
            "fastest": "2x"
        }
        
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            f"⚡ Ses hızı **{speed_text[speed_type]}** olarak ayarlandı!",
            parse_mode='Markdown'
        )

# =========================================================
# BOT BAŞLATMA
# =========================================================
async def post_init(application: Application):
    """Bot komutlarını ayarla"""
    commands = [
        BotCommand("hizli_ai", "⚡ Hızlı AI"),
        BotCommand("yeni", "🔄 Sohbeti sıfırla"),
        BotCommand("menu", "📋 Kontrol Menüsü"),
    ]
    await application.bot.set_my_commands(commands)

def load_risale_system():
    """Risale arama sistemini yükle"""
    global risale_engine, risale_interface

    if risale_available:
        try:
            risale_engine = RisaleSearchEngine()
            risale_interface = RisaleTelegramInterface(risale_engine)
            return True
        except Exception:
            risale_engine = None
            risale_interface = None
            return False
    return False

# =========================================================
# MAIN FONKSİYON
# =========================================================
def main():
    """Ana fonksiyon"""
    print("=" * 70)
    print("🚀 PersonalAI Telegram Bot Başlatılıyor...")
    print("=" * 70)
    print()
    
    risale_loaded = load_risale_system()
    
    global chat_manager, chat_analyzer
    try:
        # Config'den Neo4j enabled durumunu oku
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            neo4j_enabled = config_data.get('neo4j', {}).get('enabled', False)
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            print(f"Config okuma hatası: {e}")
            neo4j_enabled = False

        neo4j_pass = os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PASSWORD") or "senegal5454"
        chat_manager = ChatHistoryManager(neo4j_uri="bolt://localhost:7687", password=neo4j_pass, enabled=neo4j_enabled)
        chat_analyzer = ChatDataAnalyzer(neo4j_uri="bolt://localhost:7687", password=neo4j_pass, enabled=neo4j_enabled)
        if neo4j_enabled:
            print("✅ Neo4j Chat Manager başlatıldı")
    except Exception as e:
        print(f"⚠️ Neo4j Chat Manager başlatılamadı: {e}")
        chat_manager = None
        chat_analyzer = None
    
    global quantum_tree
    if quantum_available:
        quantum_initialized = initialize_quantum_tree()
        if quantum_initialized:
            print("✅ QuantumTree başarıyla başlatıldı!")
            print("🌟 Derin mod (QuantumTree + PersonalAI) aktif!")
        else:
            print("⚠️ QuantumTree başlatılamadı")
            print("⚡ Sadece PersonalAI hızlı mod aktif olacak")
    else:
        print("⚠️ QuantumTree modülü mevcut değil")
        print("⚡ Sadece PersonalAI hızlı mod aktif olacak")
    
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        print("=" * 70)
        print("❌ TELEGRAM_TOKEN ortam değişkeni bulunamadı!")
        print("📝 .env dosyasına ekle: TELEGRAM_TOKEN=your_bot_token")
        print("=" * 70)
        sys.exit(1)
    
    try:
        application = Application.builder().token(telegram_token).post_init(post_init).build()
        
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("yazili", yazili_mod_command))
        application.add_handler(CommandHandler("sesli", sesli_mod_command))
        application.add_handler(CommandHandler("hizli_ai", hizli_ai_command))
        application.add_handler(CommandHandler("derin_ai", derin_ai_command))
        application.add_handler(CommandHandler("risale", risale_command))
        application.add_handler(CommandHandler("yeni", new_chat_command))
        application.add_handler(CommandHandler("gecmis", gecmis_command))
        application.add_handler(CommandHandler("ara", ara_command))
        application.add_handler(CommandHandler("iptal", iptal_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(CommandHandler("menu", menu_command))

        application.add_handler(CallbackQueryHandler(speed_callback_handler, pattern="^speed_play_"))
        application.add_handler(CallbackQueryHandler(handle_pc_control_callbacks, pattern="^pc_"))
        application.add_handler(CallbackQueryHandler(handle_risale_callbacks, pattern="^set_mode_"))
        application.add_handler(CallbackQueryHandler(handle_risale_callbacks))
        
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
        
        start_cleanup_timer()
        
        print()
        print("=" * 70)
        print("✅ TELEGRAM BOT BAŞARILI!")
        print("=" * 70)
        print()
        print("🤖 ANA ÖZELLİKLER:")
        print("  • PersonalAI Full Integration (personal_ai.py)")
        print("  • Multi-modal Support (Text + Voice + Images)")
        print("  • Kullanıcı Yalıtımı (Her kullanıcı için ayrı instance)")
        print("  • Persistent Memory System")
        print(f"  • ⏰ TIMEOUT: {PROCESS_TIMEOUT} saniye")
        if risale_loaded:
            print("  • Risale-i Nur Arama Sistemi ✅")
        else:
            print("  • Risale-i Nur Arama Sistemi ❌")
        print()
        print("🧠 HAFIZA SİSTEMİ:")
        print("  • Telegram Session Memory (Chat History)")
        print("  • PersonalAI Internal Memory (Smart Context)")
        print("  • Neo4j GraphRAG (Long-term Memory)")
        print("  • Pasif Kullanıcı Temizliği (Her 6 saatte)")
        print()
        print("📱 KOMUTLAR:")
        print("  • /start - Bot'u başlat")
        print("  • /yazili / /sesli - Mod değiştir")
        print("  • /hizli_ai - Hızlı mod (PersonalAI)")
        print("  • /derin_ai - Derin analiz (QuantumTree)")
        print("  • /risale - Risale-i Nur arama")
        print("  • /yeni - Konuşmayı sıfırla")
        print("  • /gecmis - Sohbet özeti")
        print("  • /ara - Geçmişte arama")
        print("  • /stats - Admin istatistikleri")
        print()
        print("🛑 Durdurmak için Ctrl+C tuşuna basın.")
        print("=" * 70)
        print()
        
        application.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("👋 PersonalAI Telegram Bot kullanıcı tarafından durduruldu.")
        print("=" * 70)
        print()
        
        print("🧹 Cleanup işlemleri başlıyor...")
        
        if quantum_tree:
            try:
                quantum_tree.stop_background()
                print("  ✅ QuantumTree kapatıldı")
            except Exception as e:
                print(f"  ⚠️ QuantumTree kapatma hatası: {e}")

        if chat_manager and chat_manager.driver:
            try:
                chat_manager.driver.close()
                print("  ✅ Neo4j chat manager kapatıldı")
            except Exception as e:
                print(f"  ⚠️ Neo4j chat manager kapatma hatası: {e}")

        if chat_analyzer and chat_analyzer.driver:
            try:
                chat_analyzer.close()
                print("  ✅ Chat analyzer kapatıldı")
            except Exception as e:
                print(f"  ⚠️ Chat analyzer kapatma hatası: {e}")
        
        for telegram_id, ai_instance in list(ai_instances.items()):
            if hasattr(ai_instance, 'ai_core_instance') and ai_instance.ai_core_instance:
                try:
                    if hasattr(ai_instance.ai_core_instance, 'close'):
                        ai_instance.ai_core_instance.close()
                    print(f"  ✅ PersonalAI instance {telegram_id} cleanup completed")
                    del ai_instances[telegram_id]
                except Exception as e:
                    print(f"  ⚠️ PersonalAI cleanup error (user {telegram_id}): {e}")
        
        print()
        print("✅ Tüm cleanup işlemleri tamamlandı.")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Kritik hata: {e}")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    main()