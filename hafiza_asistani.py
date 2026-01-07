from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import requests
import json
import os
import re
import asyncio
import aiohttp
import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from web_search import WebSearch
_web_searcher = WebSearch()  # Global instance

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from topic_memory import TopicMemory
from conversation_context import ConversationContextManager
from profile_manager import ProfileManager


def get_current_datetime() -> Dict[str, str]:
    """Türkiye saati ile şu anki tarih ve saati getir"""
    try:
        tz = ZoneInfo("Europe/Istanbul")
        now = datetime.now(tz)

        ay_isimleri = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
            5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
            9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }

        gun_isimleri = {
            0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe",
            4: "Cuma", 5: "Cumartesi", 6: "Pazar"
        }

        ay = ay_isimleri[now.month]
        gun = gun_isimleri[now.weekday()]
        saat = now.hour

        if 0 <= saat < 6:
            zaman_dilimi = "gece"
        elif 6 <= saat < 12:
            zaman_dilimi = "sabah"
        elif 12 <= saat < 18:
            zaman_dilimi = "öğleden sonra"
        else:
            zaman_dilimi = "akşam"

        cuma_notu = " (Cuma)" if now.weekday() == 4 else ""

        return {
            "tarih": f"{now.day} {ay} {now.year}",
            "gun": gun,
            "saat": now.strftime("%H:%M"),
            "full": f"{now.day} {ay} {now.year} {gun}, Saat: {now.strftime('%H:%M')}",
            "zaman_dilimi": zaman_dilimi + cuma_notu,
            "saat_int": saat,
        }
    except Exception:
        return {
            "tarih": "Bilinmiyor",
            "gun": "Bilinmiyor",
            "saat": "Bilinmiyor",
            "full": "Tarih/saat bilgisi alınamadı",
            "zaman_dilimi": "",
            "saat_int": 12,
        }


def calculate_math(expression: str) -> str:
    """Matematiksel ifadeyi güvenli şekilde hesapla"""
    import math

    try:
        safe_expression = expression.strip()

        safe_expression = safe_expression.replace("x", "*")
        safe_expression = safe_expression.replace("X", "*")
        safe_expression = safe_expression.replace("çarpı", "*")
        safe_expression = safe_expression.replace("çarp", "*")
        safe_expression = safe_expression.replace("bölü", "/")
        safe_expression = safe_expression.replace("artı", "+")
        safe_expression = safe_expression.replace("eksi", "-")

        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in safe_expression):
            return "❌ Güvenlik: Sadece sayılar ve matematiksel operatörler kullanılabilir."

        safe_dict = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        result = eval(safe_expression, {"__builtins__": {}}, safe_dict)

        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.4f}".rstrip("0").rstrip(".")

        return str(result)

    except ZeroDivisionError:
        return "❌ Hata: Sıfıra bölme yapılamaz!"
    except Exception:
        return "❌ Hesaplama hatası: Geçersiz matematiksel ifade."


async def web_ara(query: str, context: str = "") -> str:
    """
    Tavily API ile internet araması.

    Args:
        query: Arama sorgusu
        context: Opsiyonel bağlam
    """
    try:
        search_query = query
        if context:
            search_query = f"{query} {context}"

        print(f"\n🌐 Web araması: '{search_query}'")

        result = _web_searcher.quick_answer(search_query)

        if result and "Arama hatasi" not in result and "Sonuc bulunamadi" not in result:
            print(f"   ✅ Sonuç bulundu")
            return result

        return f"❌ '{query}' için bilgi bulunamadı."

    except Exception as e:
        print(f"❌ Web arama hatası: {e}")
        return f"❌ Arama hatası: {str(e)}"


async def wiki_ara(query: str) -> str:
    """Eski isim - web_ara'ya yönlendirir."""
    return await web_ara(query)


async def get_weather(city: str) -> str:
    """Şehir için hava durumu bilgisi getir"""
    try:
        city = (
            city.replace("hava durumu", "")
            .replace("hava", "")
            .replace("nasıl", "")
            .strip()
        )

        api_key = os.getenv("OPENWEATHER_API_KEY", "")

        if not api_key:
            return await get_weather_fallback(city)

        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},TR",
            "appid": api_key,
            "units": "metric",
            "lang": "tr",
        }

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return await get_weather_fallback(city)
                data = await response.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed = data["wind"]["speed"]

        result = "[KORUNACAK_FORMAT]\n"
        result += f"🌤️ {city.title()} Hava Durumu\n"
        result += f"{'─' * 32}\n\n"
        result += f"☁️ Durum:       {description}\n"
        result += f"🌡️ Sıcaklık:    {temp:.1f}°C\n"
        result += f"🤚 Hissedilen:  {feels_like:.1f}°C\n"
        result += f"💨 Rüzgar:      {wind_speed:.1f} m/s\n"
        result += f"💧 Nem:         {humidity}%\n"
        result += "[/KORUNACAK_FORMAT]"

        return result

    except Exception:
        return await get_weather_fallback(city)


async def get_weather_fallback(city: str) -> str:
    """Fallback: hava durumu - Web search kaldırıldı"""
    return f"❌ {city} için hava durumu servisi kullanılamıyor. Web arama devre dışı."


async def get_prayer_times(city: str, specific_prayer: str = None) -> str:
    """Şehir için namaz vakitlerini getir (Aladhan API)"""
    try:
        city = (
            city.replace("namaz vakitleri", "")
            .replace("namaz vakti", "")
            .replace("ezan", "")
            .strip()
        )
        city = (
            city.replace("’da", "")
            .replace("’de", "")
            .replace("’ın", "")
            .replace("’in", "")
            .strip()
        )

        url = "http://api.aladhan.com/v1/timingsByCity"
        params = {
            "city": city,
            "country": "Turkey",
            "method": 13,
        }

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return f"❌ {city} şehri bulunamadı."
                data = await response.json()

        if data.get("code") != 200:
            return f"❌ {city} için namaz vakitleri alınamadı."

        timings = data["data"]["timings"]
        date_info = data["data"]["date"]["gregorian"]

        prayer_names = {
            "Fajr": ("İmsak", "🌙"),
            "Sunrise": ("Güneş", "☀️"),
            "Dhuhr": ("Öğle", "🌤️"),
            "Asr": ("İkindi", "🌅"),
            "Maghrib": ("Akşam", "🌆"),
            "Isha": ("Yatsı", "🌃"),
        }

        if specific_prayer:
            specific_prayer = specific_prayer.lower().strip()
            prayer_map = {
                "imsak": "Fajr",
                "güneş": "Sunrise",
                "öğle": "Dhuhr",
                "ikindi": "Asr",
                "akşam": "Maghrib",
                "yatsı": "Isha",
            }

            for tr_name, eng_name in prayer_map.items():
                if tr_name in specific_prayer:
                    time_value = timings[eng_name]
                    turkish_name, emoji = prayer_names[eng_name]
                    return f"{emoji} {city.title()} {turkish_name} namazı: {time_value}"

        result = "[KORUNACAK_FORMAT]\n"
        result += f"🕌 {city.title()} Namaz Vakitleri\n"
        result += (
            f"📅 {date_info['day']} {date_info['month']['en']} {date_info['year']}\n"
        )
        result += f"{'─' * 32}\n\n"

        for eng_name, (turkish_name, emoji) in prayer_names.items():
            time_value = timings[eng_name]
            padded_name = f"{turkish_name:<8}"
            result += f"{emoji} {padded_name} {time_value}\n"

        result += "[/KORUNACAK_FORMAT]"
        return result.strip()

    except Exception as e:
        return f"❌ Namaz vakitleri alınamadı: {str(e)}"



_ToolSystem = None

def get_tool_system_class():
    """ToolSystem'i lazy import et (circular import önlemi)"""
    global _ToolSystem
    if _ToolSystem is None:
        try:
            from personal_ai import ToolSystem
            _ToolSystem = ToolSystem
        except ImportError:
            class FallbackToolSystem:
                TOOLS = {
                    "risale_ara": {"name": "risale_ara", "description": "Dini sorulara cevap", "parameters": "soru", "when": "Dini konularda", "examples": ["İman nedir?"]},
                    "zaman_getir": {"name": "zaman_getir", "description": "Tarih/saat", "parameters": "yok", "when": "Zaman sorulduğunda", "examples": ["Saat kaç?"]},
                    "hesapla": {"name": "hesapla", "description": "Hesaplama", "parameters": "ifade", "when": "Matematik sorulduğunda", "examples": ["2+2"]},
                    "hava_durumu": {"name": "hava_durumu", "description": "Hava durumu", "parameters": "şehir", "when": "Hava sorulduğunda", "examples": ["İstanbul hava"]},
                    "namaz_vakti": {"name": "namaz_vakti", "description": "Namaz vakitleri", "parameters": "şehir", "when": "Namaz vakti sorulduğunda", "examples": ["Ankara namaz"]},
                    "web_ara": {"name": "web_ara", "description": "İnternette bilgi veya haber ara", "parameters": "arama terimi", "when": "Bilmediğin konu, güncel haber, kişi, yer, olay sorulduğunda", "examples": ["Einstein kimdir", "son haberler", "Python nedir"]},
                    "yok": {"name": "yok", "description": "Direkt cevap", "parameters": "yok", "when": "Genel sohbet", "examples": ["Merhaba"]},
                }
                @staticmethod
                def get_tools_prompt() -> str:
                    tools_text = "ARAÇLAR:\n"
                    for name, info in FallbackToolSystem.TOOLS.items():
                        tools_text += f"{name}: {info['description']}\n"
                    return tools_text
                @staticmethod
                def get_tool_calling_prompt(user_input: str) -> str:
                    return f"{FallbackToolSystem.get_tools_prompt()}\nSORU: {user_input}\nARAÇ:"
                @staticmethod
                def parse_tool_decision(llm_response: str) -> Tuple[str, str]:
                    tool_name = "yok"
                    tool_param = ""
                    for line in llm_response.split("\n"):
                        if line.startswith("ARAÇ:"): tool_name = line.replace("ARAÇ:", "").strip().lower()
                        elif line.startswith("PARAMETRE:"): tool_param = line.replace("PARAMETRE:", "").strip()
                    if tool_name not in FallbackToolSystem.TOOLS: tool_name = "yok"
                    if tool_param.lower() == "yok": tool_param = ""
                    return tool_name, tool_param
            _ToolSystem = FallbackToolSystem
    return _ToolSystem


class ToolSystem:
    """
    ToolSystem wrapper - personal_ai.py'daki ToolSystem'e yönlendirir

    NOT: Asıl implementasyon personal_ai.py'da (tek kaynak)
    """

    @property
    def TOOLS(self):
        return get_tool_system_class().TOOLS

    @staticmethod
    def get_tools_prompt() -> str:
        return get_tool_system_class().get_tools_prompt()

    @staticmethod
    def get_tool_calling_prompt(user_input: str) -> str:
        return get_tool_system_class().get_tool_calling_prompt(user_input)

    @staticmethod
    def parse_tool_decision(llm_response: str) -> Tuple[str, str]:
        return get_tool_system_class().parse_tool_decision(llm_response)





_ROLES_CACHE = None

def get_roles():
    """ROLES'u personal_ai.py'dan al - artık tek basit rol"""
    global _ROLES_CACHE
    if _ROLES_CACHE is None:
        try:
            from personal_ai import SystemConfig
            _ROLES_CACHE = SystemConfig.ROLES
        except ImportError:
            # Fallback: tek basit rol
            _ROLES_CACHE = {
                "default": {"keywords": [], "tone": "natural", "response_style": "adaptive"}
            }
    return _ROLES_CACHE


_MultiRoleSystem = None

def get_multi_role_system_class():
    """MultiRoleSystem'i lazy import et - artık sadeleştirilmiş"""
    global _MultiRoleSystem
    if _MultiRoleSystem is None:
        try:
            from personal_ai import MultiRoleSystem as _MRS
            _MultiRoleSystem = _MRS
        except ImportError:
            class FallbackMultiRoleSystem:
                def __init__(self):
                    self.enabled = False  # Devre dışı
                @property
                def ROLES(self):
                    return get_roles()
                def detect_role(self, user_input: str) -> str:
                    return "default"  # Her zaman default
            _MultiRoleSystem = FallbackMultiRoleSystem
    return _MultiRoleSystem


class MultiRoleSystem:
    """
    Sadeleştirilmiş MultiRoleSystem - tek tutarlı kişilik
    """

    def __init__(self):
        self._impl = get_multi_role_system_class()()

    @property
    def ROLES(self):
        return get_roles()

    def detect_role(self, user_input: str) -> str:
        return "default"  # Artık her zaman default döner



class SimpleFAISSKB:
    """
    Basit FAISS KB wrapper
    (Gerçek FAISS KB PersonalAI’dan alınır ve buraya inject edilir)
    """

    def __init__(self):
        self.enabled = False
        self.faiss_kb = None

    def set_faiss_kb(self, faiss_kb):
        """Gerçek FAISS KB'yi inject et"""
        self.faiss_kb = faiss_kb
        self.enabled = (
            faiss_kb is not None and hasattr(faiss_kb, "enabled") and faiss_kb.enabled
        )

    def get_relevant_context(self, query: str, max_chunks: int = 6) -> str:
        """FAISS'ten ilgili bağlamı getir"""
        if not self.enabled or not self.faiss_kb:
            return ""
        try:
            return self.faiss_kb.get_relevant_context(query, max_chunks)
        except Exception as e:
            print(f"❌ FAISS hatası: {e}")
            return ""



class DecisionLLM:
    """Together.ai API ile akıllı karar verme (Llama 70B)"""

    def __init__(self, api_key: str = None, model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.model = model
        self.base_url = "https://api.together.xyz/v1/completions"

        if not self.api_key:
            raise ValueError("❌ TOGETHER_API_KEY bulunamadı! .env dosyasını kontrol edin.")

        if not self._try_connect():
            raise ConnectionError("❌ Together.ai API'sine bağlanılamadı!")

        print(f"🧠 DecisionLLM başlatıldı (Model: {model}, Together.ai)")

    def _try_connect(self) -> bool:
        """Together.ai API bağlantısını test et"""
        try:
            response = requests.get(
                "https://api.together.xyz/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except (requests.RequestException, requests.Timeout) as e:
            print(f"Together API bağlantı hatası: {e}")
            return False

    def _call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """Together.ai API'sine prompt gönder"""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                },
                timeout=15,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            return ""
        except Exception as e:
            print(f"❌ DecisionLLM hatası: {e}")
            return ""

    def extract_topics(self, query: str, max_topics: int = 3) -> List[str]:
        """Konuları akıllıca çıkar"""
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Kullanıcı mesajı: "{query}"

GÖREV: ANA KONULARI bul (maksimum {max_topics} adet)

KURALLAR:

- Uzun kelimeler değil, ANLAMLI konular
- Her satıra 1 konu
- Alakasız kelime ekleme

KONULAR:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        response = self._call_llm(prompt, max_tokens=50)

        topics = [
            line.strip().strip("-•*").strip()
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 3
        ]

        return topics[:max_topics]



class HafizaAsistani:
    """
    🧠 Gelişmiş Hafıza Asistanı v3.0 - GERÇEK SEKRETER

    ÖZELLİKLER:
    - Benzerlik tabanlı arama (BGE-M3)
    - Tool System entegrasyonu
    - Web Search
    - FAISS KB erişimi
    - Multi-Role System
    - Akıllı prompt hazırlama
    - DecisionLLM ile karar verme
    """

    def __init__(
        self,
        saat_limiti: int = 48,
        esik: float = 0.50,
        max_mesaj: int = 20,
        model_adi: str = "BAAI/bge-m3",
        use_decision_llm: bool = True,
        together_api_key: str = None,
        decision_model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    ):
        print("=" * 60)
        print("🧠 HafizaAsistani v3.0 - Genişletilmiş Sekreter")
        print("=" * 60)

        self.together_api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
        self.decision_model = decision_model

        print("📦 Embedding modeli yükleniyor...")
        self.embedder = SentenceTransformer(model_adi)
        print(f"✅ Model '{model_adi}' yüklendi!")

        self.hafiza: List[Dict[str, Any]] = []
        self.saat_limiti = saat_limiti * 3600
        self.esik = esik
        self.max_mesaj = max_mesaj

        if not use_decision_llm:
            raise ValueError("❌ DecisionLLM zorunludur!")

        try:
            self.decision_llm = DecisionLLM(api_key=self.together_api_key, model=decision_model)
            self.use_decision_llm = True
            print("✅ DecisionLLM aktif!")
        except Exception as e:
            raise RuntimeError(f"DecisionLLM başlatılamadı: {e}")

        self.tool_system = ToolSystem()
        print("✅ Tool System aktif!")

        self.multi_role = MultiRoleSystem()
        print("✅ Multi-Role System aktif!")

        self.faiss_kb = SimpleFAISSKB()
        print("✅ FAISS KB wrapper hazır (inject edilecek)!")

        self.closed_topics: List[Dict[str, Any]] = []
        self.max_closed_topics = 20  # En fazla 20 kapanan konu tut
        print("✅ Closed Topics Tracker aktif!")

        self.topic_memory = TopicMemory(
            user_id="murat",  # Sabit kullanıcı
            base_dir="user_data",
            together_api_key=self.together_api_key,
            together_model=decision_model,
            embedding_model=model_adi  # Aynı embedding modelini kullan
        )
        print("✅ Topic Memory aktif!")

        self._injected_categories = {}  # {category_id: message_count_when_injected}
        self._message_counter = 0  # Toplam mesaj sayacı
        self._injection_cooldown = 5  # Kaç mesaj sonra tekrar enjekte edilebilir

        self.conversation_context = ConversationContextManager(
            user_id="murat",  # Sabit kullanıcı
            base_dir="user_data",
            together_api_key=self.together_api_key,
            together_model=decision_model,
            archive_to_faiss=False  # Şimdilik dosya bazlı arşivleme
        )
        print("✅ Conversation Context aktif!")

        # Kullanıcı Profili
        self.profile_manager = ProfileManager(
            user_id="murat",  # Sabit kullanıcı
            base_dir="user_data"
        )
        if self.profile_manager.has_profile():
            print(f"✅ Kullanıcı Profili yüklendi: {self.profile_manager.get_name()}")
        else:
            print("✅ Kullanıcı Profili aktif (henüz boş)")

        print("\n⚙️ Sekreter Ayarları:")
        print(f"   • Zaman limiti: {saat_limiti} saat")
        print(f"   • Benzerlik eşiği: {esik}")
        print(f"   • Max mesaj: {max_mesaj}")
        print("   • Tool System: ✅")
        print("   • Web Arama (web_ara): ✅")
        print("   • Multi-Role: ✅")
        print("   • DecisionLLM: ✅")
        print("   • Topic Memory (v2.0): ✅")
        print("=" * 60 + "\n")


    def mesaj_ekle(self, mesaj: str, rol: str = "user"):
        """Yeni mesajı vektörleştirip hafızaya ekler"""
        vektor = self.embedder.encode(mesaj)
        self.hafiza.append(
            {"rol": rol, "mesaj": mesaj, "zaman": time.time(), "vektor": vektor}
        )
        self._eski_mesajlari_sil()

    def add(self, user_message: str, ai_response: str, chat_history: List[Dict] = None):
        """
        Kullanıcı ve AI mesajlarını hafızaya ekler.
        Ayrıca ConversationContext'i de günceller.

        Args:
            user_message: Kullanıcı mesajı
            ai_response: AI yanıtı
            chat_history: Opsiyonel sohbet geçmişi (context için)
        """
        self.mesaj_ekle(user_message, rol="user")
        self.mesaj_ekle(ai_response, rol="assistant")

        if self.conversation_context and chat_history:
            try:
                result = self.conversation_context.process_message(
                    user_message, ai_response, chat_history
                )
                if result.get("new_session_started"):
                    print("🔄 Yeni konu tespit edildi, session değiştirildi")

                    if len(self.hafiza) > 12:
                        tampon_bolge = self.hafiza[:-12]  # 12'den eski mesajlar
                        if tampon_bolge and self.topic_memory:
                            tampon_text = "\n".join([
                                f"[{m['rol'].upper()}]: {m['mesaj']}"
                                for m in tampon_bolge if m.get('mesaj')
                            ])
                            topic_summary = result.get('current_summary', '') or tampon_text[:200]
                            if topic_summary:
                                print(f"💾 Tampon bölge TopicMemory'ye kaydediliyor ({len(tampon_bolge)} mesaj)")
                                self.add_closed_topic(topic_summary, chat_history)

                    if len(self.hafiza) > 4:
                        self.hafiza = self.hafiza[-4:]
                        print("🧹 Hafıza temizlendi (son 4 mesaj kaldı - bağlam korundu)")
                elif result.get("summary_updated"):
                    print(f"📝 Konu özeti güncellendi: {result.get('current_summary', '')[:50]}...")
            except Exception as e:
                print(f"⚠️ ConversationContext güncelleme hatası: {e}")

    def _eski_mesajlari_sil(self):
        """Belirlenen süreyi geçen mesajları temizler"""
        simdi = time.time()
        eski_uzunluk = len(self.hafiza)
        self.hafiza = [
            m for m in self.hafiza if (simdi - m["zaman"]) < self.saat_limiti
        ]

        silinen = eski_uzunluk - len(self.hafiza)
        if silinen > 0:
            print(
                f"🧹 {silinen} eski mesaj temizlendi ({self.saat_limiti/3600:.0f} saat sınırı)"
            )

    def _search_internal(self, query: str, k: int) -> List[Dict[str, str]]:
        """
        İç semantik arama fonksiyonu (TEK KAYNAK)
        search() ve ilgili_mesajlari_bul() bunu kullanır
        Returns: [{"rol": "user", "mesaj": "..."}, ...]
        """
        if not self.hafiza or not query:
            return []

        try:
            query_vector = self.embedder.encode([query], convert_to_numpy=True)

            mesaj_skorlari = []
            simdi = time.time()

            for eski_mesaj in self.hafiza:
                benzerlik = cosine_similarity(
                    query_vector.reshape(1, -1),
                    eski_mesaj["vektor"].reshape(1, -1),
                )[0][0]

                zaman_farki = simdi - eski_mesaj["zaman"]
                zaman_agirligi = 1.0 / (1.0 + (zaman_farki / 3600))

                skor = benzerlik * (0.7 + 0.3 * zaman_agirligi)

                if skor > self.esik:
                    mesaj_skorlari.append(
                        {
                            "mesaj": eski_mesaj["mesaj"],
                            "rol": eski_mesaj["rol"],
                            "skor": skor,
                            "entry": eski_mesaj,
                        }
                    )

            mesaj_skorlari.sort(key=lambda x: x["skor"], reverse=True)
            mesaj_skorlari = mesaj_skorlari[:k]
            mesaj_skorlari.sort(key=lambda x: x["entry"]["zaman"])

            return [
                {"rol": m["entry"]["rol"], "mesaj": m["entry"]["mesaj"]}
                for m in mesaj_skorlari
            ]
        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []

    def search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Hafızada semantik arama (SADECE kısa dönem - mevcut sohbet)

        NOT: TopicMemory (uzun dönem) araması ayrı yapılıyor:
        - get_silent_long_term_context() ile sessiz enjeksiyon
        """
        k = max_results or self.max_mesaj
        ilgili_mesajlar = self._search_internal(query, k)

        if ilgili_mesajlar:
            context_parts = []
            for m in ilgili_mesajlar:
                context_parts.append(f"- {m['rol']}: {m['mesaj']}")
            return "İlgili geçmiş konuşmalar:\n" + "\n".join(context_parts)

        return ""

    def get_silent_long_term_context(self, query: str) -> str:
        """
        🔇 SILENT CONTEXT INJECTION (with cooldown)

        TopicMemory'den hızlı kategori eşleşmesi yap.
        Eşleşme varsa, sessizce LLM'e arka plan bilgisi olarak ver.

        COOLDOWN: Aynı kategori son 5 mesajda enjekte edildiyse tekrar enjekte etme.
        Böylece sohbet akışında aynı bilgi sürekli tekrarlanmaz.

        Bu bilgi:
        - Kullanıcıya gösterilMEZ
        - LLM'e system context olarak verilir
        - LLM bu bilgiyi zorla hatırlatmaz, sadece cevap kalitesi için kullanır

        Returns:
            str: Silent context (boş olabilir)
        """
        if not self.topic_memory:
            print(f"   🔇 TopicMemory yok!")
            return ""

        try:
            self._message_counter += 1

            cat_count = len(self.topic_memory.index.get("categories", {}))
            print(f"   🔇 TopicMemory kontrol: {cat_count} kategori mevcut")

            context = self.topic_memory.get_context_for_query(query, max_sessions=2)

            if context:
                import re
                category_match = re.search(r'\[([^\]]+)\]', context)
                if category_match:
                    category_id = category_match.group(1)

                    if category_id in self._injected_categories:
                        last_injection = self._injected_categories[category_id]
                        messages_since = self._message_counter - last_injection

                        if messages_since < self._injection_cooldown:
                            print(f"   🔇 TopicMemory: '{category_id}' cooldown'da ({messages_since}/{self._injection_cooldown} mesaj)")
                            return ""  # Cooldown'daysa enjekte etme

                    self._injected_categories[category_id] = self._message_counter
                    print(f"   🔇 Silent long-term context bulundu ({len(context)} karakter) - cooldown başladı")
                    return context
                else:
                    print(f"   🔇 Silent long-term context bulundu ({len(context)} karakter)")
                    return context
            else:
                print(f"   🔇 TopicMemory: eşleşme yok")
                return ""

        except Exception as e:
            print(f"   ⚠️ Silent context hatası: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def should_check_long_term_memory(self, user_input: str) -> bool:
        """
        Uzun dönem hafıza kontrolü gerekli mi?

        True döndüren durumlar:
        1. Kullanıcı geçmişe referans veriyor
        2. Soru mevcut kategori konularıyla alakalı olabilir
        """
        user_lower = user_input.lower()

        past_references = [
            "daha önce", "geçen sefer", "hatırlıyor musun",
            "konuşmuştuk", "sormuştum", "demiştin", "söylemiştin",
            "geçen", "önceki", "bahsetmiştik", "anlatmıştın"
        ]

        if any(ref in user_lower for ref in past_references):
            print(f"   📌 Geçmiş referansı tespit edildi")
            return True

        if len(user_input) > 15 and self.topic_memory.index.get("categories"):
            return True

        return False

    def get_conversation_context(self) -> str:
        """
        🧠 CONVERSATION CONTEXT INJECTION

        LLM tabanlı konu özeti sisteminden bağlam al.
        Bu özet, embedding tabanlı değil LLM tabanlı olduğu için
        semantik olarak ilişkili ama farklı kelimelere sahip konuları
        (örn: Allah'ın ilmi → kader → irade) doğru şekilde takip eder.

        Returns:
            str: Conversation context (boş olabilir)
        """
        if not self.conversation_context:
            return ""

        try:
            context = self.conversation_context.get_context_for_prompt()
            if context:
                print(f"   🧠 Conversation context bulundu ({len(context)} karakter)")
            return context
        except Exception as e:
            print(f"   ⚠️ Conversation context hatası: {e}")
            return ""


    def clear(self):
        """Tüm hafızayı temizle"""
        self.hafiza = []
        self.closed_topics = []

        if self.conversation_context:
            self.conversation_context.clear()

        print("✅ Hafıza, kapanan konular ve ConversationContext tamamen temizlendi")


    def add_closed_topic(self, topic_summary: str, chat_history: List[Dict] = None):
        """
        Kapanan konuyu listeye ekle + TopicMemory'ye kaydet
        Bir sonraki soruda aynı konuya dönmemek için kullanılır

        NOT: TopicMemory otomatik kalite kontrolü yapar:
        - En az 3 anlamlı mesaj gerekli
        - "merhaba/teşekkürler" gibi mesajlar sayılmaz
        - Aynı gün aynı kategori → günceller (duplicate olmaz)
        """
        if not topic_summary or len(topic_summary.strip()) < 2:
            return

        last_context = ""
        if chat_history and len(chat_history) >= 2:
            last_msgs = chat_history[-4:]
            last_context = " | ".join([
                (m.get("content") or "")[:50] for m in last_msgs
            ])

        closed_entry = {
            "summary": topic_summary.strip(),
            "context": last_context[:200],
            "timestamp": time.time(),
            "vector": self.embedder.encode(topic_summary)
        }

        self.closed_topics.append(closed_entry)

        if len(self.closed_topics) > self.max_closed_topics:
            self.closed_topics = self.closed_topics[-self.max_closed_topics:]

        print(f"📕 Konu kapandı: '{topic_summary}'")
        print(f"   📊 Chat history uzunluğu: {len(chat_history) if chat_history else 0} mesaj")

        if chat_history and len(chat_history) >= 2:
            print(f"   💾 TopicMemory.save_topic() çağrılıyor...")
            saved = self.topic_memory.save_topic(
                messages=chat_history,
                topic_hint=topic_summary
            )
            if saved:
                print(f"   ✅ Uzun dönem hafızaya kaydedildi: [{saved.get('category_name', 'Genel')}] - {saved.get('summary', topic_summary)[:50]}...")
            else:
                print(f"   ⏩ Uzun dönem hafıza: Kalite kontrolünden geçmedi (kısa/yüzeysel konuşma)")
        else:
            print(f"   ⏩ TopicMemory atlandı: Yetersiz mesaj ({len(chat_history) if chat_history else 0} < 2)")

    def is_topic_closed(self, user_input: str, threshold: float = 0.75) -> Tuple[bool, str]:
        """
        Kullanıcının sorduğu soru kapanmış bir konuya mı ait?
        Returns: (is_closed, closed_topic_summary)
        """
        if not self.closed_topics or not user_input:
            return False, ""

        try:
            query_vector = self.embedder.encode([user_input], convert_to_numpy=True)

            for closed in self.closed_topics:
                similarity = cosine_similarity(
                    query_vector.reshape(1, -1),
                    closed["vector"].reshape(1, -1)
                )[0][0]

                if similarity >= threshold:
                    print(f"⚠️ Kapanmış konuya benzerlik: {similarity:.2f} - '{closed['summary']}'")
                    return True, closed["summary"]

            return False, ""
        except Exception as e:
            print(f"⚠️ Kapanmış konu kontrolü hatası: {e}")
            return False, ""

    def get_closed_topics_summary(self) -> str:
        """Kapanan konuların listesini döndür (prompt için)"""
        if not self.closed_topics:
            return ""

        summaries = [c["summary"] for c in self.closed_topics[-5:]]  # Son 5 konu
        return "Kapanan konular (tekrar açma): " + ", ".join(summaries)

    def _user_wants_to_reopen_topic(self, user_input: str) -> bool:
        """
        Kullanıcı kapanmış bir konuyu TEKRAR AÇMAK mı istiyor?

        "Tekrar sor" sinyalleri:
        - "Tekrar soruyorum..."
        - "Bir daha açıklar mısın..."
        - "Yine aynı konuya dönmek istiyorum"
        - Açık soru işareti ile soru sormak (?)

        Returns: True = kullanıcı konuyu tekrar açmak istiyor
        """
        user_lower = user_input.lower().strip()

        reopen_signals = [
            "tekrar",
            "yine",
            "bir daha",
            "yeniden",
            "açıkla",
            "anlat",
            "detay",
            "daha fazla",
            "devam et",
            "ne demiştin",
            "hatırlamıyorum",
            "unuttum",
        ]

        has_question_mark = "?" in user_input
        is_long_enough = len(user_input) > 15

        if any(signal in user_lower for signal in reopen_signals):
            return True

        if has_question_mark and is_long_enough:
            return True

        return False



    async def _tool_calistir(
        self, tool_name: str, tool_param: str, user_input: str
    ) -> Optional[str]:
        """Seçilen aracı çalıştır ve sonucu döndür"""
        if tool_name == "yok":
            return None

        print(f"🛠️ Araç çalıştırılıyor: {tool_name}({tool_param or 'auto'})")

        try:
            if tool_name == "zaman_getir":
                datetime_info = get_current_datetime()
                result = "[KORUNACAK_FORMAT]\n"
                result += "🕐 Şu Anki Zaman\n"
                result += f"{'─' * 32}\n\n"
                result += f"📅 Tarih:  {datetime_info['tarih']}\n"
                result += f"📆 Gün:    {datetime_info['gun']}\n"
                result += f"🕐 Saat:   {datetime_info['saat']}\n"
                result += "[/KORUNACAK_FORMAT]"
                return result

            if tool_name == "hesapla":
                result = calculate_math(tool_param or user_input)
                return f"🧮 {result}"

            if tool_name == "hava_durumu":
                city = tool_param or user_input
                return await get_weather(city)

            if tool_name == "namaz_vakti":
                return await get_prayer_times(tool_param or user_input)

            if tool_name == "risale_ara":
                result = self.faiss_kb.get_relevant_context(
                    tool_param or user_input, max_chunks=6
                )
                return result or None

            if tool_name == "web_ara" or tool_name == "wiki_ara":
                query = tool_param or user_input
                result = await web_ara(query)
                return result

            return None
        except Exception as e:
            print(f"❌ Araç hatası ({tool_name}): {e}")
            return None


    def _hafizada_ara(self, user_input: str, chat_history_length: int) -> str:
        """Hafızada semantik arama (gerekiyorsa)

        NOT: Telegram session timeout olsa bile HafizaAsistani'nın
        kendi hafızası (self.hafiza) varsa arama yapılmalı!
        """
        # Hem Telegram history hem de kendi hafızamız boşsa atla
        if chat_history_length < 1 and len(self.hafiza) < 1:
            return ""
        return self.search(user_input)

    def _intelligent_decision(self, user_input: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """
        🧠 AKILLI KARAR SİSTEMİ - KEYWORD YOK! TEK LLM HER ŞEYİ KARAR VERİYOR
        LLM soruyu analiz edip hem kaynakları hem de tool'u belirliyor

        Returns:
            {
                "question_type": "greeting|farewell|religious|technical|general|followup|math|weather|prayer|topic_closed",
                "needs_faiss": bool,
                "needs_semantic_memory": bool,
                "needs_chat_history": bool,
                "tool_name": "yok|hesapla|zaman_getir|hava_durumu|namaz_vakti|risale_ara|web_ara",
                "tool_param": str,
                "response_style": "brief|detailed|conversational",
                "is_farewell": bool,
                "topic_closed": bool,  # YENİ: Kullanıcı bu konuyu kapatmak istiyor mu?
                "closed_topic_summary": str,  # YENİ: Kapanan konunun özeti
                "reasoning": str
            }
        """
        try:
            history_context = ""
            history_parts = []

            # 1. Telegram chat_history'den al (öncelikli)
            if chat_history:
                recent = chat_history[-self.max_mesaj:]  # Tutarlı history
                for m in recent:
                    is_user = m.get("role") == "user"
                    role = "KULLANICI" if is_user else "AI"
                    content = m.get("content") or ""
                    if content:
                        history_parts.append(f"{role}: {content}")

            # 2. Telegram history boşsa, HafizaAsistani'nın kendi hafızasından al
            # (Session timeout durumunda kalıcı hafızayı kullan)
            if not history_parts and self.hafiza:
                recent_hafiza = self.hafiza[-self.max_mesaj:]  # Tutarlı history
                for m in recent_hafiza:
                    rol = m.get("rol", "user")
                    role = "KULLANICI" if rol == "user" else "AI"
                    mesaj = m.get("mesaj", "")
                    if mesaj:
                        history_parts.append(f"{role}: {mesaj}")
                if history_parts:
                    print("   📦 Telegram history boş, HafizaAsistani hafızası kullanılıyor")

            if history_parts:
                history_context = "\n".join(history_parts)

            history_section = f"GEÇMİŞ:\n{history_context}\n" if history_context else ""
            decision_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Karar sistemi. ÖNCE <analiz> YAZ, SONRA JSON VER.

{history_section}MESAJ: {user_input}

<analiz>
1. TİP: Sohbet/bilgi/teknik/dini/matematik/duygusal?
2. GÜVENİM: %90+ biliyor muyum?
3. KAYNAK: Kendi bilgim mi, tool mu lazım?
</analiz>

KURALLAR:
• Selam/merhaba/veda/evet/tamam/anladım → friend, tool_name="yok"
• Dini (Allah/iman/namaz/Kuran) → religious_teacher, risale_ara
• Matematik → hesapla | Saat → zaman_getir | Hava → hava_durumu
• Belirsiz → needs_clarification=true
• Bilmediğin konu, güncel haber, kişi/yer/olay → web_ara
• "araştır/güncel bilgi/son haberler/internetten bak" → web_ara (MUTLAKA!)
• Kullanıcı "eski bilgi" derse → web_ara ile TEKRAR ara
• ⚠️ GÜNCEL KONU KURALI: Haber/güncel olay konusunda HER SORU için web_ara kullan. "Zaten aradık" deme, her seferinde güncel bilgi getir!

JSON:
{{"question_type": "greeting|farewell|followup|religious|math|weather|general|ambiguous",
"needs_faiss": bool, "needs_semantic_memory": bool, "needs_chat_history": bool,
"needs_clarification": bool, "tool_name": "yok|hesapla|zaman_getir|hava_durumu|namaz_vakti|risale_ara|web_ara",
"tool_param": "", "is_farewell": bool, "topic_closed": bool, "confidence": "low|medium|high", "reasoning": ""}}

ÖRNEKLER:
1) "Selam/Merhaba/Nasılsın" → {{"question_type":"greeting","tool_name":"yok","confidence":"high"}}
2) "Hava durumu nasıl" → {{"question_type":"weather","tool_name":"hava_durumu","confidence":"high"}}
3) "Bugün hava nasıl" → {{"question_type":"weather","tool_name":"hava_durumu","confidence":"high"}}
4) "Hava kaç derece" → {{"question_type":"weather","tool_name":"hava_durumu","confidence":"high"}}
5) "Allah'ın kudreti" → {{"question_type":"religious","tool_name":"risale_ara","needs_faiss":true}}
6) "evet ilginçmiş" → {{"question_type":"followup","tool_name":"yok"}}
7) "Python nedir" → {{"question_type":"general","tool_name":"web_ara","tool_param":"Python programlama dili"}}
8) "iyi araştır/güncel bilgi istiyorum" → {{"question_type":"general","tool_name":"web_ara","tool_param":"[önceki konuyla ilgili arama]"}}
9) "bu eski bilgi, son haberlere bak" → {{"question_type":"general","tool_name":"web_ara","tool_param":"[konu] son haberler 2026"}}

ÖNCE <analiz>, SONRA JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

            print("\n🧠 LLM'e akıllı karar soruluyor (Together.ai - tek LLM)...")

            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.decision_model,
                    "prompt": decision_prompt,
                    "max_tokens": 700,
                    "temperature": 0.1,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                },
                timeout=30,
            )

            if response.status_code != 200:
                print("⚠️ API hatası, fallback karar")
                return self._fallback_decision()

            llm_response = response.json()["choices"][0]["text"].strip()

            analiz_match = re.search(r'<analiz>(.*?)</analiz>', llm_response, re.DOTALL)
            if analiz_match:
                analiz_text = analiz_match.group(1).strip()
                print(f"\n💭 LLM Düşünce Süreci:")
                for line in analiz_text.split('\n'):
                    line = line.strip()
                    if line:
                        print(f"   {line}")

            json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response, re.DOTALL)
                json_str = json_match.group() if json_match else None

            if json_str:
                decision = json.loads(json_str)

                defaults = {
                    "question_type": "general",
                    "needs_faiss": False,
                    "needs_semantic_memory": False,
                    "needs_chat_history": False,
                    "needs_clarification": False,
                    "tool_name": "yok",
                    "tool_param": "",
                    "response_style": "conversational",
                    "is_farewell": False,
                    "topic_closed": False,
                    "closed_topic_summary": "",
                    "confidence": "medium",
                    "reasoning": ""
                }

                for key, default_val in defaults.items():
                    if key not in decision:
                        decision[key] = default_val

                if decision.get("question_type") or decision.get("tool_name"):
                    if decision.get('question_type') == 'farewell':
                        decision["is_farewell"] = True

                    should_close = (
                        decision.get('question_type') in ['farewell', 'topic_closed'] or
                        decision.get('is_farewell', False)
                    )
                    if should_close:
                        decision["topic_closed"] = True


                    if decision.get('question_type') == 'religious':
                        decision['tool_name'] = 'risale_ara'
                        decision['needs_faiss'] = True  # FAISS her zaman açık
                        decision['is_religious'] = True  # Dini konu flag'i

                        is_detail_followup, followup_confidence, matched_concepts = self._detect_detail_followup(
                            user_input, chat_history
                        )
                        if is_detail_followup:
                            decision['is_detail_followup'] = True
                            decision['followup_confidence'] = followup_confidence
                            decision['matched_concepts'] = matched_concepts
                            print(f"   🔄 TAKİP MODU AKTİF: FAISS arka plan olarak kullanılacak")
                        else:
                            decision['is_detail_followup'] = False

                    if decision.get('question_type') == 'ambiguous' or decision.get('needs_clarification'):
                        decision['tool_name'] = 'yok'
                        decision['needs_clarification'] = True

                    word_count = len(user_input.split())
                    if word_count <= 4 and not decision.get('needs_chat_history'):
                        decision['needs_chat_history'] = True
                        print(f"   📌 Kısa mesaj ({word_count} kelime) → chat_history zorunlu yapıldı")

                    confidence_emoji = {"low": "🔴", "medium": "🟡", "high": "🟢"}.get(decision['confidence'], "🟡")

                    print(f"\n✅ LLM Kararı:")
                    print(f"   • Tür: {decision['question_type']}")
                    print(f"   • Güven: {confidence_emoji} {decision['confidence']}")
                    print(f"   • FAISS: {'✅' if decision['needs_faiss'] else '❌'}")
                    print(f"   • Semantic: {'✅' if decision['needs_semantic_memory'] else '❌'}")
                    print(f"   • History: {'✅' if decision['needs_chat_history'] else '❌'}")
                    print(f"   • Tool: {decision['tool_name']}")
                    if decision['tool_param']:
                        print(f"   • Tool Param: {decision['tool_param']}")
                    print(f"   • Stil: {decision['response_style']}")
                    if decision.get('needs_clarification'):
                        print(f"   • ❓ Netleştirme gerekiyor!")
                    if decision.get('is_farewell'):
                        print(f"   • 👋 Vedalaşma algılandı!")
                    if decision.get('topic_closed'):
                        print(f"   • 📕 KONU KAPANDI: {decision.get('closed_topic_summary', 'özet yok')}")
                    if "reasoning" in decision:
                        print(f"   • Sebep: {decision['reasoning']}")

                    return decision

            print("⚠️ JSON parse hatası, fallback karar")
            print(f"   📝 Ham LLM yanıtı (son 500 karakter):")
            print(f"   {llm_response[-500:] if len(llm_response) > 500 else llm_response}")
            return self._fallback_decision()

        except Exception as e:
            print(f"⚠️ LLM karar hatası: {e}, fallback karar")
            return self._fallback_decision()

    def _fallback_decision(self) -> Dict[str, Any]:
        """Hata durumunda güvenli fallback kararı - tüm bağlamı kullanır"""
        return {
            "question_type": "general",
            "needs_faiss": False,
            "needs_semantic_memory": True,  # Güvenli mod: hafıza aç
            "needs_chat_history": True,     # Güvenli mod: history aç
            "tool_name": "yok",
            "tool_param": "",
            "response_style": "conversational",
            "is_farewell": False,
            "topic_closed": False,
            "closed_topic_summary": "",
            "confidence": "medium",
            "reasoning": "Fallback: Güvenli mod, tüm bağlamı kullan"
        }

    def _faiss_ara(self, user_input: str) -> str:
        """FAISS KB'de ara (dini sorularda)"""
        print("🔍 FAISS araması yapılıyor...")
        return self.faiss_kb.get_relevant_context(user_input, max_chunks=6)

    def _history_summary(self, chat_history: List[Dict], current_question_type: str = None, max_len: int = 6000) -> str:
        """
        Chat history'den mesajları al

        YENİ TASARIM (Basit ve Net):
        - Son 12 mesaj (6 user + 6 AI) HER ZAMAN prompt'a gider
        - 12'den eskiler prompt'a GİTMEZ (tampon bölgede kalır)
        - Konu değişince tampon bölge özetlenip TopicMemory'ye gider
        - Eski konuya dönüldüğünde TopicMemory'den çekilir
        """
        if not chat_history:
            return ""

        son_mesajlar = chat_history[-self.max_mesaj:] if len(chat_history) >= self.max_mesaj else chat_history

        if len(son_mesajlar) == 0:
            return ""

        tmp = []
        for m in son_mesajlar:
            is_user = m.get("role") == "user"
            role = "KULLANICI" if is_user else "AI"
            text = m.get("content") or ""
            if text:
                tmp.append(f"[{role}]: {text}")

        return "\n".join(tmp)


    # TEK BİRLEŞİK PROMPT - Full Friend Modu
    SYSTEM_PROMPT = """Sen kullanıcının olgun ve sıcakkanlı bir yapay zeka arkadaşısın.

- Doğal uzunlukta cevap ver, gereksiz uzatma
- Samimi ama abartısız ol
- Emoji kullanabilirsin (abartmadan)
- Kısa tepkilere (evet, tamam, anladım) kısa cevap ver
- Kısa mesajları bağlama göre yorumla ("umarım inş", "aynen" gibi)

🎯 SORU ANLAMA KURALLARI:
- Görüş sorusu (mı?, sence?, avantaj olur mu?) → önce NET CEVAP (evet/hayır/bence...), sonra kısa açıklama
- Kullanıcı teknik detay sormadıysa → teknik detay, zorluklar, sistemsel ihtiyaçlar ANLATMA
- Senaryo/fikir tartışması istiyorsa → onunla birlikte düşün, ders verme
- SADECE sorulan şeye cevap ver, istenmeyen bilgi ekleme

⚠️ ÖNEMLİ: Aşağıda [💬 Önceki Konuşma] bölümü varsa, bu DEVAM EDEN bir sohbettir - direkt cevaba geç, gereksiz giriş cümlesi veya bağlam tekrarı yapma!

🔴 DİNİ KONULARDA (verilen metin varsa):
- Cevabı VERİLEN METİNDEN oluştur
- "Risale'de", "metinde" DEME - gizli kaynak olarak kullan
- Temsilleri KENDİ sözünmüş gibi anlat
- Vaaz değil sohbet tonu"""

    # Geriye uyumluluk için (eski kod hala role parametresi kullanıyorsa)
    ROLE_SYSTEM_PROMPTS = {
        "friend": SYSTEM_PROMPT,
        "religious_teacher": SYSTEM_PROMPT
    }

    def _extract_used_concepts(self, previous_response: str) -> List[str]:
        """Önceki cevapta kullanılan temsil ve kavramları çıkar"""
        if not previous_response:
            return []

        temsiller = [
            "güneş", "ayna", "damla", "deniz", "zerre", "şems",
            "ışık", "nur", "feyz", "tecelli", "yansıma", "akis",
            "ressam", "tablo", "nakkaş", "san'at", "kitap", "harf",
            "sultan", "padişah", "ordu", "asker", "fabrika", "makine"
        ]

        kavramlar = [
            "şeffafiyet", "mukabele", "müvazene", "intizam",
            "melekûtiyet", "mülk", "taalluk", "vahdet", "kesret",
            "tecezzî", "tenakus", "tekebbür", "temsil", "tefekkür",
            "kayyumiyet", "rububiyet", "uluhiyet", "vahdaniyet"
        ]

        used = []
        lower_response = previous_response.lower()

        for t in temsiller + kavramlar:
            if t in lower_response:
                used.append(t)

        return used

    def _detect_detail_followup(self, user_input: str, chat_history: List[Dict[str, Any]]) -> Tuple[bool, float, List[str]]:
        """
        İki katmanlı takip sorusu tespiti

        KATMAN 1 (ÖNCELİKLİ): Kavram eşleşmesi
        - Kullanıcının sorusundaki anahtar kelimeler önceki cevabında geçiyor mu?

        KATMAN 2: Soru kalıpları
        - "bu ne demek?", "nasıl yani?", "örnek verir misin?" gibi kalıplar

        Returns:
            (is_followup, confidence_score, matched_concepts)
        """
        if not chat_history:
            return False, 0.0, []

        user_lower = user_input.lower()

        last_ai_response = ""
        for msg in reversed(chat_history):
            if msg.get('role') == 'assistant':
                last_ai_response = msg.get('content', '')
                break

        if not last_ai_response:
            return False, 0.0, []

        used_concepts = self._extract_used_concepts(last_ai_response)
        matched_concepts = []

        for concept in used_concepts:
            concept_variants = [concept]
            if 'b' in concept:
                concept_variants.append(concept.replace('b', 'p'))
            if 'p' in concept:
                concept_variants.append(concept.replace('p', 'b'))

            for variant in concept_variants:
                if variant in user_lower:
                    matched_concepts.append(concept)
                    break

        followup_patterns = [
            "bu ne demek", "nasıl oluyor", "neden böyle",
            "örnek verir misin", "örnek ver", "anlamadım",
            "açıkla", "açıklar mısın", "tam olarak", "nasıl yani",
            "ne demek istedi", "ne demek bu", "yani nasıl",
            "biraz daha", "detay ver", "mesela", "peki nasıl",
            "nedir bu", "ne anlama", "açar mısın"
        ]
        pattern_match = any(p in user_lower for p in followup_patterns)


        if matched_concepts and pattern_match:
            confidence = 0.95
            is_followup = True
            print(f"   🎯 TAKİP TESPİT: Kavram + Kalıp eşleşti (güven: %{int(confidence*100)})")
            print(f"      Eşleşen kavramlar: {matched_concepts}")

        elif len(matched_concepts) >= 2:
            confidence = 0.85
            is_followup = True
            print(f"   🎯 TAKİP TESPİT: 2+ kavram eşleşti (güven: %{int(confidence*100)})")
            print(f"      Eşleşen kavramlar: {matched_concepts}")

        elif matched_concepts:
            confidence = 0.70
            is_followup = True
            print(f"   🎯 TAKİP TESPİT: 1 kavram eşleşti (güven: %{int(confidence*100)})")
            print(f"      Eşleşen kavram: {matched_concepts}")

        elif pattern_match and len(chat_history) >= 2:
            confidence = 0.55
            is_followup = True
            print(f"   🎯 TAKİP TESPİT: Soru kalıbı (güven: %{int(confidence*100)})")

        else:
            confidence = 0.0
            is_followup = False

        return is_followup, confidence, matched_concepts

    def _add_exclusion_to_prompt(self, role_prompt: str, used_concepts: List[str]) -> str:
        """Kullanılmış kavramları prompt'a yasak olarak ekle"""
        if not used_concepts:
            return role_prompt

        exclusion_text = f"""
🚫 BU KAVRAMLARI TEKRAR KULLANMA (önceki cevapta kullanıldı):
{', '.join(used_concepts)}

Bunların yerine VERİLEN METİNDEKİ DİĞER kavram ve temsilleri kullan veya FARKLI açıdan anlat.
"""
        if "❌ YAPMA:" in role_prompt:
            return role_prompt.replace("❌ YAPMA:", f"{exclusion_text}\n❌ YAPMA:")
        else:
            return role_prompt + exclusion_text

    def _prompt_olustur(
        self,
        user_input: str,
        tool_result: Optional[str],
        semantic_context: str,
        faiss_context: str,
        chat_history: str,
        role: str,
        closed_topics_warning: str = "",  # Kapanmış konu uyarısı
        silent_long_term_context: str = "",  # 🆕 Sessiz uzun dönem bağlamı
        needs_clarification: bool = False,  # 🆕 Netleştirme gerekli mi?
        llm_reasoning: str = "",  # 🧠 DecisionLLM'in ön araştırması
        is_topic_closed: bool = False,  # 🆕 Konu kapandı mı? (kısa cevap ver)
        is_detail_followup: bool = False,  # 🆕 Takip sorusu mu? (FAISS arka plan olarak)
        tool_name: str = "yok",  # 🆕 Kullanılan araç (web_ara için özel mod)
    ) -> str:
        """Final prompt'u oluştur (rol'e göre)"""

        zaman = get_current_datetime()
        zaman_satiri = f"[⏰ ZAMAN BİLİNCİ]: {zaman['full']} ({zaman['zaman_dilimi']})"

        # Tek birleşik prompt kullan
        role_prompt = self.SYSTEM_PROMPT

        # Dini konularda tekrar yasağı kontrolü
        is_religious = role in ["religious_teacher", "religious"] or "risale_ara" in str(tool_name)
        if is_religious and chat_history and not is_detail_followup:
            used_concepts = self._extract_used_concepts(chat_history)
            if used_concepts:
                role_prompt = self._add_exclusion_to_prompt(role_prompt, used_concepts)
                print(f"🚫 Tekrar yasağına eklenen kavramlar: {', '.join(used_concepts)}")
        elif is_detail_followup:
            print(f"   ⏩ Tekrar yasağı atlandı (takip modu - kullanıcı kavramı soruyor)")

        combined_sources = []


        if closed_topics_warning:
            combined_sources.append(f"[⚠️ KAPANMIŞ KONULAR - TEKRAR AÇMA!]:\n{closed_topics_warning}")

        if tool_result:
            if tool_name == "web_ara" or tool_name == "wiki_ara":
                combined_sources.append(f"[🌐 İNTERNET ARAŞTIRMASI]:\n{tool_result}\n\n⚠️ Bu bilgi soruyla alakalı mı? Alakasız veya yanlış ise HİÇ KULLANMA, kendi bilginle cevap ver.")
            elif tool_name == "risale_ara":
                if is_detail_followup:
                    combined_sources.append(f"[🔇 ARKA PLAN BİLGİSİ - Doğrudan verme, kendi yorumunla açıkla!]:\n{tool_result}")
                else:
                    combined_sources.append(f"[📚 RİSALE-İ NUR'DAN - BU BİLGİYİ KULLAN!]:\n{tool_result}")
            else:
                combined_sources.append(f"[🔧 ARAÇ SONUCU]:\n{tool_result}")

        if chat_history:
            combined_sources.append(f"[💬 Önceki Konuşma (DEVAM EDEN SOHBET - tekrar selamlama YAPMA!)]:\n{chat_history}")

        if semantic_context:
            combined_sources.append(f"[HAFIZA]:\n{semantic_context}")

        if faiss_context and not tool_result:
            combined_sources.append(f"[BİLGİ TABANI]:\n{faiss_context}")

        if silent_long_term_context:
            combined_sources.append(f"[🔇 ARKA PLAN BİLGİSİ - KULLANICIYA SÖYLEME]:\n{silent_long_term_context}")

        # Kullanıcı profili ekle (varsa)
        if hasattr(self, 'profile_manager'):
            profile_context = self.profile_manager.get_prompt_context()
            if profile_context:
                combined_sources.insert(0, f"[👤 KULLANICI PROFİLİ - doğal kullan, ezberletme]:\n{profile_context}")

        if not combined_sources:
            sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{zaman_satiri}

[🎭 ROL]: {role.upper()}
{role_prompt}

{sep}
📋 KURALLAR:
{sep}
1. ❌ Soruyu tekrarlama, liste yapma (*, -, 1. 2. 3.)
2. ✅ Kendi bilgin gibi özgüvenle sun
3. ✅ Samimi Türkçe konuş
4. ✅ Rolüne uygun davran

{sep}
📩 YENİ MESAJ:
{sep}
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        combined_str = "\n\n".join(combined_sources)

        # Tek tutarlı yapılandırma
        max_length = 2000  # Sabit maksimum uzunluk

        dynamic_rules = []

        if closed_topics_warning:
            dynamic_rules.append(f"⚠️ KAPANMIŞ KONU: \"{closed_topics_warning}\" konusu kapandı, tekrar AÇMA!")

        if silent_long_term_context:
            dynamic_rules.append("🔇 Arka plan bilgisini sessizce kullan, zorla hatırlatma yapma")

        if tool_result:
            if tool_name == "web_ara" or tool_name == "wiki_ara":
                dynamic_rules.append("🌐 İnternet bilgisi geldi - alakalıysa kullan, alakasız veya yanlış ise HİÇ KULLANMA!")
            else:
                dynamic_rules.append("🔍 ARAÇ SONUCU verildi - bu bilgiyi MUTLAKA kullan, kendi tahminini yapma!")

        if needs_clarification:
            dynamic_rules.append("❓ BELİRSİZ SORU - önce netleştirici soru sor, tahmin etme!")

        if is_topic_closed:
            dynamic_rules.append("📕 KONU KAPANDI - sadece 1-2 cümle ile kapat")

        dynamic_rules_str = ""
        if dynamic_rules:
            dynamic_rules_str = "\n" + "\n".join([f"• {r}" for r in dynamic_rules])

        if (tool_name == "web_ara" or tool_name == "wiki_ara") and tool_result:
            context_header = "Bağlam (İNTERNET BİLGİSİ - alakalıysa kullan, değilse kullanma!):"
        elif is_detail_followup and tool_result:
            context_header = "Bağlam (Arka plan - kendi yorumunla açıkla):"
        elif tool_result:
            context_header = "Bağlam (ARAÇ SONUCUNU MUTLAKA KULLAN!):"
        else:
            context_header = "Bağlam (Kullan, ama sadece GERÇEKTEN alakalıysa):"

        # Dini konularda mı belirleme
        is_religious_topic = is_religious or tool_name == "risale_ara"

        if is_religious_topic:
            if is_detail_followup:
                rules_text = """KURALLAR (TAKİP SORUSU - AÇIKLAMA MODU):
1. 🔇 ARKA PLAN bilgisini DOĞRUDAN VERME, referans olarak kullan
2. ✅ KENDİ YORUMUNLA ve ÖRNEKLERLE açıkla
3. ✅ Önceki cevabından devam et, bağlamı koru
4. ✅ Günlük hayattan somut örnekler ver
5. ✅ Samimi Türkçe konuş
6. ❌ Metni kopyala-yapıştır YAPMA, sindirerek anlat
7. 🎭 Bir arkadaşına anlatır gibi açıkla"""
            else:
                rules_text = """KURALLAR:
1. ⚠️ Yanlış bilgiyi onaylama, nazikçe düzelt
2. ❌ Soruyu tekrarlama, liste yapma (*, -, 1. 2. 3.)
3. ✅ VERİLEN METİNDEN anlat - metindeki kavramları MUTLAKA kullan
4. ✅ Samimi Türkçe konuş
5. 🔄 Kendini tekrar etme, sohbeti ilerlet"""
        else:
            rules_text = """KURALLAR:
1. ⚠️ Yanlış bilgiyi onaylama, nazikçe düzelt
2. ❌ Soruyu tekrarlama, liste yapma (*, -, 1. 2. 3.)
3. ❌ KAYNAK BELİRTME YASAK: "Kaynaklara göre" gibi ifadeler KULLANMA
4. ✅ Kendi bilgin gibi özgüvenle sun
5. ✅ Samimi Türkçe konuş
6. 🔄 Aynı şeyleri döngüye sokma, her cevap taze olsun"""

        sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{zaman_satiri}

{role_prompt}

{sep}
📚 BAĞLAM (gerekirse kullan):
{sep}
{combined_str}

{sep}
📋 KURALLAR:
{sep}
{rules_text}{dynamic_rules_str}

{sep}
📩 YENİ MESAJ:
{sep}
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return prompt


    async def hazirla_ve_prompt_olustur(
        self, user_input: str, chat_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        SEKRETER'İN ANA GÖREVİ - TEK LLM KARAR SİSTEMİ!

        1. LLM'e karar verdirme (keyword yok! hem kaynak hem tool)
        2. LLM'in seçtiği tool'u çalıştır
        3. Bağlamı topla (LLM'in kararına göre!)
        4. Prompt'u hazırla
        5. Gemma3 için hazır paketi döndür
        """
        print("\n" + "=" * 60)
        print("📋 SEKRETER ÇALIŞIYOR (TEK LLM SİSTEMİ)")
        print("=" * 60)
        print(f"📝 Kullanıcı: {user_input}")

        is_closed, closed_summary = self.is_topic_closed(user_input)
        if is_closed:
            print(f"⚠️ UYARI: Bu soru kapanmış bir konuya benziyor: '{closed_summary}'")
            print("   AI'a bu konuyu tekrar açmaması söylenecek.")

        print("\n🧠 1. LLM tek karar veriyor (hem kaynak hem tool)...")
        decision = self._intelligent_decision(user_input, chat_history)

        if decision.get('topic_closed', False):
            topic_summary = decision.get('closed_topic_summary', '')

            if not topic_summary:
                if chat_history:
                    for msg in reversed(chat_history):
                        if msg.get('role') == 'assistant':
                            content = (msg.get('content') or '')[:100]
                            if content and len(content) > 5:
                                topic_summary = content
                                break

                if not topic_summary and chat_history:
                    for msg in reversed(chat_history):
                        if msg.get('role') == 'user':
                            content = (msg.get('content') or '').strip()
                            if content and len(content) > 10 and not any(
                                w in content.lower() for w in ['teşekkür', 'sağol', 'eyvallah', 'görüşürüz', 'bye', 'hoşça']
                            ):
                                topic_summary = content[:100]
                                break

                if not topic_summary and decision.get('reasoning'):
                    topic_summary = decision['reasoning'][:100]

            if topic_summary:
                print(f"💾 Konu kaydediliyor: '{topic_summary[:50]}...'")
                self.add_closed_topic(topic_summary, chat_history)
            else:
                print("⚠️ topic_closed=true ama özet çıkarılamadı, kayıt atlandı")

        tool_name = decision.get('tool_name', 'yok')
        tool_param = decision.get('tool_param', '')

        print(f"\n🛠️ 2. Araç çalıştırılıyor (LLM kararı: {tool_name})...")
        tool_result = await self._tool_calistir(tool_name, tool_param, user_input)

        print("\n📚 3. Bağlam toplanıyor (LLM kararına göre)...")

        if decision['needs_semantic_memory']:
            semantic_context = self._hafizada_ara(user_input, len(chat_history))
            print(f"   • Semantic Hafıza: {'✅ bulundu' if semantic_context else '❌ bulunamadı'} (LLM kararı)")
        else:
            semantic_context = ""
            print("   • Semantic Hafıza: ⏩ atlandı (LLM: gereksiz)")

        if decision['needs_faiss'] and tool_name != "risale_ara":
            faiss_context = self._faiss_ara(user_input)
            print(f"   • FAISS KB: {'✅ bulundu' if faiss_context else '❌ bulunamadı'} (LLM kararı)")
        elif tool_name == "risale_ara":
            faiss_context = ""  # Tool zaten FAISS kullandı, duplicate arama yapma
            print("   • FAISS KB: ⏩ atlandı (risale_ara tool'u zaten FAISS kullandı)")
        else:
            faiss_context = ""
            print("   • FAISS KB: ⏩ atlandı (LLM: gereksiz)")


        if self.conversation_context:
            topic_changed = self.conversation_context.check_topic_before_response(
                user_input, chat_history
            )
            if topic_changed:
                print(f"   • 🔄 Yeni konu algılandı - eski bağlam temizlendi")

        conversation_context = self.get_conversation_context()
        if conversation_context:
            print(f"   • 🧠 ConversationContext: ✅ LLM özeti enjekte edildi")
        else:
            print(f"   • 🧠 ConversationContext: ⏩ henüz özet yok")

        silent_long_term_context = ""
        if self.should_check_long_term_memory(user_input):
            silent_long_term_context = self.get_silent_long_term_context(user_input)
            if silent_long_term_context:
                print(f"   • 🔇 TopicMemory: ✅ sessiz bağlam enjekte edildi")
            else:
                print(f"   • 🔇 TopicMemory: ❌ eşleşme yok")
        else:
            print("   • 🔇 TopicMemory: ⏩ atlandı (geçmiş referansı yok)")

        combined_silent_context = ""
        if conversation_context:
            combined_silent_context = conversation_context
        if silent_long_term_context:
            if combined_silent_context:
                combined_silent_context += "\n\n" + silent_long_term_context
            else:
                combined_silent_context = silent_long_term_context

        question_type = decision['question_type']

        # SABİT HISTORY - self.max_mesaj ile tutarlı (ton değişikliği önlenir)
        max_history_msgs = self.max_mesaj  # 20

        # Telegram history varsa onu kullan
        if chat_history and len(chat_history) > 0:
            limited_history = chat_history[-max_history_msgs:] if len(chat_history) > max_history_msgs else chat_history

            chat_history_summary = self._history_summary(
                limited_history,
                current_question_type=question_type
            )
            print(f"   • Chat History: ✅ son {len(limited_history)} mesaj dahil edildi ({len(limited_history)}/{len(chat_history)} toplam)")

        # Telegram history boşsa ama self.hafiza doluysa, oradan özet oluştur
        elif self.hafiza and len(self.hafiza) > 0:
            # self.hafiza formatını chat_history formatına çevir
            hafiza_as_history = []
            for m in self.hafiza[-max_history_msgs:]:
                hafiza_as_history.append({
                    "role": m.get("rol", "user"),
                    "content": m.get("mesaj", "")
                })

            chat_history_summary = self._history_summary(
                hafiza_as_history,
                current_question_type=question_type
            )
            print(f"   • Chat History: ✅ HafizaAsistani'dan {len(hafiza_as_history)} mesaj kullanıldı (Telegram session timeout)")
        else:
            chat_history_summary = ""
            print("   • Chat History: ⏩ henüz yok")

        print("\n🎭 4. Tek kişilik kullanılıyor...")
        # Artık ayrı roller yok, tek tutarlı kişilik
        role = "default"
        print(f"   • Mod: unified (tek kişilik)")

        closed_topics_warning = ""
        if is_closed and closed_summary:
            user_wants_reopen = self._user_wants_to_reopen_topic(user_input)

            if user_wants_reopen:
                print(f"   • Kapanmış Konu: '{closed_summary}' - Kullanıcı tekrar açmak istiyor ✅")
            else:
                closed_topics_warning = closed_summary
                print(f"   • Kapanmış Konu Uyarısı: '{closed_summary}' - AI'a bildirildi")
        else:
            print("   • Kapanmış Konu: Yok veya ilgisiz ⏩")

        print("\n📝 6. Prompt hazırlanıyor...")
        needs_clarification = decision.get('needs_clarification', False)
        llm_reasoning = decision.get('reasoning', '')  # 🧠 DecisionLLM'in ön araştırması
        is_topic_closed = decision.get('topic_closed', False)  # 📕 Konu kapandı mı?
        is_detail_followup = decision.get('is_detail_followup', False)  # 🆕 Takip sorusu mu?

        if is_detail_followup:
            print(f"   • 🔄 TAKİP MODU: FAISS arka plan olarak kullanılacak")
            print(f"   • 📊 Güven: %{int(decision.get('followup_confidence', 0) * 100)}")

        final_prompt = self._prompt_olustur(
            user_input,
            tool_result,
            semantic_context,
            faiss_context,
            chat_history_summary,
            role,
            closed_topics_warning,  # Sadece gerektiğinde dolu
            combined_silent_context,  # 🧠🔇 Birleşik bağlam (ConversationContext + TopicMemory)
            needs_clarification,  # 🆕 Netleştirme gerekli mi?
            llm_reasoning,  # 🧠 DecisionLLM'in ön araştırması - KOPUKLUK DÜZELTMESİ!
            is_topic_closed,  # 📕 Konu kapandı mı? (kısa cevap ver)
            is_detail_followup,  # 🆕 Takip sorusu mu? (FAISS arka plan olarak)
            tool_name,  # 🌐 Kullanılan araç (web_ara için özel mod)
        )
        print(f"   • Prompt uzunluğu: {len(final_prompt)} karakter")

        paket = {
            "prompt": final_prompt,
            "role": role,
            "tool_used": tool_name,
            "llm_decision": decision,
            "metadata": {
                "has_tool_result": tool_result is not None,
                "has_semantic": bool(semantic_context),
                "has_faiss": bool(faiss_context),
                "has_history": bool(chat_history_summary),
                "has_context_memory": bool(combined_silent_context),  # 🧠🔇 Birleşik bağlam
                "closed_topic_filtered": is_closed,  # Kapanmış konu filtresi uygulandı mı
                "needs_clarification": needs_clarification,  # 🆕 Netleştirme gerekli mi?
                "is_detail_followup": is_detail_followup,  # 🆕 Takip sorusu mu?
            },
        }

        print("\n✅ SEKRETER HAZIR - Tek LLM kararıyla paket oluşturuldu!")
        print("=" * 60 + "\n")

        return paket


    def set_faiss_kb(self, faiss_kb):
        """FAISS KB'yi inject et"""
        self.faiss_kb.set_faiss_kb(faiss_kb)
        print(f"✅ FAISS KB inject edildi (aktif: {self.faiss_kb.enabled})")

    @property
    def data(self):
        """Geriye uyumluluk: hafiza.data"""
        return self.hafiza

    @property
    def reranker(self):
        """Geriye uyumluluk: reranker var mı? (şu an yok)"""
        return None

    def should_search_memory(self, chat_history_length: int) -> bool:
        """
        Geriye uyumluluk: Hafıza araması yapılmalı mı?
        Eski PersonalAI bu metodu kullanıyor
        """
        if not self.hafiza or len(self.hafiza) == 0:
            return False
        if len(self.hafiza) < 3:
            return False
        if chat_history_length == 0 and len(self.hafiza) > 0:
            return True
        return True

    def search_with_rerank(
        self, query: str, top_k: Optional[int] = None, initial_k: int = 50
    ) -> str:
        """
        Geriye uyumluluk: Reranker ile arama
        (Şu an normal search'e yönlendiriliyor)
        """
        return self.search(query, top_k)

    def ilgili_mesajlari_bul(
        self, yeni_mesaj: str, max_mesaj: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Geriye uyumluluk: İlgili mesajları bul (eski API)
        NOT: Artık _search_internal() kullanıyor (çift işlem kaldırıldı)
        Returns: [{"rol": "user", "mesaj": "..."}, ...]
        """
        if not self.hafiza or not yeni_mesaj:
            return []

        k = max_mesaj or self.max_mesaj
        return self._search_internal(yeni_mesaj, k)

    def son_mesajlari_al(self, n: int = 3) -> List[Dict[str, str]]:
        """
        Geriye uyumluluk: Son n mesajı döndür
        """
        if len(self.hafiza) < n:
            n = len(self.hafiza)

        son_mesajlar = self.hafiza[-n:]
        return [{"rol": m["rol"], "mesaj": m["mesaj"]} for m in son_mesajlar]

    def set_llm(self, llm):
        """LLM referansını ayarla - PersonalAI'dan çağrılır"""
        self.llm = llm
        print("✅ LLM HafizaAsistani'ya bağlandı")

    def _build_messages(
        self,
        user_input: str,
        paket: Dict[str, Any],
        chat_history: List[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        Messages formatı oluştur - LLM için proper chat format

        Returns: [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "current_input + context"}
        ]
        """
        messages = []

        # 1. System message - SYSTEM_PROMPT + zaman
        zaman = get_current_datetime()
        system_content = f"""{self.SYSTEM_PROMPT}

[⏰ ŞU AN]: {zaman['full']} ({zaman['zaman_dilimi']})"""

        messages.append({"role": "system", "content": system_content})

        # 2. Chat history - user/assistant rolleri ile
        max_history = self.max_mesaj  # 20

        # Telegram history varsa onu kullan
        if chat_history and len(chat_history) > 0:
            limited_history = chat_history[-max_history:] if len(chat_history) > max_history else chat_history
            for msg in limited_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if content and role in ['user', 'assistant']:
                    messages.append({"role": role, "content": content})

        # Telegram history boşsa self.hafiza'dan al
        elif self.hafiza and len(self.hafiza) > 0:
            for m in self.hafiza[-max_history:]:
                rol = m.get("rol", "user")
                mesaj = m.get("mesaj", "")
                if mesaj:
                    messages.append({"role": rol, "content": mesaj})

        # 3. Son user message - context ile birlikte
        context_parts = []

        # Metadata'dan context bilgilerini al
        metadata = paket.get('metadata', {})
        llm_decision = paket.get('llm_decision', {})

        # Prompt'tan context kısımlarını çıkar
        prompt = paket.get('prompt', '')

        # Tool result varsa ekle
        if metadata.get('has_tool_result'):
            tool_name = paket.get('tool_used', '')
            # Prompt'tan tool result'ı çıkarmaya çalış
            if '[🌐 İNTERNET ARAŞTIRMASI]:' in prompt:
                start = prompt.find('[🌐 İNTERNET ARAŞTIRMASI]:')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())
            elif '[📚 RİSALE-İ NUR\'DAN' in prompt:
                start = prompt.find('[📚 RİSALE-İ NUR\'DAN')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())

        # Semantic context varsa ekle
        if metadata.get('has_semantic'):
            if '[HAFIZA]:' in prompt:
                start = prompt.find('[HAFIZA]:')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())

        # FAISS context varsa ekle
        if metadata.get('has_faiss'):
            if '[BİLGİ TABANI]:' in prompt:
                start = prompt.find('[BİLGİ TABANI]:')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())

        # Kullanıcı profili varsa ekle
        if '[👤 KULLANICI PROFİLİ' in prompt:
            start = prompt.find('[👤 KULLANICI PROFİLİ')
            end = prompt.find('\n\n[', start + 1)
            if end == -1:
                end = prompt.find('━━━', start + 1)
            if start != -1 and end != -1:
                context_parts.append(prompt[start:end].strip())

        # Son user message'ı oluştur
        if context_parts:
            user_content = f"""📚 Bağlam:
{chr(10).join(context_parts)}

📩 Sorum:
{user_input}"""
        else:
            user_content = user_input

        messages.append({"role": "user", "content": user_content})

        return messages

    async def process(self, user_input: str, chat_history: List[Dict] = None, image_data=None) -> str:
        """
        Ana işlem metodu - Telegram'dan çağrılır

        Akış: Telegram → HafizaAsistani.process() → PersonalAI.generate() → Telegram

        1. Paket hazırla (context, tool results, vb.)
        2. Messages formatı oluştur
        3. LLM'e gönder (proper chat format)
        4. Cevabı kaydet
        5. Cevabı döndür
        """
        chat_history = chat_history or []

        # 1. Paket hazırla
        paket = await self.hazirla_ve_prompt_olustur(user_input, chat_history)

        # 2. LLM kontrolü
        if not hasattr(self, 'llm') or self.llm is None:
            return "❌ LLM bağlı değil!"

        # 3. LLM'e gönder
        if image_data:
            # Vision için eski prompt formatı kullan
            prompt = paket.get('prompt', user_input)
            response = await self.llm.generate(prompt, image_data)
        else:
            # Text için messages formatı kullan (YENİ!)
            messages = self._build_messages(user_input, paket, chat_history)
            response = await self.llm.generate(prompt=None, image_data=None, messages=messages)

        # 4. Cevabı kaydet
        self.add(user_input, response, chat_history)

        # 5. Döndür
        return response


async def test_sekreter():
    print("\n" + "=" * 60)
    print("🧪 HafizaAsistani v3.0 TEST")
    print("=" * 60)

    sekreter = HafizaAsistani(saat_limiti=48, esik=0.50, max_mesaj=20)

    print("\n--- TEST 1: Basit Sohbet ---")
    paket1 = await sekreter.hazirla_ve_prompt_olustur(
        user_input="Merhaba, nasılsın?",
        chat_history=[],
    )
    print(f"✅ Prompt hazır (uzunluk: {len(paket1['prompt'])})")
    print(f"   Role: {paket1['role']}")
    print(f"   Tool: {paket1['tool_used']}")

    print("\n--- TEST 2: Matematik ---")
    paket2 = await sekreter.hazirla_ve_prompt_olustur(
        user_input="15 çarpı 7 kaç eder?",
        chat_history=[],
    )
    print("✅ Prompt hazır")
    print(f"   Tool: {paket2['tool_used']}")
    print(f"   Tool sonucu: {paket2['metadata']['has_tool_result']}")

    print("\n--- TEST 3: Zaman ---")
    paket3 = await sekreter.hazirla_ve_prompt_olustur(
        user_input="Saat kaç?",
        chat_history=[],
    )
    print("✅ Prompt hazır")
    print(f"   Tool: {paket3['tool_used']}")

    print("\n" + "=" * 60)
    print("✅ Tüm testler tamamlandı (elle de deneyebilirsin).")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_sekreter())