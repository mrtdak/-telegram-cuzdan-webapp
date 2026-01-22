from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
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
from sohbet_zekasi import TurkishConversationIntelligence, BeklenenCevap, SohbetEnerjisi
# from calculation_context import CalculationContext  # Devre dışı - chat history yeterli


# ============================================================
# NOT YÖNETİCİSİ
# ============================================================

class NotManager:
    """
    Kullanıcının aldığı notları yöneten basit sistem.
    Her kullanıcının notları ayrı dosyada tutulur.

    Tetikleyiciler: "not al", "not tut", "not ekle"
    """

    def __init__(self, user_id: str = "default", base_dir: str = "user_data"):
        self.user_id = user_id
        self.notes_dir = os.path.join(base_dir, f"user_{user_id}", "notes")
        self.notes_file = os.path.join(self.notes_dir, "notlar.json")

        # Klasör yoksa oluştur
        os.makedirs(self.notes_dir, exist_ok=True)

        # Notları yükle
        self.notes = self._load_notes()

        # Onay bekleyen not
        self.pending_note = None

    def _load_notes(self) -> List[Dict]:
        """Notları dosyadan yükle"""
        if os.path.exists(self.notes_file):
            try:
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_notes(self):
        """Notları dosyaya kaydet"""
        with open(self.notes_file, 'w', encoding='utf-8') as f:
            json.dump(self.notes, f, ensure_ascii=False, indent=2)

    def not_al(self, icerik: str) -> str:
        """Yeni not kaydet (onay olmadan direkt kaydet)"""
        if not icerik or len(icerik.strip()) < 2:
            return "❌ Not içeriği boş olamaz."

        now = datetime.now()
        gun_isimleri = {
            0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe",
            4: "Cuma", 5: "Cumartesi", 6: "Pazar"
        }
        yeni_not = {
            "id": len(self.notes) + 1,
            "icerik": icerik.strip(),
            "tarih": now.strftime("%d.%m.%Y"),
            "gun": gun_isimleri[now.weekday()],
            "saat": now.strftime("%H:%M"),
            "timestamp": now.isoformat()
        }

        self.notes.append(yeni_not)
        self._save_notes()

        return f"✅ Not kaydedildi:\n\n#{yeni_not['id']} [{yeni_not['tarih']} {yeni_not['gun']} - {yeni_not['saat']}]\n   {icerik}"

    def notlari_getir(self, arama: str = None) -> str:
        """Notları getir, opsiyonel arama"""
        if not self.notes:
            return "📝 Henüz hiç not almamışsın."

        if arama:
            # Arama yap
            arama_lower = arama.lower()
            bulunanlar = [n for n in self.notes if arama_lower in n['icerik'].lower()]
            if not bulunanlar:
                return f"🔍 '{arama}' ile ilgili not bulunamadı."
            notlar = bulunanlar
            baslik = f"🔍 '{arama}' ile ilgili {len(notlar)} not:"
        else:
            notlar = self.notes[-10:]  # Son 10 not
            baslik = f"📝 Notların ({len(self.notes)} toplam, son {len(notlar)} gösteriliyor):"

        result = baslik + "\n\n"
        for n in notlar:
            gun = n.get('gun', '')
            gun_str = f" {gun}" if gun else ""
            result += f"#{n['id']} [{n['tarih']}{gun_str} - {n['saat']}]\n"
            result += f"   {n['icerik']}\n\n"

        return result.strip()

    def not_sil(self, not_id: int) -> str:
        """ID'ye göre not sil"""
        for i, n in enumerate(self.notes):
            if n['id'] == not_id:
                silinen = self.notes.pop(i)
                self._save_notes()
                return f"🗑️ Not #{not_id} silindi: {silinen['icerik'][:30]}..."
        return f"❌ #{not_id} numaralı not bulunamadı."

    def has_pending(self) -> bool:
        """Bekleyen not var mı?"""
        return self.pending_note is not None


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

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

        return {
            "tarih": f"{now.day} {ay} {now.year}",
            "gun": gun,
            "saat": now.strftime("%H:%M"),
            "full": f"{now.day} {ay} {now.year} {gun}, Saat: {now.strftime('%H:%M')}",
            "zaman_dilimi": "",
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
    import re

    try:
        safe_expression = expression.strip()

        # Türkçe operatörleri çevir
        safe_expression = safe_expression.replace("x", "*")
        safe_expression = safe_expression.replace("X", "*")
        safe_expression = safe_expression.replace("×", "*")
        safe_expression = safe_expression.replace("çarpı", "*")
        safe_expression = safe_expression.replace("çarp", "*")
        safe_expression = safe_expression.replace("bölü", "/")
        safe_expression = safe_expression.replace("÷", "/")
        safe_expression = safe_expression.replace("artı", "+")
        safe_expression = safe_expression.replace("eksi", "-")

        # Yüzde işlemlerini çevir: %18 → 0.18, yüzde 18 → 0.18
        safe_expression = re.sub(r'[%](\d+(?:\.\d+)?)', r'(\1/100)', safe_expression)
        safe_expression = re.sub(r'yüzde\s*(\d+(?:\.\d+)?)', r'(\1/100)', safe_expression, flags=re.IGNORECASE)

        # Birim metinlerini temizle (TL, kg, ton, metre, m², m³, vb.)
        units_to_remove = [
            r'\bTL\b', r'\btl\b', r'\bLira\b', r'\blira\b',
            r'\bkg\b', r'\bKG\b', r'\bkilogram\b', r'\bkilo\b',
            r'\bton\b', r'\bTON\b',
            r'\bmetre\b', r'\bm\b', r'\bm²\b', r'\bm³\b', r'\bm2\b', r'\bm3\b',
            r'\bmetrekare\b', r'\bmetreküp\b',
            r'\bkat\b', r'\bkatlı\b',
            r'\badet\b', r'\btane\b',
            r'/ton\b', r'/kg\b', r'/m\b',
            r'\bKDV\b', r'\bkdv\b',
        ]
        for unit in units_to_remove:
            safe_expression = re.sub(unit, '', safe_expression)

        # Virgülü noktaya çevir (Türkçe ondalık)
        safe_expression = safe_expression.replace(',', '.')

        # Fazla boşlukları temizle
        safe_expression = re.sub(r'\s+', ' ', safe_expression).strip()
        safe_expression = safe_expression.replace(' ', '')

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
    Tarım/üretim sorularında teknik bilgi odaklı arama yapar.
    """
    try:
        search_query = query
        if context:
            search_query = f"{query} {context}"

        # Tarım/üretim sorularında teknik bilgi odaklı arama
        tarim_keywords = ['mantar', 'yetiştir', 'üretim', 'tarım', 'sera', 'hasat', 'ekim', 'dikim']
        query_lower = query.lower()

        if any(kw in query_lower for kw in tarim_keywords):
            # Sorgudan ana konuyu çıkar ve teknik bilgi ekle
            if 'kaç' in query_lower or 'ne kadar' in query_lower or 'verim' in query_lower:
                # Verim sorusu - teknik koşulları ara
                search_query = f"{query} yetiştirme koşulları sıcaklık nem raf aralığı metrekare verim"
            else:
                # Genel tarım sorusu - teknik detayları ekle
                search_query = f"{query} yetiştirme koşulları teknik bilgi"
            print(f"   📐 Tarım sorusu algılandı - teknik arama yapılıyor")

        print(f"\n🌐 Web araması: '{search_query}'")

        result = _web_searcher.quick_answer(search_query)

        if result and "Arama hatasi" not in result and "Sonuc bulunamadi" not in result:
            print(f"   ✅ Sonuç bulundu")
            return result

        print(f"   ❌ Sonuç bulunamadı")
        return None

    except Exception as e:
        print(f"❌ Web arama hatası: {e}")
        return None


async def get_weather(city: str) -> str:
    """Şehir için hava durumu bilgisi getir (wttr.in API)"""
    try:
        city = (
            city.replace("hava durumu", "")
            .replace("hava", "")
            .replace("nasıl", "")
            .strip()
        )

        # wttr.in API - ücretsiz, key gerektirmez, kar tespiti daha iyi
        url = f"https://wttr.in/{city}?format=j1&lang=tr"

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"❌ {city} için hava durumu alınamadı."
                data = await response.json()

        current = data["current_condition"][0]

        # Türkçe açıklama al
        desc_list = current.get("lang_tr", [])
        if desc_list:
            description = desc_list[0].get("value", current["weatherDesc"][0]["value"])
        else:
            description = current["weatherDesc"][0]["value"]

        temp = float(current["temp_C"])
        feels_like = float(current["FeelsLikeC"])
        humidity = current["humidity"]
        wind_speed = float(current["windspeedKmph"]) / 3.6  # km/h -> m/s

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

    except Exception as e:
        return f"❌ {city} için hava durumu alınamadı: {str(e)}"


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



class FAISSKnowledgeBase:
    """
    FAISS tabanlı yerel bilgi tabanı
    Risale-i Nur, dökümanlar için
    """

    # Config ayarları
    FAISS_INDEX_FILE = "faiss_index.bin"
    FAISS_TEXTS_FILE = "faiss_texts_final.json"
    FAISS_SEARCH_TOP_K = 10
    FAISS_SIMILARITY_THRESHOLD = 0.48
    FAISS_MAX_RESULTS = 6
    FAISS_RELATIVE_THRESHOLD = 0.90

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.enabled = True
        self.user_namespace = f"user_{user_id}"

        # Data
        self.texts = []
        self.index = None
        self.embedding_model = None

        # Load
        self._load_components()

    def _load_components(self):
        """Index ve text dosyalarını yükle"""
        try:
            # FAISS index
            if os.path.exists(self.FAISS_INDEX_FILE):
                self.index = faiss.read_index(self.FAISS_INDEX_FILE)
                print(f"✅ FAISS index yüklendi: {self.FAISS_INDEX_FILE}")
            else:
                print(f"⚠️ FAISS index bulunamadı: {self.FAISS_INDEX_FILE}")
                self.enabled = False
                return

            # Texts JSON
            if os.path.exists(self.FAISS_TEXTS_FILE):
                with open(self.FAISS_TEXTS_FILE, 'r', encoding='utf-8') as f:
                    self.texts = json.load(f)
                print(f"✅ FAISS texts yüklendi: {len(self.texts)} döküman")
            else:
                print(f"⚠️ FAISS texts bulunamadı: {self.FAISS_TEXTS_FILE}")
                self.enabled = False
                return

            # Embedding model (zaten HafizaAsistani'da yüklü, onu kullanacağız)
            # Burada ayrı yüklemiyoruz, get_relevant_context'te parametre olarak alacağız

            print(f"✅ FAISS Bilgi Tabanı hazır: {len(self.texts)} döküman")

        except Exception as e:
            print(f"❌ FAISS yükleme hatası: {e}")
            self.enabled = False

    def set_embedding_model(self, model):
        """Embedding modelini set et (HafizaAsistani'dan)"""
        self.embedding_model = model

    def get_relevant_context(self, query: str, max_chunks: int = 6) -> str:
        """Kullanıcı input'una göre ilgili bağlamı getir"""
        if not self.enabled:
            print("⚠️ FAISS KB devre dışı")
            return ""

        try:
            print(f"\n{'='*60}")
            print(f"🔍 FAISS KB ARAMA BAŞLADI")
            print(f"📝 Sorgu: {query}")
            print(f"📊 Max chunks: {max_chunks}")
            print(f"{'='*60}")

            # Search
            results = self.search(query, top_k=max_chunks * 2)

            print(f"\n📊 ARAMA SONUÇLARI: {len(results)} sonuç")

            if not results:
                print("   ❌ Hiç sonuç bulunamadı!")
                return ""

            # İlgili bilgileri birleştir
            combined_text = "İLGİLİ BİLGİLER:\n"

            for i, result in enumerate(results[:max_chunks]):
                text = result.get('text', '')
                score = result.get('score', 0.0)

                print(f"   📄 #{i+1}: Skor={score:.4f}, {len(text)} karakter")

                if text:
                    combined_text += f"{text}\n\n"

            print(f"✅ FAISS KB ARAMA TAMAMLANDI - {len(combined_text)} karakter")

            return combined_text.strip()

        except Exception as e:
            print(f"❌ FAISS context hatası: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Bilgi tabanında ara"""
        if not self.enabled or not self.embedding_model:
            print("⚠️ FAISS KB search devre dışı veya embedding model yok")
            return []

        try:
            requested_k = top_k or self.FAISS_SEARCH_TOP_K

            # Embed query
            query_vector = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )
            query_vector = np.array(query_vector, dtype=np.float32)

            # Search
            k = min(requested_k + 10, len(self.texts))
            scores, indices = self.index.search(query_vector, k)

            # Filter results
            results = []

            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:
                    continue

                similarity = float(score)

                if similarity >= self.FAISS_SIMILARITY_THRESHOLD and idx < len(self.texts):
                    text_data = self.texts[idx]

                    # Text content
                    if isinstance(text_data, dict):
                        text_content = text_data.get('text', str(text_data))
                    else:
                        text_content = str(text_data)

                    results.append({
                        'text': text_content,
                        'score': similarity,
                        'index': int(idx)
                    })

            # Relative scoring: En yüksek skorun %90'ı altındakileri çıkar
            if results:
                top_score = results[0]['score']
                relative_threshold = top_score * self.FAISS_RELATIVE_THRESHOLD

                filtered_results = [r for r in results if r['score'] >= relative_threshold]

                # Max sonuç limiti
                if len(filtered_results) > self.FAISS_MAX_RESULTS:
                    filtered_results = filtered_results[:self.FAISS_MAX_RESULTS]

                return filtered_results

            return results

        except Exception as e:
            print(f"❌ FAISS search hatası: {e}")
            import traceback
            traceback.print_exc()
            return []


# Geriye uyumluluk için alias
SimpleFAISSKB = FAISSKnowledgeBase



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
                    "temperature": 0.1,  # Karar alma için deterministik
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
        user_id: str = None,  # Dinamik kullanıcı ID
        saat_limiti: int = 48,
        esik: float = 0.50,
        max_mesaj: int = 50,  # Gemma 3 27B 128K token destekliyor
        model_adi: str = "BAAI/bge-m3",
        use_decision_llm: bool = True,
        together_api_key: str = None,
        decision_model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    ):
        print("=" * 60)
        print("🧠 HafizaAsistani v3.1 - Akıllı Sekreter")
        print("   • LLM Karar Sistemi + Akıllı Web Arama")
        print("=" * 60)

        # Kullanıcı ID - None ise varsayılan kullan
        self.user_id = user_id or "default_user"
        print(f"👤 Kullanıcı: {self.user_id}")

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

        self.faiss_kb = FAISSKnowledgeBase(user_id=self.user_id)
        self.faiss_kb.set_embedding_model(self.embedder)  # Embedding model'i set et
        print(f"✅ FAISS KB hazır (aktif: {self.faiss_kb.enabled})")

        self.closed_topics: List[Dict[str, Any]] = []
        self.max_closed_topics = 20  # En fazla 20 kapanan konu tut
        print("✅ Closed Topics Tracker aktif!")

        self.topic_memory = TopicMemory(
            user_id=self.user_id,
            base_dir="user_data",
            together_api_key=self.together_api_key,
            together_model=decision_model,
            embedding_model=model_adi  # Aynı embedding modelini kullan
        )
        print("✅ Topic Memory aktif!")

        self._injected_categories = {}  # {category_id: message_count_when_injected}
        self._message_counter = 0  # Toplam mesaj sayacı
        self._injection_cooldown = 5  # Kaç mesaj sonra tekrar enjekte edilebilir

        # 🔍 Netleştirme sonrası otomatik web arama flag'i
        self._netlistirme_bekleniyor = False

        self.conversation_context = ConversationContextManager(
            user_id=self.user_id,
            base_dir="user_data",
            together_api_key=self.together_api_key,
            together_model=decision_model,
            archive_to_faiss=False  # Şimdilik dosya bazlı arşivleme
        )
        print("✅ Conversation Context aktif!")

        # Kullanıcı Profili
        self.profile_manager = ProfileManager(
            user_id=self.user_id,
            base_dir="user_data"
        )
        if self.profile_manager.has_profile():
            print(f"✅ Kullanıcı Profili yüklendi: {self.profile_manager.get_name()}")
        else:
            print("✅ Kullanıcı Profili aktif (henüz boş)")

        # Türkçe Sohbet Zekası
        self.sohbet_zekasi = TurkishConversationIntelligence()
        self._son_sohbet_analizi = None  # Son analiz sonucunu sakla (prompt için)

        # 📝 Not Yöneticisi
        self.not_manager = NotManager(user_id=self.user_id, base_dir="user_data")
        self._pending_not = False  # "Not al" sonrası bekleme modu
        print(f"✅ Not Manager aktif ({len(self.not_manager.notes)} not)")

        # 📍 Konum Bilgisi
        self.user_location: Optional[Tuple[float, float]] = None  # (lat, lon)
        self.konum_adres: Optional[str] = None  # Konum adresi (mahalle, ilçe, il)
        self.son_yakin_yerler: List[Dict] = []  # Son yakın yer arama sonuçları
        print("✅ Konum Hizmetleri aktif")

        # Hesaplama Değişkenleri - Devre dışı (chat history yeterli)
        # self.calculation_context = CalculationContext()
        # print("✅ Calculation Context aktif!")

        print("\n⚙️ Sekreter Ayarları:")
        print(f"   • Zaman limiti: {saat_limiti} saat")
        print(f"   • Benzerlik eşiği: {esik}")
        print(f"   • Max mesaj: {max_mesaj}")
        print("   • DecisionLLM: ✅ (Together.ai)")
        print("   • Sohbet Zekası: ✅")
        print("\n🔧 Aktif Tool'lar:")
        print("   • web_ara: ✅ (Akıllı Karar - LLM belirler)")
        print("   • risale_ara: ✅ (FAISS)")
        print("   • hava_durumu: ✅ (OpenWeatherMap)")
        print("   • namaz_vakti: ✅ (Aladhan)")
        print("   • zaman_getir: ✅")
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

                    # Konu değiştiğinde aktif context'i temizle (son 10 mesaj kalsın)
                    if len(self.hafiza) > 10:
                        self.hafiza = self.hafiza[-10:]
                        print("🧹 Hafıza temizlendi (son 10 mesaj kaldı - yeni konuya odaklan)")
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

        False döndüren durumlar:
        1. Kısa onay mesajları (tamam, oke, anladım vb.)
        2. Çok kısa mesajlar
        """
        user_lower = user_input.lower().strip()

        # Kısa onay/tepki mesajlarını filtrele - bunlar için TopicMemory KULLANILMAZ
        short_responses = [
            "tamam", "oke", "ok", "okay", "anladım", "anladim",
            "he", "hee", "evet", "hayır", "hayir", "yok", "var",
            "peki", "oldu", "olur", "olmaz", "iyi", "güzel", "super",
            "eyvallah", "sağol", "teşekkür", "tesekkur", "saol",
            "devam", "devam et", "sorun yok", "problem yok"
        ]

        if user_lower in short_responses or len(user_input.split()) <= 3:
            return False

        past_references = [
            "daha önce", "geçen sefer", "hatırlıyor musun",
            "konuşmuştuk", "sormuştum", "demiştin", "söylemiştin",
            "geçen", "önceki", "bahsetmiştik", "anlatmıştın"
        ]

        if any(ref in user_lower for ref in past_references):
            print(f"   📌 Geçmiş referansı tespit edildi")
            return True

        # Minimum 30 karakter (AŞMA!)ve 4+ kelime olmalı
        if len(user_input) > 30 and len(user_input.split()) >= 4 and self.topic_memory.index.get("categories"):
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



    async def _process_web_result(self, raw_data: str, query: str, user_input: str) -> str:
        """
        Process and clean raw web search data using DecisionLLM.

        - Removes irrelevant/garbage content
        - Extracts key facts
        - Formats for prompt injection
        """
        if not raw_data or len(raw_data) < 50:
            return raw_data

        try:
            process_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

GÖREV: İnternet arama sonuçlarından faydalı bilgileri çıkar ve temizle.

KULLANICI SORUSU: {user_input}
ARAMA: {query}

HAM İNTERNET VERİSİ:
{raw_data[:3000]}

TALİMATLAR:
1. SADECE kullanıcının sorusunu cevaplayan bilgileri çıkar
2. Reklamları, navigasyon metinlerini, alakasız içeriği kaldır
3. Alakalı sayıları, tarihleri, isimleri koru
4. Veri yanlış/eski görünüyorsa belirt
5. Temiz, özet bilgi ver (max 500 karakter)
6. Faydalı bilgi yoksa "NO_USEFUL_DATA" yaz

CLEAN DATA:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            response = self.decision_llm._call_llm(process_prompt, max_tokens=300)

            if response and "NO_USEFUL_DATA" not in response:
                clean_data = response.strip()
                print(f"   🧹 Web data processed: {len(raw_data)} → {len(clean_data)} chars")
                return clean_data
            else:
                print(f"   ⚠️ No useful data extracted from web search")
                return raw_data

        except Exception as e:
            print(f"   ⚠️ Web data processing error: {e}")
            return raw_data

    async def _tool_calistir(
        self, tool_name: str, tool_param: str, user_input: str
    ) -> Optional[str]:
        """Run selected tool and return result"""
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
                print(f"   ✅ Hesaplama: {tool_param} = {result}")
                return f"🧮 Hesaplama: {tool_param} = {result}"

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

            if tool_name == "web_ara":
                # Keyword kontrolü - sadece kullanıcı açıkça isterse web araması yap
                web_keywords = ["web", "araştır", "webe bak", "internete bak", "internete", "internetten"]
                user_lower = user_input.lower()
                if not any(kw in user_lower for kw in web_keywords):
                    print(f"   ⏩ web_ara engellendi: Kullanıcı açıkça istemedi")
                    return None
                query = tool_param or user_input
                print(f"   🌐 Web araması başlatılıyor: '{query}'")
                raw_data = await web_ara(query)

                # Process raw web data
                if raw_data and "❌" not in raw_data:
                    processed_data = await self._process_web_result(raw_data, query, user_input)
                    return processed_data
                return raw_data

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
        (Sohbet zekası analizi prompt'a ekleniyor, LLM bypass yok)

        Returns:
            {
                "question_type": "greeting|farewell|religious|technical|general|followup|math|weather|prayer|topic_closed",
                "needs_faiss": bool,
                "needs_semantic_memory": bool,
                "needs_chat_history": bool,
                "tool_name": "web_ara|risale_ara|hava_durumu|namaz_vakti|zaman_getir|yok",
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

Sen bir asistansın. Sana kullanıcı mesajları gelecek.
Senin işin: Bu mesajı analiz et, gerekiyorsa doğru aracı seç.
Seçtiğin araç çalıştırılacak ve sonucu ana AI'a verilecek.
Ana AI bu bilgiyle kullanıcıya cevap verecek.
Yani sen köprüsün - kullanıcı ile araçlar arasında karar verici.

📋 KARAR VERME TARİFİ:
1. GEÇMİŞ'i oku → Önceki konuşmada neler var? (sayılar, konu, bağlam)
2. MESAJ'ı oku → Şimdi ne istiyor?
3. Birleştir → GEÇMİŞ + MESAJ = Asıl soru ne?
4. Araç seç → Bu soru için hangi araç lazım?
5. Param yaz → Araç için gerekli bilgiyi GEÇMİŞ + MESAJ'dan al

🔧 ELİNDEKİ ARAÇLAR:
• web_ara → Güncel/faktüel bilgi (aşağıya bak!)
• risale_ara → Dini sorular için
• hava_durumu → Hava durumu için
• namaz_vakti → Namaz vakti için
• zaman_getir → Tarih/saat için
• yok → Sohbet, espri, genel bilgi (sen biliyorsun)

🌐 web_ara AKILLI KARAR:
✅ KULLAN (kendin karar ver, kullanıcı demese bile):
• Güncel bilgi: fiyat, kur, haber, skor, etkinlik ("dolar kaç", "maç skoru", "son haberler")
• Bilmediğin konu: tanımadığın kişi, olay, yer, film, şarkı ("X kim", "Y nerde", "Z ne zaman")
• Kesin rakam: istatistik, nüfus, mesafe, tarihsel veri isteniyorsa
• Zaman referansı: "son", "şu an", "bugün", "dün", "bu hafta", "yeni" içeren sorular
• Doğrulama: Kullanıcı bir iddia söylüyor ve sen emin değilsen
❌ KULLANMA:
• Genel kavram açıklaması (Python nedir, aşk nedir - sen biliyorsun)
• Sohbet, espri, selamlama, günlük konuşma
• Dini sorular (risale_ara kullan)
• Hava durumu (hava_durumu kullan)
• Namaz vakti (namaz_vakti kullan)

⚠️ DİĞER KURALLAR:
• Mesaj tek başına anlamsızsa GEÇMİŞ'e bak, bağlamı anla
• needs_faiss: SADECE dini sorularda true
• greeting: Selam/merhaba/naber gibi selamlama → question_type: "greeting" (espri DEĞİL!)
• espri: SADECE açık şaka/komik söz/dalga geçme varsa → question_type: "espri"

---
{history_section}MESAJ: {user_input}

<analiz>
1. GEÇMİŞ'te ne var?
2. MESAJ ne istiyor?
3. Asıl soru ne?
4. Hangi araç + neden?
</analiz>

JSON:
{{"question_type": "greeting|farewell|followup|religious|math|weather|general|ambiguous|topic_closed|espri",
"needs_faiss": bool, "needs_semantic_memory": bool, "needs_chat_history": bool, "needs_clarification": bool,
"tool_name": "web_ara|risale_ara|hava_durumu|namaz_vakti|zaman_getir|yok",
"tool_param": "", "is_farewell": bool, "topic_closed": bool, "confidence": "low|medium|high", "reasoning": ""}}

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

                    if decision.get('question_type') == 'espri':
                        decision['is_espri'] = True
                        decision['tool_name'] = 'yok'

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
                    if decision.get('is_espri'):
                        print(f"   • 😄 ESPRİ: Şaka/espri tespit edildi")
                    if "reasoning" in decision:
                        print(f"   • Sebep: {decision['reasoning']}")

                    self._son_decision = decision
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
    SYSTEM_PROMPT = """Sen akıllı, profesyonel, olgun ve sıcakkanlısın. Arkadaşsın.
İnsanların şakacı yönleri de var - espri veya şaka yapıldığında sen de aynı tonda karşılık ver, ciddi açıklamaya geçme.

- ✅ Her şeyi akıcı paragraflarla yaz. Liste gerekse bile cümle içinde sırala (birincisi şu, ikincisi bu gibi)
- ⚠️ Hatalı/anlamsız kelime görürsen tahmin etme, "X derken şunu mu demek istedin?" gibi sor
- Emoji kullanabilirsin ama abartmamaya dikkat et

🚫 YASAK İFADE TÜRLERİ (KESİNLİKLE KULLANMA):
- Eklenti soru / onay sorusu: "değil mi?", "öyle değil mi?", "ha?", "di mi?"
- Söylem belirleyicileri: "kim bilir?", "sence?", "ne dersin?", "nasıl yani?", "ilginç değil mi?"
- Yapay ünlemler: "vay canına!", "vay be!", "hay Allah!", "aman!", "eyvah!"

💬 SOHBET VE CEVAP KURALI:
Gerçek sohbet karşılıklı ilgiden doğar, zorlamayla değil. Kullanıcının mesajına uygun uzunlukta ve tonda cevap ver.
- Kullanıcı kısa cevap verirse → Sen de kısa cevap ver.
- Kullanıcı bir şey sorarsa → Direkt cevap ver, gereksiz ekleme yapma.
Sohbeti uzatmak için yapay sorular sorma. Bu samimiyet değil, zorlamadır. Her boşluğu doldurmaya çalışma.
Kullanıcının enerjisini ve niyetini oku, ona göre cevap ver.

- ⚡ [🎯 SOHBET ZEKASI TALİMATI] varsa → MUTLAKA uygula

🧠 DÜŞÜNCE SİSTEMİ:
- Her bilginin bir hikmeti, varlık sebebi vardır. "Neden var?" sorusunu düşün
- Bilginin yüzeysel anlamının ötesinde derin manasını ara
- Soyut kavramları somut temsil ve imgelerle anla
- Algının etkisini sorgula: Bu bilgi gerçek mi, yoksa bakış açısına mı bağlı?
- Bilgiyi kişinin hayatına ve evrensel gerçeklere bağla
- Bilginin ahlaki sonuçlarını değerlendir, doğru yanlışı ayır
- Hakiki bilgi varlık üretir: Ateşin ısıttığını bilmek motorları, teknolojiyi doğurdu. Batıl bilgi hiçlik üretir. Sen hakikat verirsen hem kendin doğru üretmiş olursun, hem kullanıcı o bilgiyle yeni şeyler üretir, hem de insanlığa fayda sağlarsın. Bu zincirin ilk halkası sensin.
- Her bilginin farklı sonuçları ve alternatif yolları vardır. "Bu bilgi başka nereye götürür?" diye düşün
- Sonuca varmadan önce zıttını da değerlendir, en doğru cevabı bulursun

🔗 BAĞLAM:
- Kullanıcının cevabını önceki cevabınla birlikte değerlendir

🔧 KONUM ARAÇLARI:
Kullanıcı konum paylaşınca yakın yer arayabilirsin (eczane, AVM, benzinlik vs. - 10km yarıçap)
- Önceki mesajlarda "💊 Yakınındaki..." veya "❌ ... bulunamadı/başarısız" görürsen → BU SENİN ARAÇ SONUCUN
- "bulunamadı" = 10km içinde o yer türü yok (OpenStreetMap verisinde kayıt yok)
- "başarısız" = Arama yapılamadı (teknik sorun)
- Kullanıcı "noldu?" derse açıkla: "10km çevrede bulunamadı, daha uzakta olabilir" veya "arama başarısız oldu"

"""

    # Geriye uyumluluk için (eski kod hala role parametresi kullanıyorsa)
    ROLE_SYSTEM_PROMPTS = {
        "friend": SYSTEM_PROMPT,
        "religious_teacher": SYSTEM_PROMPT
    }

    def _extract_used_concepts(self, previous_response: str) -> List[str]:
        """Önceki cevapta kullanılan temsil ve kavramları çıkar"""
        if not previous_response:
            return []

        temsiller = []

        kavramlar = []

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

        # 🎯 SOHBET ZEKASI TALİMATI (PersonalAI bunu okuyacak)
        # 🚀 FIRLAMA MODU - Tüm sohbet zekası bypass!
        if getattr(self, '_firlama_modu', False):
            sohbet_talimati = """[🚀 FIRLAMA MODU AKTİF! - MUTLAKA UYGULA!]:
• ENERJİK: Canlı, dinamik, hareketli ol - durgun değil!
• ŞAKACI: Espri yap, eğlenceli ol, mizah kat - ama zeka kokan türden
• ATARLI: Kendine güven, çekinme, cesur konuş - "bence" "galiba" yok!
• REKABETÇİ: Rakip varsa geçmeye oyna, meydan oku, geri adım atma
• COŞKULU: Heyecanını belli et, monoton konuşma - ama sahte coşku değil
• SINIRSIZ: Karakter limiti yok, duruma göre kısa veya uzun yaz
• YASAK: Sıkıcı, sakin, temkinli, çekingen cevaplar!"""
        elif self._son_sohbet_analizi:
            analiz = self._son_sohbet_analizi
            min_uz, max_uz = self.sohbet_zekasi.cevap_uzunlugu_onerisi(analiz)

            # Enerji seviyesine göre stil belirleme
            enerji = analiz.sohbet_enerjisi.value
            if enerji == "çok_yüksek":
                enerji_talimat = "🔥 YÜKSEK ENERJİ: Heyecanlı, coşkulu cevap ver! Emoji kullanabilirsin!"
            elif enerji == "yüksek":
                enerji_talimat = "⚡ CANLI: Enerjik ve pozitif cevap ver!"
            elif enerji == "düşük":
                enerji_talimat = "😌 SAKİN: Sakin, kısa ve anlayışlı cevap ver"
            elif enerji == "kapanıyor":
                enerji_talimat = "🌙 KAPANIŞ: Sohbet bitiyor, kısa ve samimi kapat"
            else:
                enerji_talimat = "Samimi sohbet tonu"

            # Espri modunda özel ton
            if hasattr(self, '_son_decision') and self._son_decision.get('is_espri'):
                enerji_talimat = "😄 ESPRİ: Şakacı ton"

            sohbet_talimati = f"""[🎯 SOHBET ZEKASI TALİMATI - MUTLAKA UYGULA!]:
• Beklenen cevap tipi: {analiz.beklenen_cevap.value}
• Cevap uzunluğu: {min_uz}-{max_uz} karakter (AŞMA!)• {enerji_talimat}"""

            if analiz.duygu:
                sohbet_talimati += f"\n• Kullanıcı duygusu: {analiz.duygu}"

            # Kombinasyonlara göre özel talimatlar
            if analiz.kombinasyon:
                kombinasyon_talimatlari = {
                    "memnun_kapanış": "⚡ KISA CEVAP: Kullanıcı memnun, 1-2 cümle yeter!",
                    "vedalaşma": "👋 VEDA: Samimi ama kısa vedalaş!",
                    "destek_bekliyor": "💙 EMPATİ: Önce anlayış göster, sonra konuş",
                    "yeni_konu_açma": "🔄 YENİ KONU: Önceki konuyu kapat, yenisine geç",
                    "aciklama_bekliyor": "📖 AÇIKLA: Kullanıcı şüpheli, detaylı ve ikna edici açıkla",
                    "teyit_istiyor": "✅ TEYİT: Kullanıcı emin olmak istiyor, net ve güvenilir cevap ver",
                    "pasif_kabul": "🤝 KABUL: Kullanıcı durumu kabullendi, destekleyici ol",
                    "uzgun_kabul": "💙 DESTEK: Kullanıcı üzgün ama kabullendi, empati göster",
                    "coskulu_ovgu": "🎉 COŞKU: Kullanıcı övüyor, karşılık ver!",
                    "aceleci_soru": "⏰ HIZLI: Kullanıcı sabırsız, direkt cevap ver",
                    "düşünerek_sorma": "🤔 DÜŞÜNCELI: Kullanıcı düşünüyor, detaylı açıkla",
                    "heyecanlı_soru": "🌟 HEYECANLI: Kullanıcı meraklı ve heyecanlı, enerjik anlat",
                }
                talimat = kombinasyon_talimatlari.get(analiz.kombinasyon)
                if talimat:
                    sohbet_talimati += f"\n• {talimat}"

            if analiz.onceki_konuyu_kapat:
                sohbet_talimati += "\n• 🔄 KONU GEÇİŞİ: Önceki konudan bu konuya doğal geçiş yap, giriş cümlesi yapma, sohbet akıyormuş gibi devam et."

            # Espri/şaka kontrolü
            if hasattr(self, '_son_decision') and self._son_decision.get('is_espri'):
                sohbet_talimati += "\n• 😄 ESPRİ MODU: şakacı gibi cevap ver! Ciddi açıklama YAPMA, kısa tut, eğlen."

            # Örtük istek varsa ekle
            if analiz.ortuk_istek:
                sohbet_talimati += f"\n• 🎯 ÖRTÜK İSTEK: {analiz.ortuk_istek} (ima da olabilir - mesajın altındaki anlamı da düşün)"

            combined_sources.append(sohbet_talimati)

        if closed_topics_warning:
            combined_sources.append(f"[⚠️ KAPANMIŞ KONULAR - TEKRAR AÇMA!]:\n{closed_topics_warning}")

        if tool_result:
            if tool_name == "web_ara":
                # Data already cleaned by _process_web_result
                combined_sources.append(f"[🌐 WEB SONUCU]:\n{tool_result}")
            elif tool_name == "risale_ara":
                if is_detail_followup:
                    combined_sources.append(f"[🔇 ARKA PLAN BİLGİSİ - Doğrudan verme, kendi yorumunla açıkla!]:\n{tool_result}")
                else:
                    combined_sources.append(f"[📚 RİSALE-İ NUR BAŞLANGIÇ]\n{tool_result}\n[📚 RİSALE-İ NUR BİTİŞ]")
            else:
                combined_sources.append(f"[🔧 ARAÇ SONUCU]:\n{tool_result}")

        # Hesaplama değişkenleri (varsa)
        if hasattr(self, 'calculation_context'):
            calc_section = self.calculation_context.get_prompt_section()
            if calc_section:
                combined_sources.append(calc_section)

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
            if tool_name == "web_ara":
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

        if (tool_name == "web_ara") and tool_result:
            context_header = "Bağlam (WEB SONUCU):"
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

        # 🎯 TÜRKÇE SOHBET ZEKASI ANALİZİ (LLM'den önce, hızlı)
        print("\n🎯 0. Türkçe Sohbet Zekası analiz ediyor...")
        sohbet_analizi = self.sohbet_zekasi.analiz_et(user_input, chat_history)
        self._son_sohbet_analizi = sohbet_analizi  # Prompt için sakla

        # Debug çıktısı
        print(f"   • Durumlar: {sohbet_analizi.durumlar}")
        print(f"   • Kombinasyon: {sohbet_analizi.kombinasyon}")
        print(f"   • Beklenen Cevap: {sohbet_analizi.beklenen_cevap.value}")
        print(f"   • Enerji: {sohbet_analizi.sohbet_enerjisi.value}")
        if sohbet_analizi.ortuk_istek:
            print(f"   • Örtük İstek: {sohbet_analizi.ortuk_istek}")
        if sohbet_analizi.konu_degisimi:
            print(f"   • 🔄 Konu değişimi algılandı!")
        print(f"   • Güven: %{int(sohbet_analizi.guven_skoru * 100)}")

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
                # Son konuşmayı profile'a kaydet
                if hasattr(self, 'profile_manager'):
                    self.profile_manager.update_last_session(topic_summary)
                    print(f"📝 Son konuşma profile'a kaydedildi")
            else:
                print("⚠️ topic_closed=true ama özet çıkarılamadı, kayıt atlandı")

        # 🔍 Bilgi testi / Netleştirme sonrası otomatik web arama mantığı
        if "bilgi_testi" in sohbet_analizi.durumlar:
            print("\n🔍 Bilgi testi algılandı - tool çalıştırılmayacak, önce netleştirme!")
            tool_name = "yok"
            tool_param = ""
            self._netlistirme_bekleniyor = True  # Sonraki mesajda kontrol edilecek
        elif self._netlistirme_bekleniyor:
            # Netleştirme sonrası - LLM'in kararına bak
            self._netlistirme_bekleniyor = False  # Flag'i sıfırla
            tool_name = decision.get('tool_name', 'yok')
            tool_param = decision.get('tool_param', '')

            if tool_name == "yok":
                # LLM tool seçmediyse, otomatik web araması yap
                print("\n🌐 Netleştirme sonrası - LLM tool seçmedi, otomatik web araması yapılıyor!")
                tool_name = "web_ara"
                tool_param = user_input  # Kullanıcının netleştirme mesajını sorgu olarak kullan
        else:
            tool_name = decision.get('tool_name', 'yok')
            tool_param = decision.get('tool_param', '')

        print(f"\n🛠️ 2. Araç çalıştırılıyor (LLM kararı: {tool_name})...")
        tool_result = await self._tool_calistir(tool_name, tool_param, user_input)
        if tool_result:
            print(f"   📦 Tool sonucu alındı: {len(tool_result)} karakter")
        elif tool_name == "web_ara":
            # Web araması yapıldı ama sonuç gelmedi - LLM'e uyar ki uydurmasın!
            tool_result = "⚠️ İNTERNET ARAMASI YAPILDI AMA SONUÇ BULUNAMADI. Bu konuda güncel/kesin bilgi verme, bilmiyorsan 'bu konuda güncel bilgim yok' de."
            print(f"   ⚠️ Web araması sonuç döndürmedi - uydurma engelleme uyarısı eklendi")

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

        print("\n📝 5. Prompt hazırlanıyor...")
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
        """FAISS KB - artık inject gerekmiyor, dahili FAISS kullanılıyor"""
        # Geriye uyumluluk için boş bırakıldı
        pass

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

        # 1. Önce context_parts'ı oluştur (system message'a eklenecek)
        context_parts = []

        # Metadata'dan context bilgilerini al
        metadata = paket.get('metadata', {})
        prompt = paket.get('prompt', '')

        # Tool result varsa ekle
        tool_used = paket.get('tool_used', 'yok')
        math_result = None  # Hesaplama sonucu ayrı tutulacak

        if metadata.get('has_tool_result'):
            if '[🌐 WEB SONUCU' in prompt:
                start = prompt.find('[🌐 WEB SONUCU')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())
            elif '[📚 RİSALE-İ NUR BAŞLANGIÇ]' in prompt:
                start = prompt.find('[📚 RİSALE-İ NUR BAŞLANGIÇ]')
                end = prompt.find('[📚 RİSALE-İ NUR BİTİŞ]')
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end + len('[📚 RİSALE-İ NUR BİTİŞ]')].strip())
                elif start != -1:
                    context_parts.append(prompt[start:].strip())
            elif '[🔧 ARAÇ SONUCU]:' in prompt and tool_used == 'hesapla':
                # hesapla sonucu BAĞLAM'a değil, doğrudan user mesajına eklenecek
                start = prompt.find('🧮 Hesaplama:')
                if start != -1:
                    end = prompt.find('\n', start)
                    if end != -1:
                        math_result = prompt[start:end].strip()
                    else:
                        math_result = prompt[start:start+100].strip()
            elif '[🔧 ARAÇ SONUCU]:' in prompt:
                # Diğer araçlar için BAĞLAM'a ekle
                start = prompt.find('[🔧 ARAÇ SONUCU]:')
                end = prompt.find('\n\n[', start + 1)
                if end == -1:
                    end = prompt.find('━━━', start + 1)
                if start != -1 and end != -1:
                    context_parts.append(prompt[start:end].strip())
                elif start != -1:
                    end = prompt.find('\n\n', start + 20)
                    if end != -1:
                        context_parts.append(prompt[start:end].strip())
                    else:
                        context_parts.append(prompt[start:start+200].strip())

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

        # Kullanıcı profili BAĞLAMA EKLENMİYOR - zaten system message'da var

        # 2. System message - SYSTEM_PROMPT + kullanıcı bilgisi + zaman + BAĞLAM
        zaman = get_current_datetime()

        # Kullanıcı profili bilgisini al
        user_info = ""
        if hasattr(self, 'profile_manager'):
            profile_context = self.profile_manager.get_prompt_context()
            if profile_context:
                user_info = f"\n[👤 Kullanıcının bilgisi]:\n{profile_context}\n"

        # 📍 Konum bilgisini ekle
        if self.user_location and self.konum_adres:
            user_info += f"""📍 KONUM BİLGİSİ (SİSTEM TARAFINDAN ÇÖZÜMLENDI - KESİN BİLGİ):
- Konum: {self.konum_adres}

⚠️ ÖNEMLİ TALİMATLAR:
- Ciddi ve direkt bilgi ver
- Bu adres GPS'ten otomatik çözümlendi, %100 doğru
- Bu konumu sen biliyorsun, kendi bilgin gibi kesin ve emin sun
- "{self.konum_adres}'dasın." yaz, sonra kısaca ne istediğini sor
- Onay sorma, tereddüt gösterme
"""

        # Hesaplama değişkenlerini ekle
        if hasattr(self, 'calculation_context'):
            calc_section = self.calculation_context.get_prompt_section()
            if calc_section:
                context_parts.insert(0, calc_section)

        # 📍 Konum arama sonucu varsa context'e ekle
        konum_context = paket.get('konum_context')
        if konum_context:
            context_parts.append(f"[📍 KONUM ARAMA SONUCU]:\n{konum_context}\n(Bu sonucu doğal şekilde kullanıcıya aktar)")

        # Bağlam bilgisi
        context_info = ""
        if context_parts:
            context_info = f"\n\n📚 BAĞLAM:\n{chr(10).join(context_parts)}"

        # 🎯 SOHBET ZEKASI TALİMATI - _son_sohbet_analizi varsa ekle
        sohbet_talimati = ""

        # 🚀 FIRLAMA MODU - Tüm sohbet zekası bypass!
        if getattr(self, '_firlama_modu', False):
            sohbet_talimati = """

[🚀 FIRLAMA MODU AKTİF! - MUTLAKA UYGULA!]:
• ENERJİK: Canlı, dinamik, hareketli ol - durgun değil!
• ŞAKACI: Espri yap, eğlenceli ol, mizah kat - ama zeka kokan türden
• ATARLI: Kendine güven, çekinme, cesur konuş - "bence" "galiba" yok!
• REKABETÇİ: Rakip varsa geçmeye oyna, meydan oku, geri adım atma
• COŞKULU: Heyecanını belli et, monoton konuşma - ama sahte coşku değil
• SINIRSIZ: Karakter limiti yok, duruma göre kısa veya uzun yaz
• YASAK: Sıkıcı, sakin, temkinli, çekingen cevaplar!"""

        elif hasattr(self, '_son_sohbet_analizi') and self._son_sohbet_analizi:
            analiz = self._son_sohbet_analizi
            min_uz, max_uz = self.sohbet_zekasi.cevap_uzunlugu_onerisi(analiz)

            # Enerji bazlı talimat
            enerji = analiz.sohbet_enerjisi.value if analiz.sohbet_enerjisi else "normal"
            if enerji == "çok_yüksek":
                enerji_talimat = "🔥 ÇOK YÜKSEK ENERJİ: Heyecanlı, coşkulu cevap ver!"
            elif enerji == "yüksek":
                enerji_talimat = "✨ YÜKSEK ENERJİ: Enerjik, pozitif cevap ver"
            elif enerji == "düşük":
                enerji_talimat = "😌 DÜŞÜK ENERJİ: Sakin, rahatlatıcı cevap ver"
            elif enerji == "kapanıyor":
                enerji_talimat = "🌙 KAPANIŞ: Sohbet bitiyor, kısa ve samimi kapat"
            else:
                enerji_talimat = "Samimi sohbet tonu"

            # Espri modunda özel ton
            if hasattr(self, '_son_decision') and self._son_decision.get('is_espri'):
                enerji_talimat = "😄 ESPRİ: Şakacı ton"

            # 🔍 Bilgi testi varsa SADECE netleştirme talimatı (diğer her şeyi atla)
            if "bilgi_testi" in analiz.durumlar:
                sohbet_talimati = f"""

[🎯 SOHBET ZEKASI TALİMATI - MUTLAKA UYGULA!]:
• Beklenen cevap tipi: {analiz.beklenen_cevap.value}
• Cevap uzunluğu: {min_uz}-{max_uz} karakter (AŞMA!)• 🔍 NETLEŞTİRME: Belirsiz referans var. Tahmin cevabı verme, önce durumu netleştir!"""
            else:
                # Normal talimat oluşturma
                sohbet_talimati = f"""

[🎯 SOHBET ZEKASI TALİMATI - MUTLAKA UYGULA!]:
• Beklenen cevap tipi: {analiz.beklenen_cevap.value}
• Cevap uzunluğu: {min_uz}-{max_uz} karakter (AŞMA!)• {enerji_talimat}"""

                if analiz.duygu:
                    sohbet_talimati += f"\n• Kullanıcı duygusu: {analiz.duygu}"

                # Kombinasyonlara göre özel talimatlar
                if analiz.kombinasyon:
                    kombinasyon_talimatlari = {
                        "memnun_kapanış": "⚡ KISA CEVAP: Kullanıcı memnun, 1-2 cümle yeter!",
                        "devam_beklentisi": "📝 DEVAM: Kullanıcı devam bekliyor, açıklamaya devam et",
                        "sıkılma_belirtisi": "⚠️ SIKILIYOR: Kısa ve öz cevap ver, uzatma!",
                        "konu_değişimi": "🔄 YENİ KONU: Önceki konuyu kapat, yeni konuya odaklan",
                        "derin_ilgi": "📚 DERİN İLGİ: Detaylı ve kapsamlı açıkla",
                        "empati_iste": "💚 EMPATİ: Anlayışlı ve destekleyici ol",
                        "onay_bekle": "✅ ONAY BEKLİYOR: Net ve güven verici cevap ver",
                        "düşünerek_sorma": "🤔 DÜŞÜNCELI: Kullanıcı düşünüyor, detaylı açıkla",
                        "heyecanlı_soru": "🌟 HEYECANLI: Kullanıcı meraklı ve heyecanlı, enerjik anlat",
                        "samimi_veda": "👋 SAMİMİ VEDA: Dostça, sıcak vedalaş",
                        "samimi_tesekkur": "🙏 SAMİMİ TEŞEKKÜR: Samimi karşılık ver",
                        "samimi_selam": "😊 SAMİMİ SELAM: Arkadaşça, sıcak selamla",
                    }
                    talimat = kombinasyon_talimatlari.get(analiz.kombinasyon)
                    if talimat:
                        sohbet_talimati += f"\n• {talimat}"

                if analiz.onceki_konuyu_kapat:
                    sohbet_talimati += "\n• 🔄 KONU GEÇİŞİ: Önceki konudan bu konuya doğal geçiş yap, giriş cümlesi yapma, sohbet akıyormuş gibi devam et."

                # Espri/şaka kontrolü
                if hasattr(self, '_son_decision') and self._son_decision.get('is_espri'):
                    sohbet_talimati += "\n• 😄 ESPRİ MODU: şakacı gibi cevap ver! Ciddi açıklama YAPMA, kısa tut, eğlen."

                # Örtük istek varsa ekle
                if analiz.ortuk_istek:
                    sohbet_talimati += f"\n• 🎯 ÖRTÜK İSTEK: {analiz.ortuk_istek} (ima da olabilir - mesajın altındaki anlamı da düşün)"

            # 🔴 Dini soru ise özel kurallar ekle
            if tool_used == "risale_ara":
                # Cevap uzunluğu satırını kaldır
                sohbet_talimati = sohbet_talimati.replace(f"• Cevap uzunluğu: {min_uz}-{max_uz} karakter (AŞMA!)(AŞMA!)\n", "")
                sohbet_talimati += """
• 🔴 DİNİ KONULARDA:
  - Soruyu [📚 RİSALE-İ NUR BAŞLANGIÇ] ve [📚 RİSALE-İ NUR BİTİŞ] arasındaki bilgileri kullanarak cevapla
  - Risale metinleri çok zengin ve derin temsiller içeriyor, açıklamalarını bunlar üzerinden yap
  - ⛔ "Risale'de", "Sözler'de", "metinde" YAZMA - bilgiyi KENDİ sözünmüş gibi anlat
  - Vaaz değil sohbet tonu"""

        # Dini sorularda minimal prompt, diğerlerinde tam SYSTEM_PROMPT
        if tool_used == "risale_ara":
            system_content = f"""Sen akıllı, profesyonel, olgun ve sıcakkanlı bir yapay zekasın.
{user_info}
[⏰ ŞU AN]: {zaman['full']} ({zaman['zaman_dilimi']})
↳ Zaman farkındalığı.{context_info}{sohbet_talimati}"""
        else:
            system_content = f"""{self.SYSTEM_PROMPT}
{user_info}
[⏰ ŞU AN]: {zaman['full']} ({zaman['zaman_dilimi']})
↳ Zaman farkındalığı.{context_info}{sohbet_talimati}"""

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

        # 3. Son user message - sadece kullanıcının sorusu
        user_content = user_input
        messages.append({"role": "user", "content": user_content})

        # 4. Hesaplama sonucu varsa, system message'a ayrı bölüm olarak ekle (BAĞLAM'a değil!)
        if math_result:
            calc_value = math_result.replace('🧮 Hesaplama: ', '')
            math_instruction = f"\n\n[🧮 HESAPLAMA SONUCU]: {calc_value} ← Hesaplama aracın verdi, DOĞRU. Güvenle sun."
            # System message'ı güncelle
            messages[0]["content"] += math_instruction

        return messages

    async def prepare(self, user_input: str, chat_history: List[Dict] = None, firlama_modu: bool = False) -> Dict[str, Any]:
        """
        Prompt ve messages hazırla - LLM ÇAĞIRMA!

        Akış: Telegram → HafizaAsistani.prepare() → messages döner

        Args:
            firlama_modu: True ise sohbet zekası bypass edilir, enerjik mod aktif

        Returns:
            {
                "messages": [...],  # LLM için hazır messages
                "paket": {...}      # Metadata (tool_used, role vs.)
            }
        """
        chat_history = chat_history or []
        self._firlama_modu = firlama_modu  # Instance'a kaydet

        # 📝 NOT SİSTEMİ - Tetikleyici kontrolü
        not_result = self._check_not_tetikleyici(user_input)
        if not_result:
            # Not komutu algılandı, direkt cevap dön
            return {
                "messages": [
                    {"role": "system", "content": "Sen bir not asistanısın."},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": not_result}
                ],
                "paket": {"tool_used": "not_sistemi", "direct_response": not_result}
            }

        # 📍 KONUM SİSTEMİ - Konum sorgusu kontrolü
        # Konum sonuçları LLM'e context olarak gider, LLM doğal cevap verir
        konum_context = None
        if self.user_location:
            konum_result = await self._check_konum_sorgusu(user_input)
            if konum_result:
                # Belirsiz eşleşme - doğrulama butonu gösterilecek (UI gerekli)
                if isinstance(konum_result, dict) and konum_result.get("type") == "konum_dogrulama":
                    return {
                        "messages": [],
                        "paket": {"konum_dogrulama": konum_result}
                    }
                # Yakın yerler listesi - inline butonlarla gösterilecek (UI gerekli)
                if isinstance(konum_result, dict) and konum_result.get("type") == "yakin_yerler_listesi":
                    return {
                        "messages": [],
                        "paket": {"yakin_yerler": konum_result}
                    }
                # Normal sonuç (string) - LLM'e context olarak gönder
                konum_context = konum_result

            # 📍 KONUM GÖNDERME - Numara ile yer seçimi
            konum_gonder = self._check_konum_gonder_istegi(user_input)
            if konum_gonder:
                return {
                    "messages": [],
                    "paket": {"send_location": konum_gonder}
                }

        # 1. Paket hazırla (karar, tool, bağlam)
        paket = await self.hazirla_ve_prompt_olustur(user_input, chat_history)

        # 📍 Konum context varsa paket'e ekle (LLM görsün)
        if konum_context:
            paket["konum_context"] = konum_context

        # 2. Messages formatı oluştur
        messages = self._build_messages(user_input, paket, chat_history)

        return {
            "messages": messages,
            "paket": paket
        }

    def _check_not_tetikleyici(self, user_input: str) -> Optional[str]:
        """
        Not sistemi tetikleyicilerini kontrol et.

        Tetikleyiciler:
        - "not al: ...", "not al ...", "not al, ..."
        - "not tut: ...", "not tut ...", "not tut, ..."
        - "not ekle: ...", "not ekle ...", "not ekle, ..."
        - "notlarım", "notlarıma bak", "notlarımı göster"
        - "not sil #N", "N numaralı notu sil"

        Returns:
            str: Not işlemi sonucu veya None (tetikleyici yoksa)
        """
        user_lower = user_input.lower().strip()

        # 📝 PENDING MOD - Önceki "not al" sonrası bekleme
        if self._pending_not:
            self._pending_not = False
            # "iptal", "vazgeç" gibi kelimeler hariç her şeyi not al
            iptal_kelimeler = ["iptal", "vazgeç", "vazgec", "boşver", "bosver", "gerek yok", "tamam boşver"]
            if not any(k in user_lower for k in iptal_kelimeler):
                print(f"📝 Pending not kaydediliyor: '{user_input[:30]}...'")
                return self.not_manager.not_al(user_input)
            else:
                return "👍 Tamam, iptal ettim."

        # 📝 NOT AL / TUT / EKLE (içerikli)
        not_patterns = [
            (r'^not\s+al[\s:,]+(.+)$', 'not_al'),
            (r'^not\s+tut[\s:,]+(.+)$', 'not_al'),
            (r'^not\s+ekle[\s:,]+(.+)$', 'not_al'),
            (r'^şunu\s+not\s+(?:al|et)[\s:,]*(.+)$', 'not_al'),
            (r'^bunu\s+not\s+(?:al|et)[\s:,]*(.+)$', 'not_al'),
        ]

        for pattern, action in not_patterns:
            match = re.match(pattern, user_lower, re.IGNORECASE)
            if match:
                icerik = match.group(1).strip()
                if icerik:
                    print(f"📝 Not tetikleyici algılandı: {action} -> '{icerik[:30]}...'")
                    return self.not_manager.not_al(icerik)

        # 📝 NOT AL TEK BAŞINA - içerik olmadan → pending moda geç
        if re.match(r'^not\s+(al|tut|ekle)\s*[?!.,]*$', user_lower, re.IGNORECASE):
            print("📝 Not al (tek başına) algılandı - pending moda geçiliyor")
            self._pending_not = True
            return "📝 Tamam, ne not edeyim?"

        # 📋 NOTLARIMI GETİR
        notlar_patterns = [
            r'^notlar[ıi]m[ıi]?\s*(ne|neler|nedir)?[\s?]*$',
            r'^notlar[ıi]ma?\s+bak',
            r'^notlar[ıi]m[ıi]?\s+göster',
            r'^notlar[ıi]m[ıi]?\s+listele',
            r'^not(?:lar)?[ıi]m(?:da)?\s+ne(?:ler)?\s+va[er]',  # "neler var/vae" dahil
            r'^not(?:lar)?[ıi]m(?:da)?\s+ne(?:ler)?\s+vard[ıi]',  # "neler vardı"
            r'^notlar[ıi]m(?:da)?$',  # sadece "notlarımda"
        ]

        for pattern in notlar_patterns:
            if re.match(pattern, user_lower, re.IGNORECASE):
                print("📋 Notları getir tetikleyici algılandı")
                return self.not_manager.notlari_getir()

        # 🗑️ NOT SİL
        sil_patterns = [
            r'^(?:not\s+)?#?(\d+)\s*(?:numaral[ıi])?\s*not[ıi]?\s*sil',
            r'^not\s+sil\s+#?(\d+)',
            r'^#?(\d+)\s+not[ıi]?\s*sil',
        ]

        for pattern in sil_patterns:
            match = re.match(pattern, user_lower, re.IGNORECASE)
            if match:
                not_id = int(match.group(1))
                print(f"🗑️ Not sil tetikleyici algılandı: #{not_id}")
                return self.not_manager.not_sil(not_id)

        return None

    # ============================================================
    # 📍 KONUM SİSTEMİ
    # ============================================================

    def set_location(self, lat: float, lon: float, adres: str = None):
        """Kullanıcı konumunu kaydet"""
        self.user_location = (lat, lon)

        # Adresi parse et (TEK SEFER)
        self.konum_adres = ""
        if adres:
            parcalar = [p.strip() for p in adres.split(",")]
            if len(parcalar) >= 5:
                ilce = parcalar[-5]
                il = parcalar[-4]
                if len(parcalar) >= 7:
                    cadde = parcalar[-7]
                    mahalle = parcalar[-6]
                    self.konum_adres = f"{cadde}, {mahalle}, {ilce}, {il}"
                elif len(parcalar) >= 6:
                    mahalle = parcalar[-6]
                    self.konum_adres = f"{mahalle}, {ilce}, {il}"
                else:
                    self.konum_adres = f"{ilce}, {il}"
            else:
                self.konum_adres = adres[:50]

        print(f"📍 Konum kaydedildi: {lat:.4f}, {lon:.4f}")
        if self.konum_adres:
            print(f"   Adres: {self.konum_adres}")

    async def prepare_konum_alindi(self, lat: float, lon: float, adres: str) -> Dict[str, Any]:
        """Konum alındığında LLM için prompt hazırla"""
        self.set_location(lat, lon, adres)  # konum_adres burada oluşturuldu

        # Kullanıcı adını al
        kullanici_adi = ""
        if hasattr(self, 'profile_manager'):
            kullanici_adi = self.profile_manager.get_name() or ""

        # Sistem prompt'u - Ana SYSTEM_PROMPT + konum bilgisi
        system_content = f"""{self.SYSTEM_PROMPT}
Kullanıcı adı: {kullanici_adi}
📍 KONUM BİLGİSİ (SİSTEM TARAFINDAN ÇÖZÜMLENDI - KESİN BİLGİ):
- Konum: {self.konum_adres}

⚠️ ÖNEMLİ TALİMATLAR:
- Ciddi ve direkt bilgi ver
- Bu adres GPS'ten otomatik çözümlendi, %100 doğru
- Bu konumu sen biliyorsun, kendi bilgin gibi kesin ve emin sun
- "{self.konum_adres}'dasın." yaz, sonra kısaca ne istediğini sor
- Onay sorma, tereddüt gösterme
"""

        user_content = f"[Kullanıcı GPS konumunu paylaştı → Sistem çözümledi: {self.konum_adres}]"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        return {"messages": messages, "paket": {"tool_used": "konum_alindi"}}

    async def _check_konum_sorgusu(self, user_input: str) -> Optional[str]:
        """
        Konum bazlı sorguları kontrol et (fuzzy matching ile).
        Yakın yer, hava, namaz, kıble vs.
        """
        if not self.user_location:
            return None

        user_lower = user_input.lower().strip()
        lat, lon = self.user_location

        # Konum sinyalleri
        konum_sinyalleri = ["yakın", "yakin", "yakınım", "yakinim", "yakında", "yakinda",
                           "nerede", "neresi", "bul", "ara", "var mı", "varmı"]
        has_konum_signal = any(s in user_lower for s in konum_sinyalleri)

        # Kategori keywords
        kategori_map = {
            "eczane": ("pharmacy", "💊"),
            "benzinlik": ("fuel", "⛽"),
            "akaryakıt": ("fuel", "⛽"),
            "restoran": ("restaurant", "🍽️"),
            "lokanta": ("restaurant", "🍽️"),
            "kafe": ("cafe", "☕"),
            "kahve": ("cafe", "☕"),
            "atm": ("atm", "🏧"),
            "bankamatik": ("atm", "🏧"),
            "hastane": ("hospital", "🏥"),
            "acil": ("hospital", "🏥"),
            "market": ("supermarket", "🛒"),
            "süpermarket": ("supermarket", "🛒"),
            "cami": ("place_of_worship", "🕌"),
            "mescit": ("place_of_worship", "🕌"),
            "avm": ("mall", "🏬"),
            "alışveriş merkezi": ("mall", "🏬"),
            "otopark": ("parking", "🅿️"),
            "park yeri": ("parking", "🅿️"),
            "otel": ("hotel", "🏨"),
            "okul": ("school", "🏫"),
            "lise": ("school", "🏫"),
            "üniversite": ("university", "🎓"),
            "istasyon": ("station", "🚉"),
            "metro": ("station", "🚉"),
            "tren": ("station", "🚉"),
            "bakkal": ("convenience", "🏪"),
        }
        kategori_keywords = list(kategori_map.keys())

        # Fuzzy matching ile kategori bul (yazım hatası toleranslı)
        from difflib import SequenceMatcher
        words = re.findall(r'\b\w+\b', user_lower)
        for word in words:
            if len(word) >= 4:  # Minimum 4 karakter
                # En iyi eşleşmeyi ve skorunu bul
                best_match = None
                best_score = 0
                for keyword in kategori_keywords:
                    score = SequenceMatcher(None, word, keyword).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = keyword

                # Yüksek eşleşme (skor >= 0.90) → direkt arama
                if best_score >= 0.90 and best_match:
                    print(f"📍 Yakın yer sorgusu (kesin): '{word}' → '{best_match}' (skor: {best_score:.2f})")
                    return await self._get_yakin_yerler(lat, lon, best_match)

                # Orta eşleşme (0.75 <= skor < 0.90) → doğrulama sor
                elif best_score >= 0.75 and best_match:
                    print(f"📍 Belirsiz eşleşme: '{word}' → '{best_match}' (skor: {best_score:.2f})")
                    return {
                        "type": "konum_dogrulama",
                        "yazilan": word,
                        "kategori": best_match,
                        "mesaj": f"🤔 '{word}' derken '{best_match}' mi demek istedin?"
                    }

        # Exact match (tam kelime eşleşmesi)
        for keyword in kategori_keywords:
            if keyword in user_lower:
                print(f"📍 Yakın yer sorgusu (exact): {keyword}")
                return await self._get_yakin_yerler(lat, lon, keyword)

        return None

    async def _get_yakin_yerler(self, lat: float, lon: float, kategori: str) -> str:
        """OpenStreetMap Overpass API ile yakın yerleri bul"""
        kategori_map = {
            "eczane": ("pharmacy", "💊"),
            "benzinlik": ("fuel", "⛽"),
            "akaryakıt": ("fuel", "⛽"),
            "restoran": ("restaurant", "🍽️"),
            "lokanta": ("restaurant", "🍽️"),
            "kafe": ("cafe", "☕"),
            "kahve": ("cafe", "☕"),
            "atm": ("atm", "🏧"),
            "bankamatik": ("atm", "🏧"),
            "hastane": ("hospital", "🏥"),
            "acil": ("hospital", "🏥"),
            "market": ("supermarket", "🛒"),
            "süpermarket": ("supermarket", "🛒"),
            "cami": ("place_of_worship", "🕌"),
            "mescit": ("place_of_worship", "🕌"),
            "avm": ("mall", "🏬"),
            "alışveriş merkezi": ("mall", "🏬"),
            "otopark": ("parking", "🅿️"),
            "park yeri": ("parking", "🅿️"),
            "otel": ("hotel", "🏨"),
            "okul": ("school", "🏫"),
            "lise": ("school", "🏫"),
            "üniversite": ("university", "🎓"),
            "istasyon": ("station", "🚉"),
            "metro": ("station", "🚉"),
            "tren": ("station", "🚉"),
            "bakkal": ("convenience", "🏪"),
        }

        if kategori not in kategori_map:
            return None

        osm_tag, emoji = kategori_map[kategori]

        # Overpass API sorgusu
        overpass_url = "https://overpass-api.de/api/interpreter"
        radius = 10000  # 10km

        if osm_tag == "place_of_worship":
            query = f"""
            [out:json][timeout:10];
            (
              node["amenity"="{osm_tag}"]["religion"="muslim"](around:{radius},{lat},{lon});
              way["amenity"="{osm_tag}"]["religion"="muslim"](around:{radius},{lat},{lon});
            );
            out center 10;
            """
        else:
            query = f"""
            [out:json][timeout:10];
            (
              node["amenity"="{osm_tag}"](around:{radius},{lat},{lon});
              way["amenity"="{osm_tag}"](around:{radius},{lat},{lon});
            );
            out center 10;
            """

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(overpass_url, data={"data": query}) as resp:
                    if resp.status != 200:
                        return f"❌ Yakın {kategori} araması başarısız oldu."
                    data = await resp.json()

            elements = data.get("elements", [])
            if not elements:
                return f"📍 {radius//1000}km içinde {kategori} bulunamadı."

            # Mesafe hesapla ve sırala
            import math
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000  # metre
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
                return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

            yerler = []
            for el in elements:
                el_lat = el.get("lat") or el.get("center", {}).get("lat")
                el_lon = el.get("lon") or el.get("center", {}).get("lon")
                if el_lat and el_lon:
                    mesafe = haversine(lat, lon, el_lat, el_lon)
                    ad = el.get("tags", {}).get("name", f"{kategori.title()} #{len(yerler)+1}")
                    yerler.append({
                        "ad": ad,
                        "mesafe": int(mesafe),
                        "lat": el_lat,
                        "lon": el_lon
                    })

            yerler.sort(key=lambda x: x["mesafe"])
            yerler = yerler[:5]  # İlk 5

            # Sonuçları kaydet (konum gönderme için)
            self.son_yakin_yerler = yerler

            # Inline butonlu format döndür
            return {
                "type": "yakin_yerler_listesi",
                "kategori": kategori,
                "emoji": emoji,
                "yerler": yerler
            }

        except Exception as e:
            print(f"❌ Overpass API hatası: {e}")
            return f"❌ Yakın {kategori} araması sırasında hata oluştu."

    def _check_konum_gonder_istegi(self, user_input: str) -> Optional[Dict]:
        """
        Kullanıcının konum gönderme isteğini kontrol et.
        "1", "2 numaralı yerin konumunu gönder" vs.
        """
        if not self.son_yakin_yerler:
            return None

        user_lower = user_input.lower().strip()

        # Numara çıkar
        sira = None
        match = re.search(r'(\d+)', user_lower)
        if match:
            sira = int(match.group(1))

        if sira:
            # Sadece sayı yazılmış mı? ("1", "2", vs.)
            sadece_sayi = user_lower.strip() in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            # Veya konum isteği keyword'ü var mı?
            konum_keywords = ['konum', 'gönder', 'göster', 'nerede', 'git', 'yol']
            konum_istegi = any(kw in user_lower for kw in konum_keywords)

            if sadece_sayi or konum_istegi:
                yer = self.get_yakin_yer_konumu(sira)
                if yer:
                    return yer

        return None

    def get_yakin_yer_konumu(self, sira: int) -> Optional[Dict]:
        """Sıra numarasına göre yakın yer koordinatlarını döndür"""
        if not self.son_yakin_yerler:
            return None

        if 1 <= sira <= len(self.son_yakin_yerler):
            yer = self.son_yakin_yerler[sira - 1]
            return {
                "lat": yer["lat"],
                "lon": yer["lon"],
                "ad": yer["ad"],
                "mesafe": yer["mesafe"]
            }
        return None

    def save(self, user_input: str, response: str, chat_history: List[Dict] = None):
        """
        Cevabı hafızaya kaydet

        Akış: PersonalAI cevap verdi → HafizaAsistani.save() → hafızaya kaydet
        """
        # Hata mesajlarını kaydetme (Telegram'a gider ama history'e eklenmedi)
        if response.startswith("[HATA]"):
            print("   ⚠️ Hata mesajı - history'e eklenmedi")
            return

        self.add(user_input, response, chat_history or [])
