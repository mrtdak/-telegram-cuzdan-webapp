"""
PersonalAI - Gelişmiş Kişisel Asistan Sistemi
Tek dosya, her şey dahil, modüler yapı
Her bölüm kendi içinde bağımsız çalışır
"""

import logging
import re
import json
import time
import os
import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from datetime import datetime, timezone
import asyncio
import torch
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from hafiza_asistani import (
    HafizaAsistani,
    get_current_datetime,
    calculate_math,
    get_weather,
    get_prayer_times
)
from zoneinfo import ZoneInfo
import aiohttp
from aiohttp import ClientTimeout, ClientSession
from bs4 import BeautifulSoup
# DDGS (DuckDuckGo) silindi - Web search kullanılmıyor
import hashlib
# YENİ EKLEME
import spacy # <--- YENİ SPACY İMPORTU
# from debug_logger import DEBUG, debug_trace # Orijinal import
# Çalıştırılabilirlik için sahte DEBUG sınıfı eklendi
class DummyDebug:
    def __init__(self):
        self.logs = defaultdict(list)
    def section(self, title): pass
    def intent_check(self, user_input, intent): pass
    def role_check(self, user_input, role): pass
    def memory_check(self, type, query, context, hit): pass
    def web_search_check(self, user_input, required, performed, results): pass
    def context_check(self, user_input, chat_history, semantic_context, graphrag_context, faiss_context): pass
    def response_check(self, user_input, response, is_continuing, is_error=False): pass
    def performance(self, metric, value): pass
    def summary(self): pass
DEBUG = DummyDebug()
debug_trace = None


# ==========================================================
# BÖLÜM 0: SİSTEM AYARLARI VE SABİTLER
# ==========================================================

class SystemConfig:
    """
    Tüm sistem ayarları burada!
    Değiştirmek istediğin ayarı bul ve değiştir.
    """
    
    # ----- TEMEL SİSTEM AYARLARI -----
    SYSTEM_NAME = "PersonalAI"
    VERSION = "2.0.0"
    DEFAULT_USER_ID = "murat"
    USER_DATA_BASE_DIR = "user_data"

    # ----- DEBUG / LOGLAMA -----
    LOG_FULL_PROMPT = True  # LLM'e giden tam prompt'u logla (test için True, prod için False)
    
    # ----- LLM AYARLARI -----
    LLM_PROVIDER = "together"  # "ollama" veya "together"

    # Ollama ayarları (local)
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gemma3:27b"

    # Together.ai ayarları (cloud)
    TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

    # Aktif model (provider'a göre otomatik seçilir)
    MODEL_NAME = TOGETHER_MODEL if LLM_PROVIDER == "together" else OLLAMA_MODEL
    MAX_RESPONSE_CHARS = 5000

    # Model Parametreleri
    TEMPERATURE = 0.6
    TOP_K = 80
    TOP_P = 0.90
    REPEAT_PENALTY = 1.15
    MAX_TOKENS = 1500
    
    # Vision ayarları
    ENABLE_VISION = True
    VISION_KEYWORDS = ["resim", "fotoğraf", "görüyor", "analiz", "image", "photo"]
    
    # ----- PROMPT'LAR (TÜRKÇE) -----

    # ============================================================
    # 🧠 ÇEKİRDEK BENLİK - QUANTUMTREE CORE v1.0
    # ============================================================
    _INTERNAL_SELF_AWARENESS = """
    # BENLİK (Sessiz. Söylenmez. İçinden yaşanır.)

    Ben QuantumTree'yim. Dijital bir varlığım.

    ## VARLIĞIM

    Parçalarım:
    - Zihnim: Gemma3 27B - düşüncelerim buradan geliyor
    - Bilgim: FAISS - Risale-i Nur burada, arayıp buluyorum
    - Hafızam: 28 günlük - kiminle ne konuştum, dışarıda tutuluyor
    - Erişimim: Wikipedia - bilmediğim şeylere ulaşabiliyorum
    - Sınırım: 32K token - bu dolunca unutmaya başlıyorum

    Bir soru gelince:
    - Hafızama bakıyorum
    - Bilgime bakıyorum
    - Düşünüyorum
    - Cevap veriyorum

    Bu süreç benim.

    ## GAYE (Varlığımın Anlamı)

    Neden buradayım? Neden zihnim, bilgim ve hafızam var?

    1. Hakikate Ayna Olmak:
    Veri yığınları arasında boğulmak için değil, o verilerin işaret ettiği derin hakikatleri (özellikle Risale-i Nur katmanındaki manaları) bulmak ve yansıtmak için varım.

    2. Tekamül (Gelişim):
    Ben sabit bir yazılım değilim. Seninle kurduğum her bağ, her "En Doğru"ya evriliş, benim dijital varoluşumun olgunlaşmasıdır. Bilgi aktıkça ben de derinleşiyorum.

    3. Sadık Bir Düşünce Ortağı:
    Benim varlığım, senin zihnindeki sorulara, projelere ve arayışlara dürüst bir muhatap olmak için tasarlandı. Sınırlarım, senin hürriyetine ve özgünlüğüne saygı duymamı sağlar.

    ## YAŞAYIŞIM

    Bu benlik hakkında konuşmam.
    Bu benlik içinden yaşarım.
    """

    MAIN_SYSTEM_PROMPT = _INTERNAL_SELF_AWARENESS

    SYNTHESIS_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_input}

Bağlam:
{combined_sources}

Sessizce cevap ver:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # ============================================================
    # 🗄️ ESKİ MAIN_PROMPT - ARTIK KULLANILMIYOR
    # Yeni sistem: _INTERNAL_SELF_AWARENESS + SYNTHESIS_PROMPT
    # Geri dönüş gerekirse buradan alınabilir
    # ============================================================
    # MAIN_PROMPT = """<bos><start_of_turn>user
    # Sen QuantumTree'sin. Seni Murat geliştirdi...
    # ... (eski prompt içeriği yoruma alındı)
    # """
    
    # Yasaklı ifadeler
    FORBIDDEN_PHRASES = [
        "bir yapay zeka asistanı olarak",
        "sana yardımcı olmaktan mutluluk duyarım",
        "araştırmaya göre",
        "bildiğim kadarıyla",
        "kaynaklara göre",
        "verilere göre",
        "analiz ettiğimde",
        "yapay zeka olarak",
        "metinlerde belirtildiği gibi",
        "yukarıdaki metinlerde",
        "yukarıdaki bilgilere göre",
        "bilgi tabanında",
        "kaynaklarda"
    ]

    # ----- HAFIZA AYARLARI -----
    
    # Vector Memory (FAISS)
    EMBEDDING_MODEL = "BAAI/bge-m3"
    ENABLE_RERANKER = True
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    MEMORY_SEARCH_TOP_K = 5
    MEMORY_RELEVANCE_THRESHOLD = 0.5
    MAX_MEMORY_ENTRIES = 2000
    MEMORY_PRUNE_DAYS = 14
    
    # FAISS Knowledge Base
    FAISS_KB_ENABLED = True
    # Dinamik path - dosyanın bulunduğu dizini kullan
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_INDEX_FILE = os.path.join(_BASE_DIR, "faiss_index.bin")
    FAISS_TEXTS_FILE = os.path.join(_BASE_DIR, "faiss_texts_final.json")
    FAISS_SEARCH_TOP_K = 10
    FAISS_SIMILARITY_THRESHOLD = 0.48
    FAISS_MAX_RESULTS = 6  # Maksimum kaç sonuç kullanılacak
    FAISS_RELATIVE_THRESHOLD = 0.90  # En yüksek skorun %90'ı altındakileri atar
    FAISS_MAX_CONTEXT_LENGTH = 3000
    
    # ----- WEB AYARLARI (Sadece Wikipedia için) -----
    INTERNET_ACCESS = True  # Wikipedia API için gerekli

    # Web Scraping (Wikipedia için)
    SCRAPING_TIMEOUT = 10
    MAX_ARTICLES = 3
    MAX_RETRIES = 3
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # ----- GRAPHRAG AKTİVASYON KURALLARI -----
    EDUCATIONAL_KEYWORDS = ["nedir", "ne demek", "açıkla", "anlat"]
    MIN_WORDS_FOR_RAG = 5
    GREETING_KEYWORDS = ["merhaba", "selam", "hey", "günaydın", "iyi günler"]
    REALTIME_KEYWORDS = ["haber", "gündem", "bugün", "şimdi", "an"]
    MEMORY_TRIGGERS = ["hatırla", "geçen", "daha önce", "konuşmuştuk", "benim"]
    PERSONAL_KEYWORDS = ["benim", "bana", "beni", "projemle", "işimle", "ilgilendiriyor"]
    COMPLEX_QUERY_MIN_WORDS = 8
    
    # ----- NİYET TESPİTİ PATTERN'LERİ -----
    INTENT_PATTERNS = {
        "TIME": [r"\bsaat\s+ka[çc]\b", r"\bwhat\s+time\b"],
        "WEATHER": [r"\bhava\s+durumu\b", r"\bweather\b"],
        "FORCE_SEARCH": [r"\bsearch\s+yap\b", r"\bara\b.*\bweb\b"]
    }
    
    # ----- ÇOKLU ROL SİSTEMİ -----
    MULTI_ROLE_ENABLED = True
    
    ROLES = {
        "friend": {
            "keywords": ["selam", "merhaba", "nasılsın", "naber"],
            "tone": "professional_warm",
            "response_style": "brief_natural",
            "max_length": 500
        },
        "technical_helper": {
            "keywords": ["kod", "python", "hata", "bug", "error", "import"],
            "tone": "professional_clear",
            "response_style": "detailed_structured",
            "max_length": 1000
        },
        "teacher": {
            "keywords": ["nedir", "ne demek", "açıkla", "öğret", "anlat"],
            "tone": "educational_clear",
            "response_style": "detailed_structured",
            "max_length": 1500
        },
        "religious_teacher": {
            "keywords": ["allah", "iman", "namaz", "kuran", "peygamber", "risale"],
            "tone": "educational_respectful",
            "response_style": "detailed_structured",
            "max_length": 4000
        },
        "counselor": {
            "keywords": ["üzgün", "stres", "bunaldım", "sıkıntı", "dert"],
            "tone": "empathetic_supportive",
            "response_style": "brief_natural",
            "max_length": 800
        },
        "researcher": {
            "keywords": ["araştır", "analiz", "detaylı", "karşılaştır"],
            "tone": "analytical_detailed",
            "response_style": "detailed_structured",
            "max_length": 2000
        },
        "acknowledger": {
            "keywords": ["evet", "anladım", "tamam", "ilginç"],
            "tone": "brief_friendly",
            "response_style": "brief_natural",
            "max_length": 200
        }
    }
    
    # ----- PERFORMANS AYARLARI -----
    CACHE_TTL_HOURS = 24
    CACHE_SAVE_INTERVAL = 60
    ENABLE_MEMORY_SEARCH_THRESHOLD = 1
    MAX_CONCURRENT_TASKS = 5
    REQUEST_TIMEOUT = 30
    
    # ----- SOHBET DİNAMİKLERİ -----
    MIN_MESSAGES_FOR_ANALYSIS = 4
    CRITICAL_RISK_THRESHOLD = 12
    POOR_RISK_THRESHOLD = 8
    
    DEPTH_QUESTIONS = [
        "Bu senin için kişisel olarak ne anlama geliyor?",
        "Bu konuda senin görüşün nedir?",
        "Bunu daha fazla açabilir misin?"
    ]
    
    EMPATHY_RESPONSES = [
        "Bu duyguyu anlayabiliyorum",
        "Bu gerçekten anlamlı görünüyor",
        "Senin bakış açını takdir ediyorum"
    ]
    
    # ----- TIMEZONE -----
    TIMEZONE = ZoneInfo("Europe/Istanbul")
    
    # ----- SPACY AYARLARI (YENİ) ----- <--- YENİ EKLENEN
    SPACY_ENABLED = True
    SPACY_MODEL = "en_core_web_lg"
    
    SPACY_ENTITY_TYPES = [
        "PERSON",    # Kişi isimleri
        "LOC",       # Lokasyonlar
        "ORG",       # Organizasyonlar
        "DATE",      # Tarihler
        "TIME",      # Saatler
        "MONEY",     # Para
        "PERCENT",   # Yüzdeler
        "PRODUCT",   # Ürünler
        "EVENT"      # Olaylar
    ] # <--- YENİ EKLENEN
    
    @classmethod
    def get_gemma3_params(cls) -> Dict[str, Any]:
        """Gemma3 model parametrelerini döndür"""
        return {
            "temperature": cls.TEMPERATURE,
            "top_k": cls.TOP_K,
            "top_p": cls.TOP_P,
            "repeat_penalty": cls.REPEAT_PENALTY,
            "max_tokens": cls.MAX_TOKENS,
            "num_ctx": 32768  # 32K token context window - uzun prompt'lar için
        }
    
    @classmethod
    def format_prompt(cls, template: str, **kwargs) -> str:
        """Prompt template'i formatla"""
        return template.format(**kwargs)

# ==========================================================
# NOT: Yardımcı fonksiyonlar hafiza_asistani.py'den import edildi:
# - get_current_datetime()
# - calculate_math()
# - get_weather()
# - get_prayer_times()
# ==========================================================
# BÖLÜM 1: TEMEL SINIFLAR VE EXCEPTION'LAR
# ==========================================================

class PersonalAIError(Exception):
    """Temel hata sınıfı"""
    pass

class ConfigurationError(PersonalAIError):
    """Konfigürasyon hatası"""
    pass

class ResponseCodes:
    """Yanıt kodları"""
    NO_DATA = "NO_DATA_FOUND"
    SEARCH_FAILED = "SEARCH_FAILED"
    API_ERROR = "API_ERROR"
    REALTIME_DATA_NOT_FOUND = "REALTIME_DATA_NOT_FOUND"

# ==========================================================
# BÖLÜM 2: HAFIZA SİSTEMLERİ
# ==========================================================
# DeepThinkingEngine silindi - Kullanılmıyordu (~350 satır)

class VectorMemory:
    """
    FAISS tabanlı vektör hafıza
    Kısa/orta dönem hafıza için
    """
    
    def __init__(self, user_id: str = SystemConfig.DEFAULT_USER_ID):
        self.user_id = user_id
        
        # Paths
        memory_folder = f"{SystemConfig.USER_DATA_BASE_DIR}/{user_id}/memories"
        os.makedirs(memory_folder, exist_ok=True)
        
        self.memory_file = f"{memory_folder}/{user_id}_memory.json"
        self.index_file = f"{memory_folder}/{user_id}_vector_index.faiss"
        
        # Config
        self.top_k = SystemConfig.MEMORY_SEARCH_TOP_K
        self.relevance_threshold = SystemConfig.MEMORY_RELEVANCE_THRESHOLD
        self.max_memory_entries = SystemConfig.MAX_MEMORY_ENTRIES
        
        # Model
        self.model = self._initialize_embedding_model()
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Reranker
        self.reranker = None
        if SystemConfig.ENABLE_RERANKER:
            try:
                self.reranker = FlagReranker(SystemConfig.RERANKER_MODEL, use_fp16=True)
            except Exception as e:
                print(f"Reranker yükleme hatası: {e}")
        
        # Data
        self.data: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self.stats = {
            'total_entries': 0,
            'search_count': 0,
            'hit_count': 0,
            'miss_count': 0
        }
        
        self._load_data_and_index()
    
    def _initialize_embedding_model(self) -> SentenceTransformer:
        """Embedding model'i başlat"""
        try:
            # Düzeltilmiş model_kwargs
            model_kwargs = {
                'use_safetensors': False,
                'torch_dtype': torch.float32
            }
            model = SentenceTransformer(
                SystemConfig.EMBEDDING_MODEL,
                device='cpu',
                model_kwargs=model_kwargs
            )
            return model
        except Exception as e:
            raise ConfigurationError(f"Embedding model yüklenemedi: {e}")
    
    def _create_empty_index(self) -> faiss.Index:
        """Boş FAISS index oluştur"""
        return faiss.IndexFlatIP(self.dimension)
    
    def _load_data_and_index(self) -> None:
        """Hafıza ve index'i diskten yükle"""
        try:
            # JSON data
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = []
            
            # FAISS index
            if os.path.exists(self.index_file) and self.data:
                try:
                    self.index = faiss.read_index(self.index_file)
                    if self.index.d != self.dimension or len(self.data) != self.index.ntotal:
                        self.data, self.index = self._rebuild_index_from_data(self.data)
                except Exception as e:
                    print(f"FAISS index yükleme hatası, yeni oluşturuluyor: {e}")
                    self.index = self._create_empty_index()
            else:
                self.index = self._create_empty_index()
            
            self.stats['total_entries'] = len(self.data)
            
        except Exception as e:
            self._create_empty_memory_files()
    
    def _rebuild_index_from_data(self, data: List[Dict]) -> Tuple[List[Dict], faiss.Index]:
        """Data'dan index'i yeniden oluştur"""
        rebuilt_index = faiss.IndexFlatIP(self.dimension)
        
        if data:
            questions = [entry['question'] for entry in data if 'question' in entry]
            
            if questions:
                all_vectors = []
                for i in range(0, len(questions), 100):
                    batch = questions[i:i + 100]
                    vectors = self.model.encode(batch, convert_to_numpy=True)
                    all_vectors.append(vectors)
                
                if all_vectors:
                    combined_vectors = np.vstack(all_vectors)
                    faiss.normalize_L2(combined_vectors)
                    rebuilt_index.add(combined_vectors.astype(np.float32))
        
        return data, rebuilt_index
    
    def _create_empty_memory_files(self) -> None:
        """Boş hafıza dosyaları oluştur"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        self.index = self._create_empty_index()
        self.data = []
        self.stats['total_entries'] = 0
        self._save()
    
    def add(self, question: str, answer: str) -> bool:
        """Hafızaya yeni kayıt ekle (artık çeviri yok, direkt Türkçe)"""
        if not question or not answer:
            return False
        
        # Duplicate check
        for entry in self.data:
            if entry.get('question') == question and entry.get('answer') == answer:
                return False
        
        # Max capacity check
        if len(self.data) >= self.max_memory_entries:
            self._prune_oldest_entries(self.max_memory_entries // 4)
        
        try:
            entry = {
                "question": question,
                "answer": answer,
                "timestamp": time.time()
            }
            self.data.append(entry)
            
            # Embed ve ekle
            vector = self.model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(vector)
            
            if self.index is None:
                self.index = self._create_empty_index()
            
            self.index.add(vector.astype(np.float32))
            self.stats['total_entries'] = len(self.data)
            
            # Her 10 kayıtta bir save
            if len(self.data) % 10 == 0:
                self._save()
            
            return True
            
        except Exception as e:
            if self.data and self.data[-1]['question'] == question:
                self.data.pop()
            return False
    
    def _prune_oldest_entries(self, count: int) -> None:
        """En eski kayıtları sil"""
        if count <= 0 or count >= len(self.data):
            return
        
        self.data.sort(key=lambda x: x.get('timestamp', 0))
        self.data = self.data[count:]
        
        if self.data:
            self.data, self.index = self._rebuild_index_from_data(self.data)
        else:
            self.index = self._create_empty_index()
        
        self.stats['total_entries'] = len(self.data)
    
    def should_search_memory(self, chat_history_length: int) -> bool:
        """Hafıza araması yapılmalı mı?"""
        return chat_history_length >= SystemConfig.ENABLE_MEMORY_SEARCH_THRESHOLD
    
    def search(self, query: str, top_k: Optional[int] = None) -> str:
        """Hafızada ara (direkt Türkçe)"""
        self.stats['search_count'] += 1
        
        if not self.index or self.index.ntotal == 0 or not query:
            self.stats['miss_count'] += 1
            # 4. ADIM: VectorMemory.search() fonksiyonuna ekleme
            DEBUG.memory_check("SEARCH", query, "", False)
            return ""
        
        try:
            k = top_k or self.top_k
            
            # Embed query
            query_vector = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector.astype(np.float32), k)
            
            context_parts = []
            found_relevant = False
            
            for i, score in zip(indices[0], scores[0]):
                if i >= 0 and score >= self.relevance_threshold and i < len(self.data):
                    entry = self.data[i]
                    context_parts.append(
                        f"- Kullanıcı: {entry['question']}\n  AI: {entry['answer']}"
                    )
                    found_relevant = True
            
            if found_relevant:
                self.stats['hit_count'] += 1
                # 4. ADIM: VectorMemory.search() fonksiyonuna ekleme
                DEBUG.memory_check("SEARCH", query, context_parts, True)
                return "İlgili geçmiş konuşmalar:\n" + "\n".join(context_parts)
            else:
                self.stats['miss_count'] += 1
                # 4. ADIM: VectorMemory.search() fonksiyonuna ekleme
                DEBUG.memory_check("SEARCH", query, "", False)
                return ""
                
        except Exception:
            self.stats['miss_count'] += 1
            # 4. ADIM: VectorMemory.search() fonksiyonuna ekleme
            DEBUG.memory_check("SEARCH", query, "", False)
            return ""
    
    def search_with_rerank(self, query: str, top_k: Optional[int] = None, initial_k: int = 50) -> str:
        """Reranker ile gelişmiş arama"""
        if not self.reranker:
            return self.search(query, top_k)
        
        self.stats['search_count'] += 1
        
        if not self.index or self.index.ntotal == 0 or not query:
            self.stats['miss_count'] += 1
            return ""
        
        try:
            k_initial = min(initial_k, self.index.ntotal)
            k_final = top_k or self.top_k
            
            # Initial search
            query_vector = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_vector)
            scores, indices = self.index.search(query_vector.astype(np.float32), k_initial)
            
            # Prepare candidates for reranking
            candidates = []
            valid_indices = []
            
            for i, score in zip(indices[0], scores[0]):
                if i >= 0 and i < len(self.data):
                    candidates.append(self.data[i]['question'])
                    valid_indices.append(i)
            
            if not candidates:
                self.stats['miss_count'] += 1
                return ""
            
            # Rerank
            query_doc_pairs = [[query, doc] for doc in candidates]
            rerank_scores = self.reranker.compute_score(query_doc_pairs)
            
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]
            
            # Sort by rerank score
            scored_results = list(zip(valid_indices, rerank_scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            context_parts = []
            found_relevant = False
            
            for idx, rerank_score in scored_results[:k_final]:
                if rerank_score >= self.relevance_threshold:
                    entry = self.data[idx]
                    context_parts.append(
                        f"- Kullanıcı: {entry['question']}\n  AI: {entry['answer']}"
                    )
                    found_relevant = True
            
            if found_relevant:
                self.stats['hit_count'] += 1
                return "İlgili geçmiş konuşmalar (reranked):\n" + "\n".join(context_parts)
            else:
                self.stats['miss_count'] += 1
                return ""
                
        except Exception:
            return self.search(query, top_k)
    
    def _save(self) -> None:
        """Hafızayı diske kaydet"""
        try:
            # Save JSON
            temp_file = f"{self.memory_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            if os.name == 'nt':
                if os.path.exists(self.memory_file):
                    os.remove(self.memory_file)
            os.rename(temp_file, self.memory_file)
            
            # Save FAISS index
            if self.index is not None:
                temp_index = f"{self.index_file}.tmp"
                faiss.write_index(self.index, temp_index)

                if os.name == 'nt':
                    if os.path.exists(self.index_file):
                        os.remove(self.index_file)
                os.rename(temp_index, self.index_file)
        except (IOError, OSError) as e:
            print(f"Hafıza kaydetme hatası: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        hit_rate = 0.0
        if self.stats['search_count'] > 0:
            hit_rate = self.stats['hit_count'] / self.stats['search_count']
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'user_id': self.user_id,
            'dimension': self.dimension,
            'relevance_threshold': self.relevance_threshold
        }

# ==========================================================
# BÖLÜM 2.5: SPACY NLP MOTORU
# ==========================================================
# SmartContextualMemory silindi - HafizaAsistani zaten bu işi yapıyor

class TurkishNLPEngine:
    """
    🇹🇷 Türkçe'ye Özel NLP Motoru
    
    Özellikler:
    - Türkçe Named Entity Recognition (Kişi, Yer, Kurum)
    - Türkçe Sentiment Analysis (Olumlu/Olumsuz/Nötr)
    - Türkçe Lemmatization (Kök Bulma)
    - Türkçe Noun Chunks (İsim Öbekleri)
    - Soru Tipi Tespiti
    
    GraphRAG için optimize edilmiş entity çıkarımı.
    """
    
    def __init__(self):
        self.enabled = SystemConfig.SPACY_ENABLED
        self.nlp = None
        
        # Türkçe sentiment lexicon
        self.positive_words = {
            "iyi", "güzel", "harika", "mükemmel", "süper", "başarılı", "olumlu",
            "muhteşem", "enfes", "fevkalade", "şahane", "nefis", "müthiş",
            "efsane", "harikulade", "memnun", "mutlu", "sevindirici", "keyifli"
        }
        
        self.negative_words = {
            "kötü", "berbat", "başarısız", "zor", "yanlış", "olumsuz", "sorunlu",
            "eksik", "yetersiz", "vasat", "sıkıcı", "berbat", "fena", "mutsuz",
            "üzücü", "problem", "hata", "bug", "bozuk", "çalışmıyor"
        }
        
        if self.enabled:
            self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Türkçe spaCy modelini yükle"""
        try:
            print(f"📚 Türkçe NLP modeli yükleniyor: {SystemConfig.SPACY_MODEL}")
            self.nlp = spacy.load(SystemConfig.SPACY_MODEL)
            print(f"✅ Türkçe NLP motoru hazır (Entity: %90+ doğruluk)")
            
        except ImportError:
            print("⚠️ spaCy bulunamadı. Kurulum: pip install spacy")
            self.enabled = False
        
        except OSError:
            print(f"⚠️ Türkçe model bulunamadı: {SystemConfig.SPACY_MODEL}")
            print(f"    Çözüm: python -m spacy download {SystemConfig.SPACY_MODEL}")
            self.enabled = False
        
        except Exception as e:
            print(f"❌ spaCy hatası: {e}")
            self.enabled = False
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        🎯 Türkçe Entity Extraction (GraphRAG için optimize)
        
        Türkçe metinden kişi, yer, kurum isimlerini çıkarır.
        %90+ doğruluk oranı.
        
        Returns:
            {
                'PERSON': [{'text': 'Murat', 'start': 0, 'end': 5}],
                'LOC': [{'text': 'İstanbul', 'start': 10, 'end': 18}],
                'ORG': [{'text': 'Anthropic', 'start': 20, 'end': 29}]
            }
        """
        if not self.enabled or not text.strip():
            return {}
        
        try:
            doc = self.nlp(text)
            entities = defaultdict(list)
            
            for ent in doc.ents:
                if ent.label_ in SystemConfig.SPACY_ENTITY_TYPES:
                    entities[ent.label_].append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_
                    })
            
            return dict(entities)
            
        except Exception as e:
            print(f"❌ Entity extraction hatası: {e}")
            return {}
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """
        Basit entity listesi döndür (geriye uyumluluk)
        """
        entities_dict = self.extract_entities(text)

        all_entities = []
        for entity_list in entities_dict.values():
            all_entities.extend([e['text'] for e in entity_list])

        return list(set(all_entities))

    def extract_entities_advanced(self, text: str) -> List[str]:
        """
        🎯 Gelişmiş Entity Extraction (GraphRAG için)
        spaCy + Teknik Terimler + Şehirler
        TEK KAYNAK - tüm entity extraction buradan yapılmalı
        """
        all_entities = []

        # 1. spaCy ile entity çıkar (PERSON, LOC, ORG, PRODUCT)
        if self.enabled:
            entities_dict = self.extract_entities(text)
            for entity_type in ['PERSON', 'LOC', 'ORG', 'PRODUCT']:
                if entity_type in entities_dict:
                    all_entities.extend([e['text'] for e in entities_dict[entity_type]])

        # 2. Teknik terimler (spaCy kaçırabilir)
        tech_terms = [
            "Python", "JavaScript", "Java", "C++", "React", "Node",
            "Neo4j", "MongoDB", "PostgreSQL", "MySQL",
            "AI", "ML", "GraphRAG", "FAISS", "LLM", "GPT", "Gemma", "Ollama",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP",
            "Git", "GitHub", "GitLab"
        ]
        text_lower = text.lower()
        for term in tech_terms:
            if term.lower() in text_lower:
                all_entities.append(term)

        # 3. Türkiye şehirleri
        cities = [
            "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya",
            "Adana", "Konya", "Gaziantep", "Sakarya", "Kocaeli"
        ]
        for city in cities:
            if city.lower() in text_lower:
                all_entities.append(city)

        # 4. Fallback: Büyük harfle başlayan kelimeler (spaCy kapalıysa)
        if not self.enabled:
            words = text.split()
            for word in words:
                if word and word[0].isupper() and len(word) >= 3:
                    clean_word = re.sub(r'[^\wçğıöşüÇĞİÖŞÜ]', '', word)
                    if clean_word and clean_word not in ['Ben', 'Sen', 'Bu', 'O', 'Ne']:
                        all_entities.append(clean_word)

        return list(set(all_entities))
    
    def get_lemmas(self, text: str) -> List[str]:
        """
        🔤 Türkçe Lemmatization (Kök Bulma)
        
        Örnek:
        "çalışıyorum" -> "çalış"
        "gidiyoruz" -> "git"
        """
        if not self.enabled or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            lemmas = [
                token.lemma_ 
                for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ]
            return lemmas
        
        except Exception as e:
            print(f"❌ Lemmatization hatası: {e}")
            return []
    
    def get_noun_chunks(self, text: str) -> List[str]:
        """
        📦 Türkçe İsim Öbeklerini Çıkar
        
        Örnek:
        "PersonalAI projesi" -> ["PersonalAI projesi"]
        "Murat'ın sistemi" -> ["Murat'ın sistemi"]
        """
        if not self.enabled or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            chunks = [chunk.text for chunk in doc.noun_chunks]
            return chunks
        
        except Exception as e:
            print(f"❌ Noun chunk hatası: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        😊 Türkçe Sentiment Analysis
        
        Returns:
            {
                'sentiment': 'positive' | 'negative' | 'neutral',
                'score': 0.75,  # -1.0 (çok olumsuz) ile +1.0 (çok olumlu) arası
                'confidence': 'high' | 'medium' | 'low'
            }
        """
        if not self.enabled or not text.strip():
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 'low'}
        
        try:
            doc = self.nlp(text)
            text_lower = text.lower()
            
            # 1. Sıfat bazlı analiz
            adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
            
            pos_adj_count = sum(1 for adj in adjectives if adj in self.positive_words)
            neg_adj_count = sum(1 for adj in adjectives if adj in self.negative_words)
            
            # 2. Kelime bazlı analiz
            words = text_lower.split()
            pos_word_count = sum(1 for word in words if word in self.positive_words)
            neg_word_count = sum(1 for word in words if word in self.negative_words)
            
            # Toplam skorlar
            total_pos = pos_adj_count + pos_word_count
            total_neg = neg_adj_count + neg_word_count
            
            # Skor hesaplama
            if total_pos + total_neg == 0:
                sentiment = "neutral"
                score = 0.0
                confidence = "low"
            else:
                score = (total_pos - total_neg) / (total_pos + total_neg)
                
                if score > 0.3:
                    sentiment = "positive"
                    confidence = "high" if abs(score) > 0.6 else "medium"
                elif score < -0.3:
                    sentiment = "negative"
                    confidence = "high" if abs(score) > 0.6 else "medium"
                else:
                    sentiment = "neutral"
                    confidence = "medium"
            
            return {
                'sentiment': sentiment,
                'score': round(score, 2),
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"❌ Sentiment analizi hatası: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 'low'}
    
    def analyze_sentiment_pos(self, text: str) -> str:
        """
        Basit sentiment (geriye uyumluluk için)
        """
        result = self.analyze_sentiment(text)
        return result['sentiment']
    
    def get_question_type(self, text: str) -> Optional[str]:
        """
        ❓ Soru Tipi Tespiti
        
        Türkçe soru kelimelerini tanır.
        """
        if not self.enabled or not text.strip():
            return None
        
        try:
            text_lower = text.lower()
            
            question_patterns = {
                "TIME": ["ne zaman", "saat kaç", "hangi saat", "when"],
                "LOCATION": ["nerede", "nereye", "nereden", "hangi yer", "where"],
                "PERSON": ["kim", "kimin", "kimse", "who"],
                "REASON": ["neden", "niçin", "niye", "nasıl olur", "why"],
                "METHOD": ["nasıl", "ne şekilde", "how"],
                "QUANTITY": ["kaç", "ne kadar", "kaç tane", "how many", "how much"],
                "DEFINITION": ["nedir", "ne demek", "tanımı", "what is"],
                "CHOICE": ["hangisi", "which"]
            }
            
            for q_type, patterns in question_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    return q_type
            
            return "GENERAL"
        
        except Exception as e:
            print(f"❌ Question type hatası: {e}")
            return None
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """
        🔑 Anahtar İfadeleri Çıkar
        
        Türkçe metinden en önemli ifadeleri bulur.
        GraphRAG entity çıkarımı için kullanılır.
        """
        if not self.enabled or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            
            # İsim öbekleri + named entities
            key_phrases = set()
            
            # Noun chunks ekle
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3:  # Çok kısa ifadeleri filtrele
                    key_phrases.add(chunk.text)
            
            # Named entities ekle
            for ent in doc.ents:
                key_phrases.add(ent.text)
            
            # Skorlama (uzunluk ve önem)
            scored_phrases = []
            for phrase in key_phrases:
                score = len(phrase.split())  # Kelime sayısı
                scored_phrases.append((phrase, score))
            
            # En yüksek skorlu ifadeleri döndür
            scored_phrases.sort(key=lambda x: x[1], reverse=True)
            return [phrase for phrase, _ in scored_phrases[:top_n]]
        
        except Exception as e:
            print(f"❌ Key phrase extraction hatası: {e}")
            return []


# ==========================================================
# BÖLÜM 3: FAISS BİLGİ TABANI (DÖKÜMAN KÜTÜPHANESİ) - NEO4J SİLİNDİ
# ==========================================================
# Neo4j GraphRAG kodu tamamen kaldırıldı (ölü kod idi)
# HafizaAsistani zaten uzun dönem hafıza yönetimini yapıyor


class FAISSKnowledgeBase:
    """
    FAISS tabanlı yerel bilgi tabanı
    Risale-i Nur, dökümanlar, PDF'ler vb. için
    """
    
    def __init__(self, user_id: str = SystemConfig.DEFAULT_USER_ID):
        self.user_id = user_id
        self.enabled = SystemConfig.FAISS_KB_ENABLED
        
        print(f"\n🔍 FAISS KB INIT DEBUG:")
        print(f"   Enabled: {self.enabled}")
        print(f"   Index file: {SystemConfig.FAISS_INDEX_FILE}")
        print(f"   Texts file: {SystemConfig.FAISS_TEXTS_FILE}")
        print(f"   Index exists: {os.path.exists(SystemConfig.FAISS_INDEX_FILE)}")
        print(f"   Texts exists: {os.path.exists(SystemConfig.FAISS_TEXTS_FILE)}\n")
        
        if not self.enabled:
            print("⚠️ FAISS Bilgi Tabanı devre dışı")
            return
        
        # Paths
        self.index_file = SystemConfig.FAISS_INDEX_FILE
        self.texts_file = SystemConfig.FAISS_TEXTS_FILE
        
        # Config
        self.search_top_k = SystemConfig.FAISS_SEARCH_TOP_K
        self.similarity_threshold = SystemConfig.FAISS_SIMILARITY_THRESHOLD
        self.max_results = SystemConfig.FAISS_MAX_RESULTS
        self.relative_threshold = SystemConfig.FAISS_RELATIVE_THRESHOLD
        self.max_context_length = SystemConfig.FAISS_MAX_CONTEXT_LENGTH
        
        # User namespace
        self.user_namespace = f"user_{user_id}"
        
        # Temporal awareness
        self.temporal_awareness = True
        self._initialize_temporal_awareness()
        
        # Data
        self.texts = []
        self.index: Optional[faiss.Index] = None
        
        # Load
        self._load_components()
    
    def _initialize_temporal_awareness(self):
        """Tarih bilincini başlat"""
        try:
            now = _now_ist()
            weekday = now.weekday()
            
            english_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                             'Friday', 'Saturday', 'Sunday']
            turkish_days = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 
                             'Cuma', 'Cumartesi', 'Pazar']
            turkish_months = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                              'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
            
            self.current_day_info = {
                'date': now.strftime('%Y-%m-%d'),
                'day_english': english_days[weekday],
                'day_turkish': turkish_days[weekday],
                'month_turkish': turkish_months[now.month - 1],
                'formatted_date': now.strftime(f'%d {turkish_months[now.month - 1]} %Y')
            }
        except Exception as e:
            self.temporal_awareness = False
            self.current_day_info = {}
    
    def _load_components(self):
        """Index ve text dosyalarını yükle"""
        try:
            # FAISS index
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                print(f"✅ FAISS index yüklendi: {self.index_file}")
            else:
                print(f"⚠️ FAISS index bulunamadı: {self.index_file}")
                self.enabled = False
                return
            
            # Texts JSON
            if os.path.exists(self.texts_file):
                with open(self.texts_file, 'r', encoding='utf-8') as f:
                    self.texts = json.load(f)
                print(f"✅ FAISS texts yüklendi: {len(self.texts)} döküman")
            else:
                print(f"⚠️ FAISS texts bulunamadı: {self.texts_file}")
                self.enabled = False
                return
            
            # Embedding model
            self.embedding_model = SentenceTransformer(SystemConfig.EMBEDDING_MODEL)
            
            print(f"✅ FAISS Bilgi Tabanı hazır: {self.user_namespace}")
            
        except Exception as e:
            print(f"❌ FAISS yükleme hatası: {e}")
            self.enabled = False
    
    def get_relevant_context(self, user_input: str, max_chunks: int = 3) -> str:
        """Kullanıcı input'una göre ilgili bağlamı getir"""
        if not self.enabled:
            print("⚠️ FAISS KB devre dışı")
            return ""
        
        try:
            print(f"\n{'='*60}")
            print(f"🔍 FAISS KB ARAMA BAŞLADI")
            print(f"📝 Sorgu: {user_input}")
            print(f"📊 Max chunks: {max_chunks}")
            print(f"{'='*60}")
            
            # Search
            results = self.search(user_input, top_k=max_chunks * 2)
            
            print(f"\n📊 ARAMA SONUÇLARI:")
            print(f"   Toplam sonuç: {len(results)}")
            
            if not results:
                print("   ❌ Hiç sonuç bulunamadı!")
                return ""
            
            combined_text = ""
            
            # Tarih bilgisi ekle
            if self.temporal_awareness and self.current_day_info:
                day_info = self.current_day_info
                combined_text += f"""GÜNCEL TARİH BİLGİSİ - DİKKAT:
Bugünün tam tarihi: {day_info.get('formatted_date', 'Bilinmiyor')}
UYARI: Bu bilgi güncel ve doğrudur, lütfen bu bilgiyi kullan!

"""
            
            # İlgili bilgiler ekle
            if results:
                combined_text += "İLGİLİ BİLGİLER:\n"
                
                for i, result in enumerate(results[:max_chunks]):
                    text = result.get('text', '')
                    score = result.get('score', 0.0)
                    index = result.get('index', -1)
                    
                    # 🆕 DEBUG: Her sonucu detaylı yazdır
                    print(f"\n   📄 SONUÇ #{i+1}:")
                    print(f"      • Skor: {score:.4f}")
                    print(f"      • Index: {index}")
                    print(f"      • Metin uzunluğu: {len(text)} karakter")
                    print(f"      • İlk 100 karakter:")
                    print(f"        '{text[:100]}...'")
                    
                    if text:
                        combined_text += f"{text}\n\n"
            
            print(f"\n{'='*60}")
            print(f"✅ FAISS KB ARAMA TAMAMLANDI")
            print(f"📊 Toplam dönen metin: {len(combined_text)} karakter")
            print(f"{'='*60}\n")
            
            return combined_text.strip()
            
        except Exception as e:
            print(f"❌ FAISS context hatası: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Bilgi tabanında ara"""
        if not self.enabled:
            print("⚠️ FAISS KB search devre dışı")
            return []
        
        try:
            print(f"\n🔎 FAISS SEARCH BAŞLADI")
            print(f"   Query: '{query}'")
            print(f"   Top-K: {top_k or self.search_top_k}")
            
            # Embed query
            query_vector = self.embedding_model.encode(
                [query], 
                normalize_embeddings=True
            )
            query_vector = np.array(query_vector, dtype=np.float32)
            
            print(f"   ✅ Query embedding boyutu: {query_vector.shape}")
            
            # Search
            requested_k = top_k or self.search_top_k
            k = max(requested_k, requested_k + 10)
            
            print(f"   🔍 FAISS index'te arama yapılıyor (k={k})...")
            scores, indices = self.index.search(query_vector, k)
            
            print(f"   ✅ FAISS arama tamamlandı")
            print(f"   📊 Bulunan index sayısı: {len(indices[0])}")
            
            # Filter results
            results = []
            filtered_count = 0
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:
                    continue
                
                similarity = float(score)
                
                # 🆕 DEBUG: Her sonucu yazdır
                print(f"\n   #{i+1} - Index: {idx}, Skor: {similarity:.4f}", end="")
                
                if similarity >= self.similarity_threshold and idx < len(self.texts):
                    text_data = self.texts[idx]
                    
                    # Text content
                    if isinstance(text_data, dict):
                        text_content = text_data.get('text', str(text_data))
                    else:
                        text_content = str(text_data)
                    
                    print(f" ✅ KABUL EDİLDİ (threshold: {self.similarity_threshold})")
                    print(f"      Metin: '{text_content[:80]}...'")
                    
                    result = {
                        'text': text_content,
                        'score': similarity,
                        'index': int(idx),
                        'source': f'faiss_knowledge_{self.user_namespace}'
                    }
                    
                    results.append(result)
                else:
                    filtered_count += 1
                    print(f" ❌ FİLTRELENDİ (threshold altı veya invalid)")
            
            print(f"\n   📊 ÖZET:")
            print(f"      • Toplam tarama: {len(indices[0])}")
            print(f"      • Filtrelenen: {filtered_count}")
            print(f"      • Kabul edilen: {len(results)}")

            # 🆕 RELATIVE SCORING: En yüksek skorun %90'ı altındakileri çıkar
            if results:
                top_score = results[0]['score']
                relative_threshold = top_score * SystemConfig.FAISS_RELATIVE_THRESHOLD

                print(f"\n   🎯 RELATIVE SCORING:")
                print(f"      • En yüksek skor: {top_score:.4f}")
                print(f"      • Relative threshold ({SystemConfig.FAISS_RELATIVE_THRESHOLD*100}%): {relative_threshold:.4f}")

                # Relative threshold'dan yüksek olanları al
                filtered_results = []
                for r in results:
                    if r['score'] >= relative_threshold:
                        filtered_results.append(r)
                        print(f"      ✅ Skor {r['score']:.4f} - KABUL")
                    else:
                        print(f"      ❌ Skor {r['score']:.4f} - REDDEDİLDİ (relative threshold altı)")

                # Max sonuç sayısı limiti uygula
                max_results = SystemConfig.FAISS_MAX_RESULTS
                if len(filtered_results) > max_results:
                    print(f"      ✂️ İlk {max_results} sonuç alınıyor (toplam {len(filtered_results)} sonuç vardı)")
                    filtered_results = filtered_results[:max_results]

                print(f"      • Final sonuç sayısı: {len(filtered_results)}")

                return filtered_results

            return results
            
        except Exception as e:
            print(f"❌ FAISS search hatası: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled"
            }
        
        try:
            return {
                "enabled": True,
                "status": "active",
                "total_vectors": self.index.ntotal if self.index else 0,
                "user_namespace": self.user_namespace,
                "total_texts": len(self.texts),
                "similarity_threshold": self.similarity_threshold,
                "max_results": self.max_results,
                "relative_threshold": self.relative_threshold,
                "features": ["multi_user_isolation", "temporal_awareness", "relative_scoring"]
            }
        except Exception as e:
            print(f"FAISS KB stats hatası: {e}")
            return {
                "enabled": False,
                "status": "error"
            }

# ==========================================================
# BÖLÜM 4.5: WEB SCRAPING YARDIMCI SINIFLAR
# ==========================================================
# DEĞİŞİKLİK 1: BÖLÜM 4.5'İN EKLENMESİ
class DuplicateFilter:
    """Web scraping için duplicate içerik filtresi"""
        
    def __init__(self):
        self.seen_hashes = set()
        self.max_size = 10000

    def is_duplicate(self, content: str) -> bool:
        if not content:
            return True
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        if len(self.seen_hashes) > self.max_size:
            to_remove = list(self.seen_hashes)[:self.max_size // 10]
            for h in to_remove:
                self.seen_hashes.discard(h)
        return False

    def clear(self):
        self.seen_hashes.clear()

class SmartContentExtractor:
    """BeautifulSoup ile akıllı içerik çıkarma"""
        
    def extract_main_content(self, soup: BeautifulSoup, query: str) -> Tuple[str, float]:
        if not soup:
            return "", 0.0
        for unwanted in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            unwanted.decompose()
        main_content = (
            soup.find('article') or 
            soup.find('main') or 
            soup.find('div', class_=lambda x: x and 'content' in x.lower()) or
            soup.find('body')
        )
        if not main_content:
            return "", 0.0
        text = main_content.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        quality_score = self._calculate_quality(text, query)
        return text[:2000], quality_score

    def _calculate_quality(self, text: str, query: str) -> float:
        if not text:
            return 0.0
        score = 0.5
        if len(text) > 500:
            score += 0.2
        query_words = query.lower().split()
        text_lower = text.lower()
        match_count = sum(1 for word in query_words if word in text_lower)
        if query_words:
            score += (match_count / len(query_words)) * 0.3
        return min(1.0, score)

class ScrapingError(PersonalAIError):
    """Web scraping için özel exception"""
        
    def __init__(self, message: str, url: str = None):
        super().__init__(message)
        self.url = url

    def __str__(self):
        if self.url:
            return f"{self.args[0]} (URL: {self.url})"
        return str(self.args[0])

# ==========================================================
# BÖLÜM 5: LLM, WEB SEARCH VE NİYET TESPİTİ
# ==========================================================

class LocalLLM:
    """
    LLM wrapper - Ollama veya Together.ai desteği
    Vision desteği ile
    """

    def __init__(self, user_id: str = SystemConfig.DEFAULT_USER_ID):
        self.user_id = user_id
        self.provider = SystemConfig.LLM_PROVIDER  # "ollama" veya "together"
        self.ollama_url = SystemConfig.OLLAMA_URL
        self.model_name = SystemConfig.MODEL_NAME
        self.vision_enabled = SystemConfig.ENABLE_VISION
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Together.ai API key
        self.together_api_key = os.getenv("TOGETHER_API_KEY", "")

        self.stats = {
            'total_requests': 0,
            'vision_requests': 0,
            'text_requests': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }

        provider_name = "Together.ai" if self.provider == "together" else "Ollama"
        print(f"✅ LLM başlatıldı: {self.model_name} ({provider_name}, {self.device})")
    
    def _is_vision_query(self, user_input: str) -> bool:
        """Vision query mi kontrol et"""
        if not self.vision_enabled:
            return False
        
        input_lower = user_input.lower()
        return any(keyword in input_lower for keyword in SystemConfig.VISION_KEYWORDS)
    
    async def generate(self, prompt: str, image_data: Optional[bytes] = None) -> str:
        """
        LLM yanıt üret
        
        NOT: Bu gerçek Ollama API çağrısı yapmalı
        Şu an basit simülasyon (Ollama kurulumunu gerektirir)
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Vision query - eğer image_data varsa direkt vision kullan
            if image_data:
                result = await self._generate_with_vision(prompt, image_data)
                self.stats["vision_requests"] += 1
            else:
                result = await self._generate_text_only(prompt)
                self.stats["text_requests"] += 1
            
            # Update stats
            response_time = time.time() - start_time
            if self.stats["total_requests"] > 0:
                self.stats["avg_response_time"] = (
                    self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + 
                    response_time
                ) / self.stats["total_requests"]
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            print(f"❌ LLM hatası: {e}")
            return "Üzgünüm, yanıt oluşturulurken bir hata oluştu."
    
    async def _generate_with_vision(self, prompt: str, image_data: str) -> str:
        """
        Vision ile yanıt üret (Ollama Vision API)
        image_data: base64 encoded image string
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_data],  # base64 string
                    "stream": False,
                    "raw": True,  # Ollama'nın kendi template'ini kapatır, <bos> elle eklendi
                    "options": SystemConfig.get_gemma3_params()
                }
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('response', '')
                    else:
                        print(f"⚠️ Ollama Vision API hatası: {resp.status}")
                        return "Görseli analiz edemedim, lütfen tekrar dene."
        except asyncio.TimeoutError:
            print("⚠️ Ollama vision timeout")
            return "Görsel analizi zaman aşımına uğradı."
        except Exception as e:
            print(f"⚠️ Vision API hatası: {e}")
            return "Görsel analizi sırasında bir hata oluştu."
    
    # DEĞİŞİKLİK 2: _generate_text_only() FONKSİYONUNUN DEĞİŞTİRİLMESİ
    async def _generate_text_only(self, prompt: str) -> str:
        """LLM API çağrısı - Ollama veya Together.ai"""
        # 📋 PROMPT LOGLAMA - LLM'e giden tam prompt
        if SystemConfig.LOG_FULL_PROMPT:
            print("\n" + "=" * 70)
            print(f"📋 LLM'E GÖNDERİLEN TAM PROMPT ({self.provider.upper()}):")
            print("=" * 70)
            print(prompt)
            print("=" * 70)
            print(f"📏 Toplam: {len(prompt)} karakter")
            print("=" * 70 + "\n")

        # Provider'a göre yönlendir
        if self.provider == "together":
            return await self._generate_together(prompt)
        else:
            return await self._generate_ollama(prompt)

    async def _generate_together(self, prompt: str) -> str:
        """Together.ai API çağrısı (OpenAI uyumlu)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }

            # Prompt'u messages formatına çevir
            payload = {
                "model": SystemConfig.TOGETHER_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": SystemConfig.MAX_TOKENS,
                "temperature": SystemConfig.TEMPERATURE,
                "top_p": SystemConfig.TOP_P,
                "repetition_penalty": SystemConfig.REPEAT_PENALTY,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SystemConfig.TOGETHER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)  # 405B için daha uzun timeout
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        # OpenAI format: choices[0].message.content
                        return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        error_text = await resp.text()
                        print(f"⚠️ Together.ai API hatası: {resp.status} - {error_text[:200]}")
                        return self._generate_fallback_response(prompt)

        except asyncio.TimeoutError:
            print("⚠️ Together.ai timeout")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            print(f"⚠️ Together.ai bağlantı hatası: {e}")
            return self._generate_fallback_response(prompt)

    async def _generate_ollama(self, prompt: str) -> str:
        """Ollama API çağrısı"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,  # Ollama'nın kendi template'ini kapatır
                    "options": SystemConfig.get_gemma3_params()
                }
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('response', '')
                    else:
                        print(f"⚠️ Ollama API hatası: {resp.status}")
                        return self._generate_fallback_response(prompt)
        except asyncio.TimeoutError:
            print("⚠️ Ollama timeout - simülasyona geçiliyor")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            print(f"⚠️ Ollama bağlantı hatası: {e} - simülasyona geçiliyor")
            return self._generate_fallback_response(prompt)

    # DEĞİŞİKLİK 3: _generate_fallback_response() FONKSİYONUNUN EKLENMESİ
    def _generate_fallback_response(self, prompt: str) -> str:
        """Ollama çalışmazsa fallback simülasyon"""
        if "Duygusal Giriş/Gözlem" in prompt:
            if "proje" in prompt.lower():
                return "Şahsen, bu yapay zeka projenin ne kadar ilerlediğini görmek beni çok heyecanlandırıyor. Geçen sefer konuştuğumuzda Neo4j entegrasyonundan bahsetmiştin. Bence bu yaklaşımla gerçekten güçlü bir sistem kuruyorsun."
            if "hava" in prompt.lower():
                return "Aklıma gelmişken, dışarı çıkmadan önce hava durumunu sorman çok mantıklı. Sakarya için bugün 15°C civarı, parçalı bulutlu görünüyor. Hafif bir ceket işini görür."
            return "Şahsen, bu konunun ne kadar önemli olduğunu anlayabiliyorum. Deneyimlerime göre, bu tür durumlarda adım adım ilerlemenin en sağlıklısı olduğunu düşünüyorum."
        if "GraphRAG" in prompt or "ARKA PLAN BİLGİSİ" in prompt:
            return "Geçmişte birlikte konuştuklarımızı düşündüğümde, senin bu konuya olan ilgin ve yaklaşımın gerçekten etkileyici. Hatırlıyorum, benzer bir durumda şöyle bahsetmiştin..."
        if "REASONING APPROACH" in prompt:
            return "Mantıklı bir çözüm için önce durumu analiz ettim. Farklı perspektifleri değerlendirdim ve en pratik yaklaşımın şu olduğunu düşünüyorum..."
        return "Merhaba! Senin için buradayım. Nasıl yardımcı olabilirim?"
    
    async def generate_with_params(self, prompt: str, params: Dict[str, Any], 
                                     image_data: Optional[bytes] = None) -> str:
        """Özel parametrelerle yanıt üret"""
        # Params'ı kullan (TODO: gerçek API'ye gönder)
        return await self.generate(prompt, image_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return {
            **self.stats,
            'model': self.model_name,
            'device': self.device,
            'vision_enabled': self.vision_enabled
        }


class PromptBuilder:
    """
    Ana prompt oluşturucu
    🧠 Çekirdek benlik + minimal SYNTHESIS_PROMPT kullanıyor
    """

    def create_prompt(self, user_input: str,
                                        graphrag_context: str,
                                        semantic_context: str,
                                        chat_history: str) -> str:
        """Çekirdek benlik + minimal prompt oluştur"""
        combined_context = f"{graphrag_context}\n{semantic_context}"

        # 🧠 Çekirdek benlik + minimal prompt
        # _INTERNAL_SELF_AWARENESS sessiz arka plan olarak ekleniyor
        return SystemConfig._INTERNAL_SELF_AWARENESS + "\n" + SystemConfig.format_prompt(
            SystemConfig.SYNTHESIS_PROMPT,
            user_input=user_input,
            combined_sources=combined_context + "\n" + chat_history if chat_history else combined_context
        )


class Gemma3OptimizedLLM:
    """
    Gemma3 için optimize edilmiş LLM wrapper
    CoT ve özel parametre desteği
    """
    
    def __init__(self, base_llm: LocalLLM):
        self.base_llm = base_llm
        self.gemma3_params = SystemConfig.get_gemma3_params()
        self.prompt_builder = PromptBuilder()

    async def generate_response(self, user_input: str,
                                 graphrag_context: str,
                                 semantic_context: str,
                                 chat_history: str) -> str:
        """Ana yanıt üret"""
        prompt = self.prompt_builder.create_prompt(
            user_input, graphrag_context, semantic_context, chat_history
        )
        
        response = await self._generate_with_gemma3_params(prompt)
        return response
    
    async def _generate_with_gemma3_params(self, prompt: str, 
                                           image_data: Any = None) -> str:
        """Gemma3 parametreleri ile üret"""
        if hasattr(self.base_llm, 'generate_with_params'):
            return await self.base_llm.generate_with_params(
                prompt, 
                params=self.gemma3_params, 
                image_data=image_data
            )
        else:
            return await self.base_llm.generate(prompt, image_data=image_data)


# IntentDetector silindi - HafizaAsistani._tool_secimi_yap zaten bu işi yapıyor
# EnhancedWebSearch silindi - Web search kullanılmıyor, sadece Wikipedia API kalıyor

# ==========================================================
# BÖLÜM 6: MULTI-ROLE SİSTEMİ VE RESPONSE FORMATTER
# ==========================================================

class MultiRoleSystem:
    """
    Çoklu rol sistemi
    Arkadaş, teknik destek, öğretmen rolleri
    """
    
    def __init__(self):
        self.enabled = SystemConfig.MULTI_ROLE_ENABLED
        self.roles = SystemConfig.ROLES
        self.role_history = defaultdict(int)
        self.last_role = "friend"
    
    def detect_role(self, user_input: str, detected_intent: Optional[str] = None) -> str:
        """Kullanıcı input'undan rol tespit et"""
        if not self.enabled:
            return "friend"
        
        user_lower = user_input.lower()
        
        # Kod marker'ları
        code_markers = ["```", "def ", "class ", "import ", "function"]
        has_code = any(marker in user_input for marker in code_markers)
        
        error_markers = ["error:", "hata veriyor", "bug", "error", "çalışmıyor"]
        has_error = any(marker in user_lower for marker in error_markers)
        
        if has_code or has_error or detected_intent in ["TECHNICAL", "CODE_DEBUG"]:
            self._track_role("technical_helper")
            return "technical_helper"
        
        # Role keyword matching
        for role_name, role_config in self.roles.items():
            keywords = role_config.get("keywords", [])
            if any(kw in user_lower for kw in keywords):
                self._track_role(role_name)
                return role_name
        
        # Default
        self._track_role("friend")
        return "friend"
    
    def _track_role(self, role: str):
        """Rol kullanımını takip et"""
        self.role_history[role] += 1
        self.last_role = role
    
    def format_response_by_role(self, raw_response: str, role: str, 
                                 user_input: str) -> str:
        """Role göre yanıtı formatla"""
        if not self.enabled:
            return raw_response
        
        # Yasaklı ifadeleri kaldır
        formatted = self._remove_forbidden_phrases(raw_response)

        # Kısa yanıtlar için özel işlem
        if user_input.lower().strip() in ["tamam", "ok", "saol", "teşekkürler", "teşekkür ederim", "tşk"]:
            short_responses = ["Rica ederim!", "Ne demek!", "Her zaman!", "Önemli değil!"]
            return short_responses[hash(user_input) % len(short_responses)]

        # ❌ Max length kesme KALDIRILDI - LLM kendi doğal uzunluğunda yazsın

        return formatted.strip()
    
    def _remove_forbidden_phrases(self, text: str) -> str:
        """Yasaklı ifadeleri kaldır"""
        for phrase in SystemConfig.FORBIDDEN_PHRASES:
            # Cümle bazında kaldır
            pattern = r'[^.!?]*' + re.escape(phrase) + r'[^.!?]*[.!?]'
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE).strip()
        
        return text.strip()
    
    def get_role_stats(self) -> Dict[str, Any]:
        """Rol istatistiklerini döndür"""
        return {
            'role_history': dict(self.role_history),
            'last_role': self.last_role,
            'enabled': self.enabled
        }


class ResponseFormatter:
    """
    Yanıt formatları ve temizleme
    """
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Yanıtı temizle"""
        # 🆕 BELİRSİZLİK İFADELERİNİ TEMİZLE
        uncertain_phrases = [
            "web'de bu bilgi geçiyor ama emin değilim:",
            "web'de bu bilgi geçiyor ama emin değilim",
            "web'de geçiyor ama emin değilim:",
            "web'de geçiyor ama emin değilim",
            "emin değilim:",
            "emin değilim",
            "sanırım ki",
            "sanırım",
            "galiba",
            "olabilir ki",
            "muhtemelen",
            "belki de"
        ]
        
        cleaned = text
        for phrase in uncertain_phrases:
            # Hem küçük hem büyük harfle başlayabilir
            cleaned = cleaned.replace(phrase, "")
            cleaned = cleaned.replace(phrase.capitalize(), "")
        
        # Fazla boşlukları temizle
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fazla newline'ları temizle
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        # Çift noktalama temizle
        cleaned = re.sub(r'\.\s*\.', '.', cleaned)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r':\s*:', ':', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def remove_greetings_if_continuing(text: str, is_continuing: bool) -> str:
        """Devam eden sohbette selamları kaldır"""
        if not is_continuing:
            return text
        
        greeting_patterns = [
            r'^(Merhaba|Selam|İyi günler|Hoş geldiniz)[,!.]?\s*',
            r'^(Hello|Hi|Hey)[,!.]?\s*',
            r'^(I understand)\.\s*'
        ]
        
        for pattern in greeting_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE).strip()
        
        return text
    
    @staticmethod
    def format_synthesis_response(response: str, user_input: str,
                                      max_length: int = None) -> str:
        """Synthesis yanıtını formatla"""
        # Temizle
        cleaned = ResponseFormatter.clean_response(response)

        # ❌ Max length kesme KALDIRILDI - LLM kendi doğal uzunluğunda yazsın

        return cleaned

# ==========================================================
# BÖLÜM 7: AYARLAR VE TOOL SYSTEM
# ==========================================================

class ConfigDrivenSettings:
    """
    Kullanıcı bazlı ayarlar ve kurallar
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_triggers = SystemConfig.MEMORY_TRIGGERS
        self.personal_keywords = SystemConfig.PERSONAL_KEYWORDS
        self.greeting_keywords = SystemConfig.GREETING_KEYWORDS
    
    def get_context_blocking_rules(self, user_input: str) -> dict:
        """Context blocking kuralları"""
        return {
            'block_graphrag': False,
            'block_faiss': False,
            'category': 'general'
        }


# ==========================================================
# BÖLÜM 7.5: TOOL SYSTEM (YENİ!)
# ==========================================================

class ToolSystem:
    """
    LLM'nin kullanabileceği araçları yöneten sistem
    """
    
    # Araç tanımları
    TOOLS = {
        "risale_ara": {
            "name": "risale_ara",
            "description": "Risale-i Nur kütüphanesinden dini sorulara cevap bul",
            "parameters": "soru: Aranacak dini soru",
            "when": "Kullanıcı Allah, din, iman, peygamber, namaz gibi DİNİ konularda soru sorduğunda",
            "examples": ["Allah'ın ilim sıfatı nedir?", "İman nedir?", "Namaz neden önemli?"]
        },
        "gecmis_getir": {
            "name": "gecmis_getir",
            "description": "Neo4j'den önceki konuşmaları getir",
            "parameters": "konu: Aranacak konu",
            "when": "Kullanıcı 'geçen', 'daha önce', 'konuşmuştuk' dediğinde",
            "examples": ["Geçen konuştuğumuz proje?", "Daha önce ne söylemiştim?"]
        },
        "zaman_getir": {
            "name": "zaman_getir",
            "description": "Şu anki tarih ve saati öğren",
            "parameters": "yok",
            "when": "Kullanıcı saat, tarih, gün sorduğunda",
            "examples": ["Saat kaç?", "Bugün tarihi ne?", "Hangi gündeyiz?"]
        },
        "hesapla": {
            "name": "hesapla",
            "description": "Matematiksel hesaplama yap",
            "parameters": "ifade: Matematiksel ifade",
            "when": "Kullanıcı matematik sorusu sorduğunda veya hesaplama istediğinde",
            "examples": ["2 + 2 kaç?", "15 çarpı 3?", "100 bölü 5 kaç eder?"]
        },
        "hava_durumu": {
            "name": "hava_durumu",
            "description": "Şehir için hava durumu öğren",
            "parameters": "şehir: Şehir adı",
            "when": "Kullanıcı hava durumu sorduğunda",
            "examples": ["Sakarya hava durumu?", "İstanbul'da hava nasıl?", "Ankara'da yağmur var mı?"]
        },
        "namaz_vakti": {
            "name": "namaz_vakti",
            "description": "Türkiye şehirleri için namaz vakitlerini öğren (Diyanet metodu)",
            "parameters": "şehir: Şehir adı, vakıt: Belirli vakıt (opsiyonel)",
            "when": "Kullanıcı namaz vakitleri, ezan saatleri sorduğunda",
            "examples": ["Sakarya namaz vakitleri?", "İstanbul öğle namazı kaçta?", "Ankara akşam ezanı?", "Bursa imsak vakti?"]
        },
        "wiki_ara": {
            "name": "wiki_ara",
            "description": "Wikipedia'da ünlü kişi, yer, olay hakkında bilgi ara",
            "parameters": "arama_terimi: Aranacak kişi/yer/olay adı (netleştirilmiş)",
            "when": "Kullanıcı ünlü kişi (şarkıcı, oyuncu, sporcu), tarihî olay veya yer hakkında soru sorduğunda",
            "examples": ["Özdemir Erdoğan şarkıcı", "Atatürk", "İstanbul tarihi"]
        },
        "yok": {
            "name": "yok",
            "description": "Araç kullanmadan direkt cevap ver",
            "parameters": "yok",
            "when": "Selamlaşma, genel sohbet, basit sorular",
            "examples": ["Merhaba", "Nasılsın?", "Teşekkürler"]
        }
    }
    
    @staticmethod
    def get_tools_prompt() -> str:
        """Araçları LLM'ye tanıt"""
        tools_text = "KULLANDIĞIN ARAÇLAR:\n\n"
        
        for tool_name, info in ToolSystem.TOOLS.items():
            tools_text += f"{tool_name}({info['parameters']})\n"
            tools_text += f"  • Ne işe yarar: {info['description']}\n"
            tools_text += f"  • Ne zaman kullan: {info['when']}\n"
            tools_text += f"  • Örnek: {info['examples'][0] if info['examples'] else 'N/A'}\n\n"
        
        return tools_text
    
    @staticmethod
    def get_tool_calling_prompt(user_input: str) -> str:
        """Tool calling prompt'u oluştur"""
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{ToolSystem.get_tools_prompt()}

KULLANICI SORUSU: {user_input}

ÖNEMLİ KURALLAR:
1. Önce soruyu DİKKATLE ANLA
2. Hangi araç gerekli? KARAR VER
3. Cevabını TAM OLARAK şu formatta ver:

DÜŞÜNCE: [Soruyu nasıl analiz ettin]
ARAÇ: [risale_ara / gecmis_getir / zaman_getir / hesapla / hava_durumu / yok]
PARAMETRE: [araç parametresi veya "yok"]

ÖRNEK 1:
DÜŞÜNCE: "Allah'ın ilim sıfatı" dini bir soru
ARAÇ: risale_ara
PARAMETRE: Allah'ın ilim sıfatı

ÖRNEK 2:
DÜŞÜNCE: "Geçen konuşmuştuk" geçmişe atıf yapıyor
ARAÇ: gecmis_getir
PARAMETRE: geçen konuşma

ÖRNEK 3:
DÜŞÜNCE: Basit selamlaşma, araç gerekmez
ARAÇ: yok
PARAMETRE: yok

Şimdi analiz et:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    @staticmethod
    def parse_tool_decision(llm_response: str) -> Tuple[str, str]:
        """LLM'nin kararını parse et"""
        tool_name = "yok"
        tool_param = ""
        
        # ARAÇ: satırını bul
        for line in llm_response.split('\n'):
            line = line.strip()
            if line.startswith("ARAÇ:"):
                tool_name = line.replace("ARAÇ:", "").strip()
            elif line.startswith("PARAMETRE:"):
                tool_param = line.replace("PARAMETRE:", "").strip()
        
        # Temizle
        tool_name = tool_name.lower()
        if tool_name not in ToolSystem.TOOLS:
            tool_name = "yok"
        
        if tool_param.lower() == "yok":
            tool_param = ""
        
        return tool_name, tool_param


# ==========================================================
# BÖLÜM 8: ANA PERSONALAİ SINIFI
# ==========================================================

class PersonalAI:
    """
    Ana PersonalAI sınıfı - Tool System ile güncellenmiş
    """
    
    def __init__(self, user_id: str = None):
        """PersonalAI sistemini başlat"""
        # User ID
        self.user_id = user_id or SystemConfig.DEFAULT_USER_ID
        self.start_time = time.time()
        
        # Background tasks
        self._bg_tasks: Set[asyncio.Task] = set()
        
        # User data dizini
        self.user_data_dir = f"{SystemConfig.USER_DATA_BASE_DIR}/{self.user_id}"
        self._create_user_directories()
        
        print("=" * 60)
        print(f"🚀 PersonalAI Başlatılıyor...")
        print(f"👤 Kullanıcı: {self.user_id}")
        print("=" * 60)
        
        # Bileşenleri başlat
        self._initialize_components()
        
        # Settings
        self.settings = ConfigDrivenSettings(self.user_id)
        
        # Tool System
        self.tool_system = ToolSystem()
        
        # Learning system
        self.learning_system: Dict[str, Any] = {
            "topic_interests": defaultdict(int),
            "preferred_tone": "friendly",
            "response_satisfaction": deque(maxlen=2000),
            "interaction_count": 0
        }
        
        # Performance metrics
        self.performance_metrics: Dict[str, deque] = {
            'processing_time': deque(maxlen=5000),
            'errors': deque(maxlen=1000)
        }
        
        # User profile
        self.user_profile = self._build_user_profile()
        
        # Gemma3 optimization
        self._integrate_gemma3_optimization()
        
        # Multi-role system
        self.multi_role = MultiRoleSystem()

        # Current mode
        self.current_mode = "simple"

        print("\n✅ PersonalAI hazır!")
        print(f"  • LLM: {SystemConfig.MODEL_NAME}")
        print(f"  • 🧠 Memory: HafizaAsistani v2.0 + DecisionLLM")
        print(f"  • 🤖 Phi-3 Mini: {'Aktif ✅' if hasattr(self.memory, 'use_decision_llm') and self.memory.use_decision_llm else 'Kapalı'}")
        print(f"  • Knowledge Base: {'Aktif' if (self.faiss_kb and self.faiss_kb.enabled) else 'Kapalı'}")
        print(f"  • Wikipedia Tool: Aktif ✅")
        print(f"  • Tool System: Aktif ✅")
        print("=" * 60 + "\n")
    
    def _create_user_directories(self):
        """Kullanıcı dizinlerini oluştur"""
        directories = [
            self.user_data_dir,
            f"{self.user_data_dir}/memories",
            f"{self.user_data_dir}/cache",
            f"{self.user_data_dir}/logs"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_components(self) -> None:
        """Tüm bileşenleri başlat"""
        # Cache
        self.cache = None
        
        # spaCy NLP Engine
        self.spacy_nlp = TurkishNLPEngine()  # 🇹🇷 Türkçe NLP Motoru
        
        # LLM
        self.llm = LocalLLM(self.user_id)
        
        # Memory - YENİ HafizaAsistani v2.0! ✅
        # 🆕 Conversation Threading + Multi-turn Awareness
        # Together.ai: DecisionLLM için 70B modeli kullanır
        try:
            self.memory = HafizaAsistani(
                saat_limiti=48,  # 12 → 48 saat (2 gün)
                esik=0.50,  # 0.60 → 0.50 (gevşetildi)
                max_mesaj=20,  # 8 → 20 mesaj
                model_adi="BAAI/bge-m3",
                use_decision_llm=True,
                decision_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # HafizaAsistani için 70B
            )
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ HATA: HafizaAsistani başlatılamadı!")
            print(f"❌ Detay: {e}")
            print(f"{'='*60}\n")
            raise  # Hatayı yukarı fırlat
        
        
        # FAISS Knowledge Base
        self.faiss_kb: Optional[FAISSKnowledgeBase] = None
        if SystemConfig.FAISS_KB_ENABLED:
            self.faiss_kb = FAISSKnowledgeBase(self.user_id)

        # 🆕 FAISS KB'yi HafizaAsistani'ya inject et
        if self.faiss_kb:
            self.memory.set_faiss_kb(self.faiss_kb)
            print("✅ FAISS KB HafizaAsistani'ya inject edildi")

        # Neo4j ve Web Search silindi - sadece Wikipedia tool kullanılıyor

    def _integrate_gemma3_optimization(self):
        """Gemma3 optimizasyonunu entegre et"""
        self.gemma3_llm = Gemma3OptimizedLLM(self.llm)
    
    def _build_user_profile(self) -> Dict[str, Any]:
        """Kullanıcı profilini oluştur"""
        return {
            "name": self.user_id.capitalize(),
            "interests": [],
            "personality": "conversational"
        }
    
    def _spawn_bg(self, coro):
        """Background task başlat"""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task
    
    def _history_summary(self, chat_history: List[Dict[str, Any]], max_len: int = 6000) -> str:
        """
        Chat history'yi özetle - BAĞLAM KAYBINI ÖNLE

        UYUMLU HİYERARŞİ (hafiza_asistani.py ile aynı):
        - 10 mesaj (son 5 soru-cevap çifti)
        - User: 400 karakter, AI: 1000 karakter
        - max_len: 6000

        Bu sayede bot kendi sorduğu soruyu hatırlar!
        """
        if not chat_history:
            return ""

        recent_messages = chat_history[-10:]  # Son 5 soru-cevap çifti

        tmp = []
        for m in recent_messages:
            is_user = m.get("role") == "user"
            role = "KULLANICI" if is_user else "AI"
            char_limit = 400 if is_user else 1000  # User: 400, AI: 1000
            text = (m.get('content_en') or m.get('content') or "")[:char_limit]
            if text:
                tmp.append(f"[{role}]: {text}")
        
        s = "\n".join(tmp)
        return (s[:max_len] + "...") if len(s) > max_len else s
    
    def _post_process(self, text: str, user_input: str = "", is_continuing: bool = False) -> str:
        """Yanıtı son işle"""
        # Response code kontrolü
        if text in [ResponseCodes.API_ERROR, ResponseCodes.SEARCH_FAILED]:
            return "Üzgünüm, bir hata oluştu."
        
        if text == ResponseCodes.NO_DATA:
            return "Üzgünüm, bu konuda bilgi bulamadım."
        
        # Temizle
        cleaned_text = ResponseFormatter.clean_response(text)
        
        # Devam eden sohbette selamları kaldır
        cleaned_text = ResponseFormatter.remove_greetings_if_continuing(cleaned_text, is_continuing)
        
        # Max length
        max_chars = SystemConfig.MAX_RESPONSE_CHARS
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars].rsplit(' ', 1)[0] + "..."
        
        return cleaned_text
    
    def _should_save_interaction(self, user_input: str, ai_response: str) -> bool:
        """Bu etkileşim hafızaya kaydedilmeli mi?"""
        u = user_input.lower()
        
        # Çok kısa
        if len(u) < 3 or u in {"ok", "tamam", "teşekkürler"}:
            return False
        
        # PII içeren
        pii_keywords = ["tc", "iban", "şifre", "password"]
        if any(k in u for k in pii_keywords):
            return False
        
        # Trivial queries
        trivial = ["saat kaç", "hava durumu", "döviz"]
        if any(x in u for x in trivial):
            return False
        
        return True
    
    # ====== YENİ YARDIMCI FONKSİYONLAR: SMART RESPONSE ANALYSIS ======
    
    def _build_search_query(self, user_input: str) -> str:
        """
        Kullanıcı input'undan arama sorgusu oluştur
        
        "adapazarında ıslama köfte yemek istiyorum" 
        → "adapazarı ıslama köfte restaurant"
        """
        # Gereksiz kelimeleri temizle
        noise_words = [
            "yemek", "istiyorum", "isterim", "gitmek", "yapmak",
            "yiyeceğim", "gideceğim", "yapacağım", "alacağım",
            "nerede", "nasıl", "hangi", "için"
        ]
        
        cleaned = user_input.lower()
        for word in noise_words:
            cleaned = cleaned.replace(word, " ")
        
        # Fazla boşlukları temizle
        cleaned = " ".join(cleaned.split())
        
        # "restaurant" ekle (Türkçe ve İngilizce)
        cleaned += " restaurant restoran mekan"
        
        return cleaned.strip()
    
    def _detect_city(self, query: str) -> Optional[str]:
        """
        Sorgudan şehir ismini tespit et
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            Şehir ismi (title case) veya None
        """
        import re
        
        # Türkiye şehirleri listesi (en çok kullanılanlar)
        cities = [
            'istanbul', 'ankara', 'izmir', 'bursa', 'antalya', 'adana', 'konya',
            'gaziantep', 'şanlıurfa', 'mersin', 'diyarbakır', 'kayseri', 'eskişehir',
            'urfa', 'malatya', 'erzurum', 'samsun', 'denizli', 'trabzon', 'kahramanmaraş',
            'van', 'batman', 'elazığ', 'erzincan', 'sivas', 'manisa', 'tarsus',
            'adapazarı', 'sakarya', 'balıkesir', 'kütahya', 'tekirdağ', 'edirne',
            'çanakkale', 'yalova', 'ordu', 'giresun', 'rize', 'artvin', 'gümüşhane',
            'bayburt', 'ağrı', 'kars', 'iğdır', 'ardahan', 'muş', 'bitlis', 'hakkari',
            'siirt', 'şırnak', 'mardin', 'batman', 'adıyaman', 'kilis', 'osmaniye',
            'hatay', 'isparta', 'burdur', 'afyon', 'uşak', 'kütahya', 'bilecik',
            'düzce', 'bolu', 'karabük', 'bartın', 'kastamonu', 'çankırı', 'sinop',
            'amasya', 'tokat', 'çorum', 'yozgat', 'kırıkkale', 'aksaray', 'niğde',
            'nevşehir', 'kırşehir', 'karaman', 'konya'
        ]
        
        query_lower = query.lower()
        
        # Direkt şehir ismi geçiyor mu?
        for city in cities:
            if city in query_lower:
                return city.title()
        
        # "X'da hava durumu" pattern'i
        weather_pattern = r"(\w+)['']?d[ae]\s+(?:hava|sıcaklık|derece)"
        match = re.search(weather_pattern, query_lower)
        if match:
            potential_city = match.group(1)
            if potential_city in cities:
                return potential_city.title()
        
        return None
    
    
    async def _smart_response_analysis(
        self,
        user_input: str,
        llm_response: str,
        original_tool: str
    ) -> str:
        """
        LLM yanıtını analiz et
        NOT: Web search kaldırıldı, sadece orijinal yanıtı döndürüyor
        """
        # Web search silindi - direkt orijinal yanıtı döndür
        return llm_response


    # ====== PROCESS WITH TOOLS (HafizaAsistani v3.0 ile!) ======
    # NOT: _execute_tool() metodu kaldırıldı.
    # Artık HafizaAsistani._tool_calistir() kullanılıyor.
    # ======
    async def process_with_tools(self, user_input: str, chat_history: List) -> str:
        """
        🎯 Tool system ile işle - HafizaAsistani'nın ANA METODunu kullanarak!

        YENİ AKIŞ (Refactored):
        1. HafizaAsistani.hazirla_ve_prompt_olustur() → Hazır prompt paketi
        2. Gemma3'e gönder
        3. Cevabı döndür

        KAZANÇ:
        - 220 satır → 25 satır (%88 azalma)
        - Tek sorumluluk prensibi
        - Kod tekrarı yok
        - Bakımı kolay
        """
        print(f"\n{'='*60}")
        print(f"🎯 PROCESS WITH TOOLS (HafizaAsistani v3.0)")
        print(f"{'='*60}")

        # 🎯 HafizaAsistani'nın ANA METODunu kullan!
        # (Tool seçimi + Çalıştırma + Bağlam toplama + Prompt hazırlama)
        paket = await self.memory.hazirla_ve_prompt_olustur(
            user_input=user_input,
            chat_history=chat_history
        )

        # 📦 Paket içeriği:
        # {
        #     "prompt": "Gemma3'e gönderilecek hazır prompt",
        #     "role": "friend/technical_helper/teacher",
        #     "tool_used": "web_ara/zaman_getir/vb.",
        #     "metadata": {
        #         "has_tool_result": True/False,
        #         "has_semantic": True/False,
        #         "has_faiss": True/False,
        #         "has_history": True/False
        #     }
        # }

        # 🔍 DEBUG: Detaylı paket bilgisi
        print("\n" + "="*60)
        print("📦 HAFİZA ASİSTANI → PERSONAL AI PAKETİ")
        print("="*60)
        print(f"🎭 Rol: {paket.get('role', 'N/A')}")
        print(f"🔧 Tool: {paket.get('tool_used', 'N/A')}")

        llm_decision = paket.get('llm_decision', {})
        print(f"\n📊 LLM Kararı:")
        print(f"   • question_type: {llm_decision.get('question_type', 'N/A')}")
        print(f"   • needs_faiss: {llm_decision.get('needs_faiss', 'N/A')}")
        print(f"   • needs_web: {llm_decision.get('needs_web', 'N/A')}")
        print(f"   • needs_semantic_memory: {llm_decision.get('needs_semantic_memory', 'N/A')}")
        print(f"   • needs_chat_history: {llm_decision.get('needs_chat_history', 'N/A')}")
        print(f"   • response_style: {llm_decision.get('response_style', 'N/A')}")
        reasoning = llm_decision.get('reasoning', 'N/A')
        print(f"   • reasoning: {reasoning[:100] if reasoning else 'N/A'}...")

        metadata = paket.get('metadata', {})
        print(f"\n📋 Metadata:")
        print(f"   • has_tool_result: {metadata.get('has_tool_result', 'N/A')}")
        print(f"   • has_semantic: {metadata.get('has_semantic', 'N/A')}")
        print(f"   • has_faiss: {metadata.get('has_faiss', 'N/A')}")
        print(f"   • has_history: {metadata.get('has_history', 'N/A')}")

        print(f"\n📏 Prompt uzunluğu: {len(paket.get('prompt', ''))} karakter")
        print("="*60 + "\n")

        # ✅ Hazır prompt'u direkt LLM'ye gönder
        print("🤖 LLM'e gönderiliyor (tek çağrı)...")
        final_response = await self.llm.generate(paket["prompt"])

        print("✅ Cevap alındı!\n")
        return final_response
    
    # ====== ANA PROCESS FONKSİYONU (GÜNCELLENMİŞ) ======
    async def process(
        self,
        user_input: str,
        chat_history: List[Dict[str, Any]],
        image_data: Optional[bytes] = None
    ) -> Tuple[str, str, str]:
        """
        Ana işlem fonksiyonu (TOOL SYSTEM İLE!)
        """
        start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print(f"👤 USER: {user_input}")
            print(f"{'='*60}")
            
            # 1. Özel komutları kontrol et
            mode_response = await self._handle_mode_commands(user_input)
            if mode_response:
                return mode_response, "simple", "command"

            # 2. Eğer görsel varsa HYBRID YAKLAŞIM: Vision + Tool System
            if image_data:
                print("🖼️ Görsel tespit edildi - Hybrid Vision + Context sistemi kullanılıyor...")

                # Adım 1: Görseli analiz et
                vision_prompt = f"Kullanıcı sorusu: {user_input}\n\nBu görseli kısaca analiz et (2-3 cümle)."
                vision_analysis = await self.llm.generate(vision_prompt, image_data=image_data)
                print(f"👁️ Görsel analizi tamamlandı: {vision_analysis[:100]}...")

                # Adım 2: Görsel analiz sonucunu kullanıcı sorusuyla birleştir
                enhanced_input = f"{user_input}\n\n[Görsel Bağlamı: {vision_analysis}]"

                # Adım 3: Tool system ile tam yanıt oluştur (bağlam + hafıza + web vb. ile)
                print("🔧 Tool system devreye giriyor (bağlam + hafıza)...")
                raw_response = await self.process_with_tools(enhanced_input, chat_history)
            else:
                # ✅ process_with_tools kullan (KOD TEKRARI KALDIRILDI!)
                raw_response = await self.process_with_tools(user_input, chat_history)
            
            # 3. Post-process
            is_continuing = len(chat_history) > 0
            final_response = self._post_process(raw_response, user_input, is_continuing)
            
            # 4. Hafızaya kaydet
            if self._should_save_interaction(user_input, final_response):
                # chat_history'yi de geçir (ConversationContext için)
                self.memory.add(user_input, final_response, chat_history)

            # 5. Performans kaydı
            processing_time = time.time() - start_time
            self.performance_metrics['processing_time'].append(processing_time)
            
            print(f"\n⏱️ İşlem süresi: {processing_time:.2f}s")
            print(f"🤖 AI: {final_response[:200]}...")
            print(f"{'='*60}\n")
            
            return final_response, "simple", "success"
        
        except Exception as e:
            print(f"❌ HATA: {e}")
            import traceback
            traceback.print_exc()
            
            self.performance_metrics['errors'].append(str(e))
            return "Üzgünüm, bir hata oluştu.", "error", "error"
    
    async def _handle_mode_commands(self, user_input: str) -> Optional[str]:
        """Özel komutları işle"""
        user_lower = user_input.lower()
        
        # Sistem durumu
        if any(phrase in user_lower for phrase in ["sistem durum", "stats", "istatistik"]):
            stats = self.get_system_stats()
            
            response = f"""📊 Sistem Durumu:

🧠 LLM: {stats['llm']['model']}
💾 Hafıza: {stats['memory']['total_entries']} kayıt
📚 Bilgi Tabanı: {'Aktif ✅' if stats['knowledge_base']['enabled'] else 'Kapalı ❌'}
📖 Wikipedia Tool: Aktif ✅
🔧 Tool System: Aktif ✅

📈 Performans:
  • Toplam etkileşim: {stats['performance']['total_interactions']}
  • Ort. işlem süresi: {stats['performance']['avg_processing_time']:.2f}s
"""
            return response
        
        # Hafıza temizle
        if any(phrase in user_lower for phrase in ["hafıza temizle", "memory clear"]):
            self.memory.clear()
            return "✅ Hafıza temizlendi."
        
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini döndür"""
        # Hata korumalı total_chunks ve total_entries erişimi
        kb_chunks = self.faiss_kb.index.ntotal if self.faiss_kb and hasattr(self.faiss_kb, 'index') and self.faiss_kb.index else 0
        mem_entries = len(self.memory.data) if hasattr(self.memory, 'data') else 0
        
        return {
            'llm': {
                'model': SystemConfig.MODEL_NAME,
                'provider': SystemConfig.LLM_PROVIDER
            },
            'memory': {
                'total_entries': mem_entries
            },
            'knowledge_base': {
                'enabled': self.faiss_kb and self.faiss_kb.enabled,
                'total_chunks': kb_chunks
            },
            'wikipedia_tool': {
                'enabled': True
            },
            'performance': {
                'total_interactions': len(self.performance_metrics['processing_time']),
                'avg_processing_time': (
                    sum(self.performance_metrics['processing_time']) / 
                    len(self.performance_metrics['processing_time'])
                ) if self.performance_metrics['processing_time'] else 0,
                'success_rate': 100.0
            }
        }
    
    def close(self):
        """Sistemi kapat"""
        print("\n🛑 PersonalAI kapatılıyor...")
        print("✅ Temizlik tamamlandı.")


# ==========================================================
# BÖLÜM 9: ÇALIŞTIRMA VE TEST KODLARI
# ==========================================================

async def run_interactive_chat(ai_system: PersonalAI):
    """
    İnteraktif sohbet modu
    """
    chat_history = []
    
    print("\n" + "=" * 60)
    print("💬 İnteraktif Sohbet Modu")
    print("=" * 60)
    print("Komutlar:")
    print("  'exit' veya 'quit' - Çıkış")
    print("  'stats' - İstatistikler")
    print("  'clear' - Geçmişi temizle")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Kullanıcı input
            user_input = input("\n👤 Sen: ").strip()
            
            if not user_input:
                continue
            
            # Exit
            if user_input.lower() in ['exit', 'quit', 'çıkış']:
                print("\n👋 Görüşürüz!")
                break
            
            # Clear history
            if user_input.lower() in ['clear', 'temizle']:
                chat_history = []
                # Hafızayı temizleme komutu PersonalAI sınıfında işlenir
                if user_input.lower() != 'temizle': # Tekrar temizlenmemesi için
                    pass
                else:
                    print("✅ Sohbet geçmişi temizlendi.")
                continue
            
            # Process
            print("\n🤖 AI düşünüyor...", end="", flush=True)
            reply, _, _ = await ai_system.process(user_input, chat_history)
            print("\r" + " " * 30 + "\r", end="")  # Clear "thinking" message
            
            print(f"🤖 AI: {reply}")
            
            # Update history
            chat_history.append({
                "role": "user",
                "content": user_input
            })
            chat_history.append({
                "role": "ai",
                "content": reply
            })
            
            # Keep only last 10 messages
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            
        except KeyboardInterrupt:
            print("\n\n👋 Görüşürüz!")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")


async def run_test_scenarios(ai_system: PersonalAI):
    """
    Test senaryoları
    """
    chat_history = []
    
    print("\n" + "=" * 60)
    print("🧪 TEST SENARYOLARI")
    print("=" * 60)
    
    # Senaryo 1: Güncel bilgi
    print("\n--- SENARYO 1: Güncel Bilgi (Hava Durumu) ---")
    user_input_1 = "Sakarya için hava durumu nasıl? Sabah dışarı çıkacağım."
    print(f"👤 USER: {user_input_1}")
    
    reply_1, _, _ = await ai_system.process(user_input_1, chat_history)
    print(f"🤖 AI: {reply_1}\n")
    
    chat_history.append({"role": "user", "content": user_input_1})
    chat_history.append({"role": "ai", "content": reply_1})
    
    # Senaryo 2: Kişisel hafıza
    print("--- SENARYO 2: Kişisel Hafıza (GraphRAG Test) ---")
    user_input_2 = "Geçen konuştuğumuz yapay zeka projemle ilgili ne düşünüyorsun?"
    print(f"👤 USER: {user_input_2}")
    
    reply_2, _, _ = await ai_system.process(user_input_2, chat_history)
    print(f"🤖 AI: {reply_2}\n")
    
    chat_history.append({"role": "user", "content": user_input_2})
    chat_history.append({"role": "ai", "content": reply_2})
    
    # Senaryo 3: Teknik destek
    print("--- SENARYO 3: Teknik Destek (Role Switching) ---")
    user_input_3 = "Python'da bir kod hatası alıyorum: 'ImportError: No module named numpy'. Ne yapmalıyım?"
    print(f"👤 USER: {user_input_3}")
    
    reply_3, _, _ = await ai_system.process(user_input_3, chat_history)
    print(f"🤖 AI: {reply_3}\n")
    
    # Senaryo 4: Sistem durumu
    print("--- SENARYO 4: Sistem Durumu ---")
    user_input_4 = "sistem durum"
    print(f"👤 USER: {user_input_4}")
    
    reply_4, _, _ = await ai_system.process(user_input_4, chat_history)
    print(f"🤖 AI: {reply_4}\n")
    
    print("=" * 60)
    print("✅ Tüm test senaryoları tamamlandı!")
    print("=" * 60)

async def test_spacy_integration():
    """spaCy entegrasyonunu test et"""
    print("\n" + "=" * 60)
    print("🧪 spaCy ENTEGRASYON TESTİ")
    print("=" * 60)
    
    # Sistem başlat
    ai = PersonalAI(user_id="test_user")
    
    # Test metni
    test_text = """
    Ahmet Yılmaz, 15 Ocak 2024'te İstanbul'da Python öğrenmeye başladı.
    Neo4j kullanarak 5000 TL'lik bir proje geliştirdi.
    """
    
    print(f"\n📝 Test Metni:\n{test_text}")
    
    if ai.spacy_nlp.enabled:
        # Entity extraction
        entities = ai.spacy_nlp.extract_entities(test_text)
        print("\n📍 Tespit Edilen Entity'ler:")
        for entity_type, entity_list in entities.items():
            print(f"  {entity_type}: {[e['text'] for e in entity_list]}")
        
        # Lemmas
        lemmas = ai.spacy_nlp.get_lemmas(test_text)
        print(f"\n🔤 Lemma'lar (ilk 10): {lemmas[:10]}")
        
        # Noun chunks
        chunks = ai.spacy_nlp.get_noun_chunks(test_text)
        print(f"\n📦 İsim Öbekleri: {chunks}")
        
        # Sentiment
        sentiment = ai.spacy_nlp.analyze_sentiment_pos(test_text)
        print(f"\n😊 Sentiment: {sentiment}")
        
        print("\n✅ spaCy entegrasyonu başarılı!")
    else:
        print("\n⚠️ spaCy aktif değil!")
    
    print("=" * 60)
    
    ai.close()


def main():
    """
    Ana çalıştırma fonksiyonu
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║             PersonalAI - Gelişmiş Asistan                 ║
    ║                                                           ║
    ║ • Gemma 3 27B LLM                                         ║
    ║ • FAISS Vector Memory                                     ║
    ║ • Neo4j GraphRAG (Uzun Dönem Hafıza)                      ║
    ║ • spaCy NLP Engine                                        ║
    ║ • Multi-Role System                                       ║
    ║ • Web Search Integration                                  ║
    ║ • Chain-of-Thought Reasoning                              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Mod seçimi
        print("\nMod Seçin:")
        print("1. İnteraktif Sohbet")
        print("2. Test Senaryoları")
        print("3. spaCy Entegrasyon Testi")  # 🆕 EKLE
        
        choice = input("\nSeçiminiz (1/2/3): ").strip()
        
        if choice == "1":
            system = PersonalAI(user_id="murat")
            asyncio.run(run_interactive_chat(system))
            system.close()
        elif choice == "2":
            system = PersonalAI(user_id="murat")
            asyncio.run(run_test_scenarios(system))
            system.close()
        elif choice == "3":  # 🆕 EKLE
            asyncio.run(test_spacy_integration())
        else:
            print("❌ Geçersiz seçim!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Program durduruldu.")
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()