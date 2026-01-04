import threading
import time
import random
import wikipedia
from flask import Flask, request, render_template_string, session, redirect, url_for
from neo4j import GraphDatabase
import requests
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import faiss
import torch
import re
import concurrent.futures
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# .env dosyasını yükle
load_dotenv()

# YENİ İMPORTLAR
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LOGGİNG AYARLARI - Sadece önemli bilgiler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Harici kütüphanelerin debug loglarını kapat
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def load_config(config_path='config.json'):
    """
    Konfigürasyon dosyasını yükler veya varsayılan ayarları oluşturur.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        default_config = {
            "llm_provider": "ollama",  # "ollama" veya "together"
            "ollama_model_name": "gemma3:27b",
            "together": {
                "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "similarity_threshold": 0.85,
            "coherence_threshold": 0.60,
            "llm_coherence_threshold": 0.60,
            "external_sources": {
                "wikipedia": False,
                "scholar": False,
                "twitter": False,
                "pubmed": False,
                "ieee": False,
                "wikipedia_lang": "en"
            },
            "generation": {
                "enabled": True,
                "max_tokens": 1500,
                "temperature": 0.85,
                "confidence_threshold": 0.85,
                "num_ctx": 8192
            },
            "auto_background": {
                "enabled": True,
                "interval_minutes": 10,
                "relation_generation_depth": 2,
                "max_random_entity_pairs": 4,
                "relation_similarity_threshold": 0.7
            },
            "case_sensitive": False,
            "thinking_framework": {
                "enabled": True,
                "min_unique_categories": 9,
                "synthesis_retry_limit": 2,
                "steps_enabled": {
                    "opposites": True,
                    "quantum_field": True,
                    "zero_point": True,
                    "wisdom": True,
                    "holistic_meaning": True,
                    "divine_order": True,
                    "recursive_questioning": True,
                    "speculative_theory": True,
                    "truth_void": True
                },
                "synthesis_enabled": True,
                "max_opposites": 4,
                "max_potentials": 5,
                "max_wisdom_insights": 4,
                "max_timeline_stages": 4,
                "protected_categories": ["speculative_theory", "holistic_meaning", "truth_void"],
                "force_include_protected": True,
                "similarity_threshold_protected": 0.65,
                "content_preservation_keywords": ["özgün", "felsefi", "derin", "yeni teori"]
            },
            "beautification": {
                "enabled": True,
                "auto_beautify_synthesis": True,
                "style": "perspective_rich",
                "max_length": 12000
            },
            "update_interval_days": 7,
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "senegal5454"
            },
            "file_paths": {
                "thinking_framework": "thinking_framework.json",
                "local_knowledge": "local_knowledge.json",
                "faiss_index": "faiss_index.bin",
                "faiss_texts": "faiss_texts_final.json"
            }
        }
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(default_config, file, indent=4, ensure_ascii=False)
        logging.info('[!] Config dosyası bulunamadı. Varsayılan ayarlar oluşturuldu.')
        return default_config
    except json.JSONDecodeError:
        logging.error('[!] Config dosyasında hata var. Lütfen geçerli bir JSON girin.')
        return {}

def save_config(config, config_path='config.json'):
    """
    Konfigürasyon dosyasını kaydeder.
    """
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

def together_generate(prompt, config):
    """
    Together AI API ile metin üretir (Llama 405B).
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        logging.error("TOGETHER_API_KEY bulunamadı!")
        return "Error: TOGETHER_API_KEY not found", 0

    together_config = config.get("together", {})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": together_config.get("model", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": together_config.get("max_tokens", 4096),
        "temperature": together_config.get("temperature", 0.7),
        "top_p": together_config.get("top_p", 0.9),
        "repetition_penalty": 1.15,  # Tekrarları önle
        "stop": ["---", "###", "\n\n\n\n"]  # Aşırı tekrar durumunda durdur
    }

    try:
        logging.info(f"🚀 Together AI çağrılıyor: {data['model']}")
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"].strip()
        logging.info(f"✅ Together AI yanıt aldı!")
        return generated_text, 3
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Together AI hatası: {e}")
        return f"Error: {str(e)}", 0

def load_thinking_framework(framework_path='thinking_framework.json'):
    """
    Düşünce çerçevesi dosyasını yükler.
    """
    try:
        with open(framework_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if "thinking_framework" in data:
                return data["thinking_framework"]
            else:
                logging.warning(f"'thinking_framework' anahtarı bulunamadı, tüm JSON'u döndürüyorum")
                return data
    except FileNotFoundError:
        logging.error(f"Thinking framework file not found: {framework_path}")
        default_framework = {
            "core_principle": {
                "rule": "Analyze topics from multiple perspectives to gain deeper understanding.",
                "description": "Comprehensive analysis through systematic exploration of different viewpoints and possibilities."
            },
            "steps": [
                {
                    "name": "opposites",
                    "description": "Generate two completely opposing interpretations of the given topic.",
                    "instructions": ["Generate two completely opposing interpretations.", "Each interpretation must be at least 50 words.", "Explore different perspectives."],
                    "output_format": {"opposite_1": "string", "opposite_2": "string"}
                },
                {
                    "name": "quantum_field",
                    "description": "Explore potential scenarios as a field of possibilities.",
                    "instructions": ["Generate three unique, specific potential outcomes.", "Each outcome must be at least 30 words.", "Provide a probability for each scenario."],
                    "output_format": {"scenarios": [{"scenario": "string", "probability": "float"}]}
                },
                {
                    "name": "zero_point",
                    "description": "Identify the core essence or origin of the topic.",
                    "instructions": ["Define the technical zero-point.", "Define the philosophical zero-point.", "Explain the fundamental nature."],
                    "output_format": {"technical": "string", "philosophical": "string"}
                },
                {
                    "name": "wisdom",
                    "description": "Extract universal wisdom insights.",
                    "instructions": ["Provide two concise, specific insights.", "Each insight must be at least 30 words.", "Connect the insight to a universal principle."],
                    "output_format": {"insights": ["string", "string"]}
                },
                {
                    "name": "holistic_meaning",
                    "description": "Synthesize all previous outputs into a comprehensive, cohesive meaning.",
                    "instructions": ["Combine all insights, opposites, and potentials.", "Create a single, narrative-rich explanation.", "Provide a unified understanding."],
                    "output_format": {"meaning": "string"}
                },
                {
                    "name": "divine_order",
                    "description": "Define the temporal progression of the topic.",
                    "instructions": ["Outline the topic's evolution in 3-4 distinct stages.", "Each stage should have a clear description and timestamp."],
                    "output_format": {"timeline": [{"stage": "string", "description": "string", "timestamp": "string"}]}
                },
                {
                    "name": "recursive_questioning",
                    "description": "Generate deeper, more profound questions based on the holistic meaning.",
                    "instructions": ["Generate three new, thought-provoking questions.", "Each question must challenge the current understanding.", "Questions should explore deeper aspects."],
                    "output_format": {"questions": ["string", "string", "string"]}
                },
                {
                    "name": "speculative_theory",
                    "description": "Develop a bold speculative theory that pushes beyond current understanding.",
                    "instructions": ["Create a theory that connects seemingly unrelated phenomena.", "The theory must have at least one testable prediction.", "Ensure the theory is coherent and profound."],
                    "output_format": {"theories": [{"hypothesis": "string", "based_on_nodes": ["string"], "alternative_theory": "string"}]}
                },
                {
                    "name": "truth_void",
                    "description": "Identify the fundamental unknowns and assumptions.",
                    "instructions": ["List what is currently unknown.", "Identify assumptions made without evidence.", "Generate paradoxes or contradictions in the topic."],
                    "output_format": {"truth": {"core_truth": "string", "outcome": "string", "paired_opposite_contribution": "string"}, "void": {"illusion": "string", "outcome": "string", "paired_opposite_contribution": "string"}}
                }
            ]
        }
        with open(framework_path, 'w', encoding='utf-8') as file:
            json.dump({"thinking_framework": default_framework}, file, indent=4, ensure_ascii=False)
        logging.info(f"Varsayılan thinking framework oluşturuldu: {framework_path}")
        return default_framework
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in thinking framework file: {framework_path}")
        return None

# Config yükle
config = load_config('config.json')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

llama_lock = threading.Lock()

class QuantumTree:
    """
    QuantumTree ana sınıfı, tüm veri çekme, düşünce işleme ve sentezleme mantığını içerir.
    """
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None, thinking_framework_path=None):
        
        # Config'ten Neo4j ayarlarını al
        if neo4j_uri is None:
            neo4j_uri = config.get('neo4j', {}).get('uri', 'bolt://localhost:7687')
        if neo4j_user is None:
            neo4j_user = config.get('neo4j', {}).get('user', 'neo4j')
        if neo4j_password is None:
            neo4j_password = config.get('neo4j', {}).get('password', os.environ.get("NEO4J_PASS", "senegal5454"))
        
        # Thinking framework path'i config'ten al
        if thinking_framework_path is None:
            thinking_framework_path = config.get('file_paths', {}).get('thinking_framework', 'thinking_framework.json')

        # Neo4j enabled kontrolü
        neo4j_enabled = config.get('neo4j', {}).get('enabled', False)

        if neo4j_enabled:
            try:
                self.core = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                with self.core.session() as session:
                    session.run("RETURN 1")
                logging.info("Neo4j bağlantısı başarılı")
                self.fix_neo4j_schema()
            except Exception as e:
                logging.error(f"Neo4j bağlantı hatası: {e}")
                self.core = None
        else:
            logging.info("Neo4j devre dışı (config: neo4j.enabled=false)")
            self.core = None

        # TEK MODEL SEÇİMİ - BAAI/bge-m3
        UNIFIED_MODEL_NAME = 'BAAI/bge-m3'
        
        # Sentence Transformer ve Embedder'ı birleştirilmiş tek model olarak yükle
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedder = SentenceTransformer(UNIFIED_MODEL_NAME, device=device)
            self.sentence_model = self.embedder # Artık tek bir model kullanılıyor
            logging.info(f"Sentence transformer model loaded: {UNIFIED_MODEL_NAME} on {device}")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model: {str(e)}")
            self.embedder = None
            self.sentence_model = None
        
        if self.core:
            self.similarity_threshold = 0.7
            logging.info("🔗 Enhanced Neo4j integration loaded with BAAI/bge-m3")
        
        self.model_prompt = (
            "You are a helpful and intelligent assistant. Your goal is to provide a comprehensive and profound overview of the given topic. "
            "Think freely and deeply, exploring all possible perspectives, connections, and philosophical implications without being constrained by specific categories. "
            "Synthesize all available information into a single, cohesive, and natural-sounding conversation or essay in Turkish. Ensure your response is engaging and insightful."
        )
        
        self.running = True
        self.background_thread = threading.Thread(target=self.auto_fetch_background, daemon=True)
        
        self.thinking_framework = load_thinking_framework(thinking_framework_path)
        if self.thinking_framework:
            logging.info(f"Thinking framework loaded: {len(self.thinking_framework.get('steps', []))} steps")
            logging.debug(f"Thinking framework steps: {[step['name'] for step in self.thinking_framework.get('steps', [])]}")
        else:
            logging.error("Failed to load thinking framework. Disabling advanced thought processing.")
        
        try:
            knowledge_file = config.get('file_paths', {}).get('local_knowledge', 'local_knowledge.json')
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                self.local_knowledge = json.load(f)
            logging.info("Local knowledge base loaded from JSON.")
        except FileNotFoundError:
            logging.warning("Local knowledge file not found, creating default")
            self.local_knowledge = {"sources": []}
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.local_knowledge, f, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            logging.error("Invalid JSON in local knowledge file: local_knowledge.json")
            self.local_knowledge = {"sources": []}
        
        self.faiss_index = None
        self.faiss_texts = []
        self._load_faiss_data()

    def _load_faiss_data(self):
        index_path = config.get('file_paths', {}).get('faiss_index', 'faiss_index.bin')
        texts_path = config.get('file_paths', {}).get('faiss_texts', 'faiss_texts.json')
        
        if os.path.exists(index_path) and os.path.exists(texts_path):
            try:
                self.faiss_index = faiss.read_index(index_path)
                with open(texts_path, 'r', encoding='utf-8') as f:
                    self.faiss_texts = json.load(f)
                logging.info(f"FAISS verileri yüklendi. Vektör sayısı: {self.faiss_index.ntotal}")
            except Exception as e:
                logging.error(f"FAISS veri yükleme hatası: {e}")
                self.faiss_index = None
                self.faiss_texts = []
        else:
            logging.warning("FAISS dosyaları bulunamadı.")

    def extract_smart_keywords(self, text):
        """Metinden akıllı anahtar kelimeler çıkar"""
        # Temizlik
        clean_text = text.lower()
        
        # Gereksiz kelimeleri çıkar
        stop_words = ['what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'you', 'me', 'i', 'we', 'they', 'this', 'that', 'these', 'those', 'nedir', 'nasıl', 'neden', 'ne', 'bu', 'şu', 'o']
        
        # Kelimeleri ayır ve filtrele
        words = clean_text.replace('?', '').replace('.', '').replace(',', '').split()
        meaningful_words = []
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                meaningful_words.append(word)
        
        # En önemli 2 kelimeyi döndür
        return ' '.join(meaningful_words[:2]) if meaningful_words else 'science'

    def faiss_search(self, query: str, k: int = 5) -> list:
        if not self.faiss_index or not self.faiss_texts or not self.sentence_model:
            logging.debug("FAISS arama atlandı: index, texts veya sentence_model eksik")
            return []
        
        try:
            query_vector = self.sentence_model.encode([query], convert_to_tensor=True).cpu().numpy().astype('float32')
            query_vector = np.ascontiguousarray(query_vector)
            
            distances, indices = self.faiss_index.search(query_vector, k)
            
            results = []
            for i in range(len(indices[0])):
                text_index = indices[0][i]
                if text_index < len(self.faiss_texts):
                    dist = float(distances[0][i])
                    similarity = 1.0 / (1.0 + max(dist, 0.0))
                    
                    result_text = self.faiss_texts[text_index]
                    
                    # VERİ FORMATI DÜZELTMESİ
                    if isinstance(result_text, dict):
                        # Dict ise 'text' anahtarını kullan
                        clean_text = result_text.get('text', str(result_text))
                    else:
                        clean_text = str(result_text)
                    
                    # YENI: Bulunan sonuçları detaylı logla
                    logging.info(f"📋 FAISS SONUÇ {i+1}: Score={similarity:.3f}")
                    logging.info(f"📄 İçerik: {clean_text[:200] if isinstance(clean_text, str) else str(clean_text)[:200]}...")
                    
                    results.append({
                        "text": clean_text,
                        "score": similarity
                    })
            
            logging.info(f"FAISS araması tamamlandı, {len(results)} sonuç bulundu.")
            return results
        except Exception as e:
            logging.error(f"FAISS arama hatası: {e}")
            return []

    def semantic_distance(self, text1, text2):
        if not self.sentence_model:
            logging.warning("Sentence transformer model not loaded, falling back to word-based check.")
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            common_words = len(words1.intersection(words2)) / max(len(words1), len(words2))
            return 1.0 - common_words
        
        embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return 1.0 - similarity

    # GÜNCELLENMİŞ coherence_check fonksiyonu
    def coherence_check(self, question, article_text, threshold=None):
        if not self.sentence_model:
            logging.warning("Sentence model yok, tutarlılık kontrolü atlanıyor.")
            return True, 0.5
        
        threshold = config.get('coherence_threshold', 0.60) if threshold is None else threshold
        
        # VERİ TİPİ KONTROLÜ EKLENDİ
        if not isinstance(article_text, str):
            if isinstance(article_text, dict):
                # Dict ise 'text' anahtarını ara
                article_text = article_text.get('text', str(article_text))
            else:
                # Diğer tipleri string'e çevir
                article_text = str(article_text)
        
        # BOŞ VERİ KONTROLÜ EKLENDİ
        if not article_text or len(article_text.strip()) < 3:
            logging.warning("Boş veya çok kısa article_text, coherence atlanıyor")
            return False, 0.0
        
        try:
            q = self.sentence_model.encode(question, convert_to_tensor=True)
            a = self.sentence_model.encode(article_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(q, a).item()
            logging.debug(f"Coherence skoru: {similarity:.2f} (threshold: {threshold})")
            return similarity >= threshold, similarity
        except Exception as e:
            logging.error(f"Coherence check hatası: {e}")
            return False, 0.0

    def fetch_from_core(self, text):
        if not self.core or not self.embedder:
            return None, None, 0
        
        try:
            # Akıllı arama - exact match + semantic similarity
            results = self.smart_neo4j_search(text, max_results=3)
            
            if not results:
                return None, None, 0
            
            best_result = results[0]
            topic_key = best_result['topic_key']
            
            # Graph bilgisi al
            graph_info = self.get_knowledge_graph(topic_key, depth=2)
            
            if graph_info:
                # Bağlantılı bilgilerle zenginleştir
                enriched_content = self.enrich_with_graph(
                    graph_info['main_content'], 
                    graph_info.get('connections', [])
                )
                
                connection_count = len([c for c in graph_info.get('connections', []) 
                                       if c.get('topic')])
                
                logging.info(f"🧠 EVOLVED DEEPKNOWLEDGE: {topic_key} "
                             f"(Quality: {best_result.get('quality_score', 0):.2f}, "
                             f"Connections: {connection_count})")
                
                return (enriched_content, 
                        best_result.get('source', 'unknown'), 
                        best_result.get('final_score', 0))
            
            return None, None, 0
            
        except Exception as e:
            logging.error(f"Enhanced Neo4j fetch error: {e}")
            return None, None, 0
            
    def fix_neo4j_schema(self):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, şema düzeltmesi atlanıyor.")
            return
        try:
            with self.core.session() as session:
                # UNIQUE constraint oluşturmayı dene
                try:
                    session.run("CREATE CONSTRAINT topic_key_unique IF NOT EXISTS FOR (n:DeepKnowledge) REQUIRE n.topic_key IS UNIQUE")
                    logging.info("✅ Unique constraint oluşturuldu veya zaten var.")
                except Exception as constraint_error:
                    if "An index exists" in str(constraint_error) or "IndexAlreadyExists" in str(constraint_error):
                        logging.warning("⚠️ Unique constraint oluşturulamadı, çakışan indeks kaldırılıyor.")
                        try:
                            # Çakışan indeksi kaldır
                            session.run("DROP INDEX topic_key_index IF EXISTS")
                            logging.info("🗑️ Çakışan topic_key_index başarıyla kaldırıldı.")
                            # Kısıtlamayı tekrar dene
                            session.run("CREATE CONSTRAINT topic_key_unique FOR (n:DeepKnowledge) REQUIRE n.topic_key IS UNIQUE")
                            logging.info("✅ Unique constraint oluşturuldu.")
                        except Exception as e:
                            logging.error(f"❌ İndeks kaldırma/constraint oluşturma hatası: {e}")
                    else:
                        logging.error(f"❌ Unique constraint oluşturulurken beklenmeyen hata: {constraint_error}")

                # Diğer indeksleri oluştur (kısıtlama içermeyen)
                session.run("CREATE INDEX content_quality_index IF NOT EXISTS FOR (n:DeepKnowledge) ON (n.quality_score)")
                logging.info("✅ content_quality_index oluşturuldu.")
                
                logging.info("✅ Neo4j şeması başarıyla düzeltildi.")
        except Exception as e:
            logging.error(f"❌ Neo4j şema düzeltme hatası: {e}")


    def create_intelligent_connections(self, session, current_topic, content):
        """Akıllı bağlantılar kur - KEŞİF SİSTEMİ"""
        
        # Mevcut tüm konuları al
        existing_topics = session.run(
            "MATCH (n:DeepKnowledge) WHERE n.topic_key <> $current "
            "RETURN n.topic_key, n.content, n.quality_score "
            "ORDER BY n.quality_score DESC LIMIT 20",
            current=current_topic
        )
        
        for record in existing_topics:
            other_topic = record['n.topic_key']
            other_content = record['n.content']
            
            # Semantic similarity ile bağlantı gücü hesapla
            connection_strength = self.calculate_connection_strength(content, other_content)
            
            if connection_strength > 0.3:  # Eşik değer
                # BAĞLANTI OLUŞTUR
                relationship_type = self.determine_relationship_type(content, other_content)
                
                session.run(
                    "MATCH (a:DeepKnowledge {topic_key: $topic1}) "
                    "MATCH (b:DeepKnowledge {topic_key: $topic2}) "
                    "MERGE (a)-[r:" + relationship_type + "]->(b) "
                    "SET r.strength = $strength, "
                    "    r.discovery_date = timestamp(), "
                    "    r.type = $rel_type",
                    topic1=current_topic, topic2=other_topic, 
                    strength=connection_strength, rel_type=relationship_type
                )
                
                logging.info(f"🔗 CONNECTION CREATED: {current_topic} --{relationship_type}--> {other_topic} (Strength: {connection_strength:.2f})")

    def calculate_connection_strength(self, content1, content2):
        """İki içerik arasındaki bağlantı gücünü hesapla"""
        if not self.sentence_model:
            # Fallback: kelime overlap
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2), 1)
            
        # Semantic similarity
        embeddings = self.sentence_model.encode([content1, content2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity

    def determine_relationship_type(self, content1, content2):
        """Bağlantı tipini belirle"""
        # Basit keyword-based relationship detection
        if any(word in content1.lower() and word in content2.lower()
               for word in ['cause', 'result', 'because', 'due to']):
            return 'CAUSES'
        elif any(word in content1.lower() and word in content2.lower()
                             for word in ['similar', 'like', 'same', 'also']):
            return 'SIMILAR_TO'
        elif any(word in content1.lower() and word in content2.lower()
                             for word in ['part of', 'contains', 'includes']):
            return 'CONTAINS'
        else:
            return 'RELATED_TO'
            
    def enrich_with_connections(self, main_content, connections):
        """Ana içeriği bağlantılarla zenginleştir"""
        if not connections or not any(c['content'] for c in connections):
            return main_content
            
        enriched = main_content
        # Güçlü bağlantıları ekle
        strong_connections = [c for c in connections if c.get('strength', 0) > 0.5 and c.get('content')]
        
        if strong_connections:
            connection_texts = []
            for conn in strong_connections[:3]:  # En güçlü 3 bağlantı
                rel_type = conn.get('type', 'RELATED_TO')
                if rel_type == 'CAUSES':
                    connection_texts.append(f"Bu konu {conn['topic']} konusunu etkiler: {conn['content'][:100]}...")
                elif rel_type == 'SIMILAR_TO':
                    connection_texts.append(f"Benzer konu {conn['topic']}: {conn['content'][:100]}...")
                else:
                    connection_texts.append(f"İlgili konu {conn['topic']}: {conn['content'][:100]}...")
            
            if connection_texts:
                enriched += f"\n\n--- BAĞLANTILI BİLGİLER ---\n" + "\n".join(connection_texts)
        return enriched
        
    def discover_patterns(self):
        """Beklenmedik bağlantılar ve pattern'ler keşfet"""
        if not self.core:
            return []
            
        with self.core.session() as session:
            # Uzak bağlantıları keşfet (2-3 hop away)
            discoveries = session.run(
                "MATCH path = (a:DeepKnowledge)-[*2..3]-(b:DeepKnowledge) "
                "WHERE a.topic_key <> b.topic_key "
                "AND length(path) >= 2 "
                "RETURN DISTINCT a.topic_key as topic1, b.topic_key as topic2, "
                "      length(path) as distance, "
                "      [node in nodes(path) | node.topic_key] as path_topics "
                "ORDER BY distance DESC "
                "LIMIT 10"
            )
            
            patterns = []
            for record in discoveries:
                patterns.append({
                    'topic1': record['topic1'],
                    'topic2': record['topic2'], 
                    'distance': record['distance'],
                    'path': record['path_topics'],
                    'discovery_type': 'INDIRECT_CONNECTION'
                })
            
            logging.info(f"🔍 PATTERNS DISCOVERED: {len(patterns)} new connections found!")
            return patterns

    def wikipedia_verify(self, text):
        if not config['external_sources']['wikipedia']:
            logging.info("Wikipedia verification disabled in config.")
            return False, None, None, 0
        
        try:
            wikipedia.set_lang(config['external_sources'].get('wikipedia_lang', 'en'))
            search_text = self.extract_smart_keywords(text)
            
            # Sorguyu İngilizce'ye çevir
            try:
                search_text_en = GoogleTranslator(source='auto', target='en').translate(search_text)
                logging.info(f"🌐 WİKİPEDİA ARAMASI: '{search_text}' → '{search_text_en}' (dil: en)")
            except Exception as e:
                search_text_en = search_text
                logging.info(f"🌐 WİKİPEDİA ARAMASI: '{search_text}' (çeviri hatası: {e})")

            results = wikipedia.search(search_text_en, results=10)
            logging.info(f"🔍 Wikipedia {len(results)} sonuç buldu: {results[:5]}")

            for result in results[:3]:
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    if "disambiguation" in page.title.lower():
                        logging.info(f"⏭️ '{result}' disambiguation sayfası, atlanıyor")
                        continue

                    summary = wikipedia.summary(result, sentences=5, auto_suggest=False).lower()
                    ok, coherence_score = self.coherence_check(text, summary)
                    logging.info(f"📊 '{result}' coherence: {coherence_score:.2f} (ok={ok})")

                    if ok:
                        logging.info(f"✅ WİKİPEDİA SONUÇ BULUNDU!")
                        logging.info(f"📝 Wikipedia başlık: {result}")
                        logging.info(f"📄 Wikipedia özet (ilk 500 karakter): {summary[:500]}")
                        logging.info(f"🎯 Wikipedia coherence: {coherence_score:.2f}")

                        return True, f"Wikipedia - {result}", summary, 8
                        
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    logging.warning(f"Wikipedia page error: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Wikipedia verification failed: {str(e)}")
        
        logging.warning(f"❌ WİKİPEDİA'DA UYGUN VERİ BULUNAMADI")
        return False, None, None, 0

    def pubmed_verify(self, text):
        logging.info("PubMed verification disabled (Biopython dependency)")
        return False, None, None, 0

    def fetch_from_external_sources(self, text):
        logging.info(f"Fetching external sources for: {text}")
        external_infos = []
        verified_infos = []
        
        valid_wiki, wiki_source, wiki_data, wiki_score = self.wikipedia_verify(text)
        if valid_wiki and wiki_data:
            external_infos.append((wiki_data.lower(), wiki_source, True, wiki_score))
            verified_infos.append((wiki_data.lower(), wiki_source, wiki_score))
            logging.info(f"Added Wikipedia data to verified_infos: {wiki_source}")
        
        return external_infos, verified_infos

    def llama_generate(self, text, model_prompt=None, force_json=False):
        # Config kontrolü - hem eski hem yeni config anahtarlarını destekle
        llm_config = config.get('llm_generation', config.get('generation', {}))
        if not llm_config.get('enabled', True):
            logging.info("LLM generation disabled in config.")
            return "LLM generation is disabled.", 0

        # Provider seçimi: "together" veya "ollama"
        provider = config.get("llm_provider", "ollama")

        try:
            logging.info(f"🤖 LLM GENERATE BAŞLIYOR (Provider: {provider}, Force JSON: {force_json})")

            # Prompt hazırla
            if force_json:
                system_prompt = "Sen sadece ve kesinlikle JSON döndüren bir asistansın. Herhangi bir açıklama, giriş, çıkış veya ek metin kullanma. Markdown formatı yok, sadece geçerli bir JSON objesi döndür."
                prompt = f"{system_prompt}\n\nKonu: {text}\n\nTalimatlar: {model_prompt}\n\nJSON:"
            else:
                # model_prompt=None ise genel prompt kullan, "" ise sadece text kullan
                if model_prompt is None:
                    prompt = f"{self.model_prompt} {text}"
                elif model_prompt == "":
                    prompt = text  # Thinking framework için sadece step talimatları
                else:
                    prompt = f"{model_prompt} {text}"

            logging.info(f"📝 LLM Prompt (ilk 200 karakter): {prompt[:200]}")

            # Provider'a göre API çağır
            if provider == "together":
                # Together AI (Llama 405B)
                generated_text, score = together_generate(prompt, config)
            else:
                # Ollama (varsayılan)
                with llama_lock:
                    if force_json:
                        options = {
                            "temperature": 0.0,
                            "num_predict": llm_config.get('max_tokens', 4096),
                            "top_p": 0.9,
                            "stop": ["\n\n", "```", "##"],
                            "repeat_penalty": 1.1,
                            "num_ctx": llm_config.get('num_ctx', 32768)
                        }
                    else:
                        options = {
                            "temperature": llm_config.get('temperature', 0.75),
                            "num_predict": llm_config.get('max_tokens', 4096),
                            "top_p": llm_config.get('top_p', 0.85),
                            "num_ctx": llm_config.get('num_ctx', 32768)
                        }

                    response = ollama.generate(
                        model=config.get("ollama", {}).get("model_name", "gemma3:27b"),
                        prompt=prompt,
                        options=options
                    )
                    generated_text = response['response'].strip()
                    score = 3

            logging.info(f"✅ LLM YANIT ALINDI!")
            logging.info(f"📄 LLM tam yanıt: {generated_text}")
            logging.info(f"🎯 LLM score: {score}")

            # LLM yanıtını Neo4j'ye kaydet
            if hasattr(self, 'current_query_text') and self.current_query_text:
                try:
                    self.save_to_core(
                        f"Query: {self.current_query_text}\n\nResponse: {generated_text}",
                        "LLM_Generated",
                        is_valid=True,
                        score=3.0
                    )
                    logging.info("📚 LLM yanıtı Neo4j'ye kaydedildi")
                except Exception as e:
                    logging.error(f"LLM response storage error: {e}")

            return generated_text, score
        except Exception as e:
            logging.error(f"❌ LLM HATASI: {str(e)}")
            return f"Error: {str(e)}", 0

    def fetch_from_libraries(self, query):
        results = []
        
        if not self.local_knowledge.get("sources"):
            logging.info("Local knowledge sources boş")
            return results
            
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        for source in self.local_knowledge["sources"]:
            source_name = source.get("name", "Unknown").lower()
            if source_name in config["external_sources"] and not config["external_sources"][source_name]:
                logging.info(f"{source['name']} disabled in config.")
                continue
            
            try:
                base_url = source.get("base_url", "")
                params = source.get("params", {})
                
                if not base_url:
                    continue
                
                query_params = {key: value.replace("{query}", query) for key, value in params.items()}
                response = requests.get(base_url, params=query_params, headers=headers, timeout=10)
                time.sleep(random.uniform(0.5, 1))
                
                if response.status_code == 200:
                    if source.get("access_method") == "Web Scraping":
                        soup = BeautifulSoup(response.text, 'html.parser')
                        content = soup.find('article') and soup.find('article').get_text()[:300] or "No data found"
                    else:
                        try:
                            data = response.json()
                            path = source.get("response_processing", {}).get("path", "")
                            content = "No data found" if not path else self.extract_from_json(data, path)
                        except json.JSONDecodeError:
                            content = "No data found"
                    
                    ok, coherence_score = self.coherence_check(query, content)
                    if ok:
                        results.append({"source": source["name"], "content": content, "score": coherence_score})
                        
            except Exception as e:
                logging.error(f"Library fetch error {source.get('name')}: {str(e)}")
        
        return results

    def extract_from_json(self, data, path):
        default = "No data found"
        keys = path.split(".")
        temp = data
        
        try:
            for key in keys:
                if key.endswith("]"):
                    key, index = key.split("[")
                    index = int(index[:-1])
                    temp = temp.get(key, [])[index] if isinstance(temp.get(key), list) else default
                else:
                    temp = temp.get(key, default)
        except (KeyError, IndexError, ValueError):
            return default
        
        return temp if temp != default else default

    def get_step_prompt(self, step_name, text, context=""):
        if not self.thinking_framework:
            return f"Analyze {text} for {step_name}"
        
        step_data = None
        for step in self.thinking_framework.get("steps", []):
            if step["name"] == step_name:
                step_data = step
                break
        
        if not step_data:
            return f"Analyze {text} for {step_name}"
        
        description = step_data.get("description", "")
        instructions = step_data.get("instructions", [])
        output_format = step_data.get("output_format", {})
        
        # Doğal metin için sadeleştirilmiş prompt
        prompt = f"""Konu: {text}
Bağlam: {context}
Görev: {description}
Talimatlar:
{chr(10).join([f"• {inst}" for inst in instructions])}
Türkçe yanıt ver."""
        return prompt

    def clean_and_parse_json(self, raw_output):
        """LLM'den gelen JSON'u temizle ve parse et, fallbackler dahil"""
        # 1. Markdown JSON bloklarını temizle
        if "```json" in raw_output:
            start_idx = raw_output.find("```json") + 7
            end_idx = raw_output.rfind("```")
            if end_idx > start_idx:
                raw_output = raw_output[start_idx:end_idx].strip()
        
        # 2. JSON'u bulmak için regex fallback'i
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            raw_output = match.group(0)
        
        # 3. Son temizlik
        raw_output = raw_output.strip()
        
        # 4. JSON parse et
        try:
            parsed_data = json.loads(raw_output)
            logging.debug("✅ JSON başarıyla parse edildi.")
            return parsed_data
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON parse hatası: {e}")
            logging.debug(f"Hatalı JSON içeriği: {raw_output[:500]}...")
            return {"error": "Parse edilemedi", "raw_content": raw_output}
    
    def parse_opposites_output(self, raw_output):
        # Doğal metni parse et - madde işaretli veya numaralı listeler
        opposites = []
        lines = raw_output.split('\n')

        for line in lines:
            line = line.strip()
            # Numaralı liste (1. 2. vs) veya madde işareti (*, -)
            if re.match(r'^[\d]+\.\s*\*?\*?|^[\*\-]\s*\*?\*?', line):
                # Başlık kısmını temizle (bold markdown vs.)
                content = re.sub(r'^[\d]+\.\s*\*?\*?|^[\*\-]\s*\*?\*?', '', line)
                content = re.sub(r'\*\*([^*]+)\*\*:?\s*', r'\1: ', content)  # **bold**: -> bold:
                content = content.strip()
                if len(content) > 30:
                    opposites.append(content)

        # Eğer liste bulunamadıysa, paragraf bazlı dene
        if len(opposites) < 2:
            paragraphs = [p.strip() for p in raw_output.split('\n\n') if p.strip() and len(p.strip()) > 50]
            opposites = paragraphs[:4]

        return opposites[:4] if len(opposites) >= 2 else ["Varsayılan karşıt görüş 1", "Varsayılan karşıt görüş 2"]

    def parse_potentials_output(self, raw_output):
        # Doğal metni parse et
        scenarios = []
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        for line in lines:
            # Liste formatı (-, *, sayı)
            if re.match(r'^[\*\-]\s*|^[\d]+\.\s*', line):
                scenario_text = re.sub(r'^[\*\-]\s*|^[\d]+\.\s*', '', line)
            # Düz metinden senaryo çıkarma
            else:
                sentences = re.split(r'[.!?]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 30:
                        scenarios.append({"scenario": sentence, "probability": random.uniform(0.1, 0.9)})
        return scenarios[:5] if scenarios else [{"scenario": "Varsayılan potansiyel senaryo.", "probability": 0.5}]

    def parse_wisdom_output(self, raw_output):
        # Doğal metni parse et
        insights = []
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        for line in lines:
            # Liste formatı (-, *, sayı)
            if re.match(r'^[\*\-]\s*|^[\d]+\.\s*', line):
                insight_text = re.sub(r'^[\*\-]\s*|^[\d]+\.\s*', '', line)
                if len(insight_text) > 30:
                    insights.append(insight_text)
        return insights[:4] if insights else ["Varsayılan bilgelik içgörüsü."]

    def parse_holistic_meaning_output(self, raw_output):
        # Doğal metin zaten bir bütün olduğu için doğrudan döndür
        return raw_output.strip() if raw_output.strip() else "Bütüncül bir anlamlandırma yapılamadı."

    def parse_divine_order_output(self, raw_output):
        # Doğal metni parse et
        timeline = []
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        for line in lines:
            # Liste formatı (-, *, sayı)
            if re.match(r'^[\*\-]\s*|^[\d]+\.\s*', line):
                stage_text = re.sub(r'^[\*\-]\s*|^[\d]+\.\s*', '', line)
                timeline.append({"stage": f"Aşama {len(timeline)+1}", "description": stage_text, "timestamp": "Geçmişten Geleceğe"})
        return timeline[:4] if timeline else [{"stage": "Varsayılan Aşama", "description": "Zaman çizelgesi oluşturulamadı.", "timestamp": "Bilinmiyor"}]

    def parse_zero_point_output(self, raw_output):
        # Basit yaklaşım - tüm çıktıyı kullan, içerik kaybetme!
        # Sadece "philosophical" kullanıldığı için direkt tüm içeriği al
        content = raw_output.strip()

        # Çok uzunsa ilk 1500 karakteri al
        if len(content) > 1500:
            content = content[:1500]

        return {"technical": content, "philosophical": content}

    def parse_recursive_questioning_output(self, raw_output):
        # Doğal metni parse et
        questions = []
        lines = raw_output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Numaralı liste formatı (1. 2. 3.)
            if re.match(r'^\d+\.', line):
                question = re.sub(r'^\d+\.\s*', '', line)
                if len(question) > 10:  # Çok kısa olanları atla
                    questions.append(question)
            # Soru işareti ile biten satırlar
            elif line.endswith('?') and len(line) > 15:
                questions.append(line)
        return questions[:3] if questions else ["Varsayılan derin soru oluşturuldu."]

    def parse_speculative_theory_output(self, raw_output):
        # Doğal metni parse et, varsayılan değerler ata
        hypothesis = raw_output.strip() if raw_output.strip() else "Varsayılan spekülatif hipotez."
        return [{"hypothesis": hypothesis}]

    def parse_truth_void_output(self, raw_output):
        # Doğal metni parse et, varsayılan değerler ata
        truth = "Varsayılan temel hakikat."
        void = "Varsayılan temel boşluk/yanılgı."
        lines = raw_output.split('\n')
        
        truth_found = False
        void_found = False
        
        for line in lines:
            line_lower = line.lower()
            if "hakikat" in line_lower or "gerçek" in line_lower:
                truth = line.split(":", 1)[-1].strip()
                truth_found = True
            elif "boşluk" in line_lower or "yanılgı" in line_lower:
                void = line.split(":", 1)[-1].strip()
                void_found = True
            elif truth_found and not void_found and len(line) > 20:
                void = line.strip()
                void_found = True
            elif void_found and not truth_found and len(line) > 20:
                truth = line.strip()
                truth_found = True
                
        return {"truth": {"core_truth": truth}, "void": {"illusion": void}}
            
    def apply_thinking_framework(self, text, verified_info=None):
        logging.info(f"🧠 THİNKİNG FRAMEWORK BAŞLIYOR...")
        # DEBUG kodları ekle:
        logging.info(f"DEBUG: self.thinking_framework var mı? {self.thinking_framework is not None}")
        if self.thinking_framework:
            steps = self.thinking_framework.get("steps", [])
            logging.info(f"DEBUG: steps listesi: {len(steps)} adet step bulundu")
            for i, step in enumerate(steps[:3]):  # İlk 3 step'i logla
                logging.info(f"DEBUG: Step {i}: {step.get('name', 'NO_NAME')}")
        else:
            logging.error("DEBUG: thinking_framework YOK!")
        
        if not self.thinking_framework or not config['thinking_framework']['enabled']:
            logging.error("Thinking framework disabled or not loaded.")
            return {"text": text, "status": "Error", "thought_process": []}
        
        steps = self.thinking_framework.get("steps", [])
        thought_process = []
        verified_info = verified_info or text
        
        for step in steps:
            step_name = step.get("name", "unknown_step")
            if not config['thinking_framework']['steps_enabled'].get(step_name, True):
                continue
                
            logging.info(f"🔄 STEP İŞLENİYOR: {step_name}")
            result = {"step": step_name, "output": None}
            raw_output = ""  # Tanımla ki exception durumunda hata vermesin

            try:
                prompt = self.get_step_prompt(step_name, text, context=verified_info)
                # DEĞİŞİKLİK 1: model_prompt="" ile sadece step talimatları kullanılır
                raw_output, _ = self.llama_generate(prompt, model_prompt="", force_json=False)
                
                # Her step için özel parse ve fallback yönetimi
                if step_name == "opposites":
                    parsed_data = self.parse_opposites_output(raw_output)
                    result["output"] = {"opposites": parsed_data}
                    self.save_opposites_to_core(verified_info, parsed_data)

                elif step_name == "quantum_field":
                    parsed_data = self.parse_potentials_output(raw_output)
                    result["output"] = {"potentials": parsed_data}
                    self.save_potentials_to_core(verified_info, parsed_data)
                    
                elif step_name == "zero_point":
                    parsed_data = self.parse_zero_point_output(raw_output)
                    result["output"] = parsed_data
                    self.save_zero_point_to_core(verified_info, parsed_data)
                
                elif step_name == "wisdom":
                    parsed_data = self.parse_wisdom_output(raw_output)
                    result["output"] = {"wisdom_insights": parsed_data}
                    self.save_wisdom_to_core(verified_info, parsed_data)
                
                elif step_name == "holistic_meaning":
                    combined_context = self.extract_context_for_holistic_meaning(thought_process)
                    prompt = self.get_step_prompt("holistic_meaning", text, context=combined_context)
                    # DEĞİŞİKLİK 1 (İKİNCİ YER)
                    raw_output, _ = self.llama_generate(prompt, force_json=False)
                    parsed_data = self.parse_holistic_meaning_output(raw_output)
                    result["output"] = {"holistic_meaning": parsed_data}
                    self.save_holistic_meaning_to_core(verified_info, parsed_data)
                
                elif step_name == "divine_order":
                    parsed_data = self.parse_divine_order_output(raw_output)
                    result["output"] = {"timeline": parsed_data}
                    self.save_timeline_to_core(verified_info, parsed_data)
                
                elif step_name == "recursive_questioning":
                    parsed_data = self.parse_recursive_questioning_output(raw_output)
                    result["output"] = {"recursive_questions": parsed_data}
                    if parsed_data:
                        # DÜZELTME 2
                        self.save_to_core(parsed_data[0], "RecursiveQuestion Insight", is_valid=True, score=0.8)
                
                elif step_name == "speculative_theory":
                    parsed_data = self.parse_speculative_theory_output(raw_output)
                    result["output"] = {"theories": parsed_data}
                    self.save_speculative_theory_to_core(verified_info, parsed_data)
                
                elif step_name == "truth_void":
                    parsed_data = self.parse_truth_void_output(raw_output)
                    result["output"] = parsed_data
                    if parsed_data:
                        self.save_truth_void_to_core(verified_info, parsed_data)
                
                else:
                    logging.warning(f"Unknown thinking step: {step_name}")
                    result["output"] = {"message": f"Step {step_name} not implemented"}
            
            except Exception as e:
                logging.error(f"❌ Düşünce adımında KRİTİK HATA ({step_name}): {str(e)}")
                result["output"] = {"error": f"Adım sırasında hata oluştu: {str(e)}", "raw_input": raw_output}
                
            thought_process.append(result)
            logging.info(f"✅ STEP TAMAMLANDI: {step_name}")
            logging.info(f"📄 {step_name} çıktısı:\n{str(result['output'])}")
            
        logging.info(f"Thinking framework tamamlandı: {len(thought_process)} step processed")
        return {"text": text, "status": "ThoughtProcessed", "thought_process": thought_process}
        
    def extract_context_for_holistic_meaning(self, thought_process):
        context = ""
        for step in thought_process:
            if step["step"] != "holistic_meaning":
                output = step.get("output", {})
                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, list):
                            context += f" {key}: " + " ".join([str(item) for item in value])
                        elif isinstance(value, dict):
                            context += f" {key}: " + " ".join([str(v) for v in value.values()])
                        else:
                            context += f" {key}: {value}"
                else:
                    context += f" {step['step']}: {output}"
        return context

    def save_opposites_to_core(self, text, opposites):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return
        
        if not opposites or not isinstance(opposites, list):
            logging.warning("Boş veya geçersiz zıtlıklar kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                for opp in opposites:
                    if isinstance(opp, dict) and 'opposite' in opp:
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (o:OppositeKnowledge {text: $opposite}) "
                               "MERGE (n)-[:OPPOSES]->(o)",
                               text=text.lower(), opposite=opp['opposite'].lower())
                    elif isinstance(opp, str):
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (o:OppositeKnowledge {text: $opposite}) "
                               "MERGE (n)-[:OPPOSES]->(o)",
                               text=text.lower(), opposite=opp.lower())
                tx.commit()
                logging.debug("Zıtlıklar Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye zıtlık kaydederken hata oluştu: {e}")


    def save_potentials_to_core(self, text, potentials):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return

        if not potentials or not isinstance(potentials, list):
            logging.warning("Boş veya geçersiz potansiyeller kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                for pot in potentials:
                    if isinstance(pot, dict) and 'scenario' in pot:
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (p:PotentialScenario {text: $scenario, probability: $probability}) "
                               "MERGE (n)-[:HAS_POTENTIAL]->(p)",
                               text=text.lower(), scenario=pot.get("scenario", "N/A").lower(),
                               probability=pot.get("probability", 0.0))
                tx.commit()
                logging.debug("Potansiyeller Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye potansiyelleri kaydederken hata oluştu: {e}")

    def save_zero_point_to_core(self, text, zero_point):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return
            
        if not zero_point or not isinstance(zero_point, dict):
            logging.warning("Boş veya geçersiz sıfır noktası kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                session.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                            "MERGE (z:ZeroPoint {technical: $technical, philosophical: $philosophical}) "
                            "MERGE (n)-[:ORIGIN]->(z)",
                            text=text.lower(), technical=zero_point.get("technical", "N/A"),
                            philosophical=zero_point.get("philosophical", "N/A"))
            logging.debug("Sıfır noktası Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye sıfır noktası kaydederken hata oluştu: {e}")

    def save_wisdom_to_core(self, text, wisdom_insights):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return

        if not wisdom_insights or not isinstance(wisdom_insights, list):
            logging.warning("Boş veya geçersiz bilgelik içgörüleri kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                for insight in wisdom_insights:
                    if isinstance(insight, dict) and 'insight' in insight:
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (w:WisdomInsight {text: $insight}) "
                               "MERGE (n)-[:HAS_WISDOM]->(w)",
                               text=text.lower(), insight=insight['insight'].lower())
                    elif isinstance(insight, str):
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (w:WisdomInsight {text: $insight}) "
                               "MERGE (n)-[:HAS_WISDOM]->(w)",
                               text=text.lower(), insight=insight.lower())
                tx.commit()
                logging.debug("Bilgelik içgörüleri Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye bilgelik içgörüleri kaydederken hata oluştu: {e}")

    def save_holistic_meaning_to_core(self, text, meaning):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return
            
        if not meaning or not isinstance(meaning, str):
            logging.warning("Boş veya geçersiz bütüncül anlam kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                session.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                            "MERGE (m:HolisticMeaning {text: $meaning}) "
                            "MERGE (n)-[:HAS_MEANING]->(m)",
                            text=text.lower(), meaning=meaning.lower())
            logging.debug("Bütüncül anlam Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye bütüncül anlam kaydederken hata oluştu: {e}")

    def save_timeline_to_core(self, text, timeline):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return

        if not timeline or not isinstance(timeline, list):
            logging.warning("Boş veya geçersiz zaman çizelgesi kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                for stage in timeline:
                    if isinstance(stage, dict) and 'stage' in stage:
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (t:TimelineStage {stage: $stage, description: $description, timestamp: $timestamp}) "
                               "MERGE (n)-[:PART_OF]->(t)",
                               text=text.lower(), stage=stage['stage'],
                               description=stage.get('description', 'N/A').lower(),
                               timestamp=stage.get('timestamp', 'N/A'))
                tx.commit()
                logging.debug("Zaman çizelgesi Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye zaman çizelgesi kaydederken hata oluştu: {e}")

    def save_speculative_theory_to_core(self, text, theories):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return

        if not theories or not isinstance(theories, list):
            logging.warning("Boş veya geçersiz spekülatif teori kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                for theory in theories:
                    if isinstance(theory, dict) and 'hypothesis' in theory:
                        tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                               "MERGE (t:SpeculativeTheory {hypothesis: $hypothesis}) "
                               "MERGE (n)-[:HAS_THEORY]->(t)",
                               text=text.lower(), hypothesis=theory['hypothesis'].lower())
                tx.commit()
                logging.debug("Spekülatif teori Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye spekülatif teori kaydederken hata oluştu: {e}")

    def save_truth_void_to_core(self, text, void_data):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, kaydetme atlanıyor")
            return

        if not void_data or not isinstance(void_data, dict):
            logging.warning("Boş veya geçersiz hakikat/boşluk verisi kaydedilemiyor.")
            return

        try:
            with self.core.session() as session:
                tx = session.begin_transaction()
                truth = void_data.get("truth", {})
                void = void_data.get("void", {})
                
                if truth and "core_truth" in truth:
                    tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                           "MERGE (t:CoreTruth {core_truth: $truth_text}) "
                           "MERGE (n)-[:HAS_TRUTH]->(t)",
                           text=text.lower(), truth_text=truth['core_truth'].lower())
                
                if void and "illusion" in void:
                    tx.run("MERGE (n:DeepKnowledge {topic_key: $text}) "
                           "MERGE (v:TruthVoid {illusion: $void_text}) "
                           "MERGE (n)-[:HAS_VOID]->(v)",
                           text=text.lower(), void_text=void['illusion'].lower())
                
                tx.commit()
                logging.debug("Hakikat ve Boşluk Neo4j'ye kaydedildi: %s", text[:100])
        except Exception as e:
            logging.error(f"Neo4j'ye hakikat ve boşluk kaydederken hata oluştu: {e}")
            
    def extract_topic_key(self, text):
        """Metinden benzersiz bir anahtar kelime/cümle oluştur"""
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = cleaned_text.split()
        return "_".join(words[:5]) if len(words) > 5 else "_".join(words) if words else "unknown"

    def collect_branches_enhanced(self, result):
        logging.debug("BRANCH COLLECTION BAŞLADI - Thought process step sayısı: %d", len(result.get("thought_process", [])))
        branches = []
        for step in result.get("thought_process", []):
            step_name = step.get("step", "unknown")
            output = step.get("output")
            
            logging.debug("STEP İŞLENİYOR: %s", step_name)
            logging.debug("OUTPUT VAR MI: %s", output is not None)
            logging.debug("OUTPUT TİPİ: %s", type(output))

            if not output or "error" in str(output).lower():
                logging.debug("STEP ATLANDI: %s - Reason: Output boş veya hata içeriyor", step_name)
                continue
            
            if step_name == "quantum_field":
                potentials = output.get("potentials", [])
                logging.debug("Potansiyel item sayısı: %d", len(potentials))
                for potential in potentials:
                    scenario = potential.get("scenario", "")
                    if len(scenario) > 50:
                        # Thinking framework için benzersizlik kontrolü DEVRE DIŞI - her perspektif değerli
                        branches.append({"type": "future_applications", "content": scenario, "weight": 0.9})
                        logging.debug("Potansiyel eklendi. Uzunluk: %d. İçerik: %s...", len(scenario), scenario[:100])
            
            elif step_name == "recursive_questioning":
                questions = output.get("recursive_questions", [])
                logging.debug("Soru item sayısı: %d", len(questions))
                for question in questions:
                    if len(question) > 50:
                        branches.append({"type": "recursive_questioning", "content": question, "weight": 0.8})
                        logging.debug("Soru eklendi. Uzunluk: %d. İçerik: %s...", len(question), question[:100])
            
            elif step_name == "wisdom":
                wisdom_insights = output.get("wisdom_insights", [])
                logging.debug("Bilgelik item sayısı: %d", len(wisdom_insights))
                for insight in wisdom_insights:
                    insight_text = str(insight)
                    if len(insight_text) > 50:
                        branches.append({"type": "wisdom_insights", "content": insight_text, "weight": 0.8})
                        logging.debug("İçgörü eklendi. Uzunluk: %d. İçerik: %s...", len(insight_text), insight_text[:100])
            
            elif step_name == "zero_point":
                philosophical = output.get("philosophical", "")
                if philosophical and len(philosophical) > 50:
                    branches.append({"type": "core_essence", "content": philosophical, "weight": 0.7})
                    logging.debug("Felsefi öz eklendi. Uzunluk: %d. İçerik: %s...", len(philosophical), philosophical[:100])
            
            elif step_name == "divine_order":
                timeline = output.get("timeline", [])
                if timeline:
                    timeline_content = " ".join([stage.get("description", "") for stage in timeline if isinstance(stage, dict)])
                    if len(timeline_content) > 50:
                        branches.append({"type": "divine_order", "content": timeline_content, "weight": 0.6})
                        logging.debug("Zaman çizelgesi eklendi. Uzunluk: %d. İçerik: %s...", len(timeline_content), timeline_content[:100])
            
            elif step_name == "opposites":
                opposites = output.get("opposites", [])
                logging.debug("Zıtlık item sayısı: %d", len(opposites))
                if len(opposites) >= 2:
                    opp1 = str(opposites[0])
                    opp2 = str(opposites[1])
                    content = f"Birinci Bakış Açısı: {opp1}\n\nKarşıt Bakış Açısı: {opp2}"
                    branches.append({"type": "opposite_viewpoints", "content": content, "weight": 0.9})
                    logging.debug("Zıtlıklar eklendi. Uzunluk: %d. İçerik: %s...", len(content), content[:100])
            
            elif step_name == "holistic_meaning":
                meaning = output.get("holistic_meaning", "")
                if meaning and len(meaning) > 100:
                    branches.append({"type": "holistic_meaning", "content": meaning, "weight": 1.0})
                    logging.debug("Bütüncül anlam eklendi. Uzunluk: %d. İçerik: %s...", len(meaning), meaning[:100])
            
            elif step_name == "speculative_theory":
                theories = output.get("theories", [])
                logging.debug("Spekülatif teori item sayısı: %d", len(theories))
                for theory in theories:
                    hypothesis = theory.get("hypothesis", "")
                    if hypothesis and len(hypothesis) > 50:
                        branches.append({"type": "speculative_theory", "content": hypothesis, "weight": 0.95})
                        logging.debug("Spekülatif teori eklendi. Uzunluk: %d. İçerik: %s...", len(hypothesis), hypothesis[:100])
            
            elif step_name == "truth_void":
                truth = output.get("truth", {}).get("core_truth", "")
                void = output.get("void", {}).get("illusion", "")
                if truth and len(truth) > 50:
                    branches.append({"type": "truth_void", "content": truth, "weight": 0.85})
                    logging.debug("Hakikat eklendi. Uzunluk: %d. İçerik: %s...", len(truth), truth[:100])
                if void and len(void) > 50:
                    branches.append({"type": "truth_void_illusion", "content": void, "weight": 0.85})
                    logging.debug("Boşluk eklendi. Uzunluk: %d. İçerik: %s...", len(void), void[:100])
            
        logging.debug("BRANCH COLLECTION BİTTİ - Toplam branch: %d", len(branches))
        for branch in branches:
            logging.debug("Branch - Tip: %s, Weight: %.2f, İçerik Uzunluğu: %d", branch['type'], branch['weight'], len(branch['content']))
        
        return branches

    def is_content_unique(self, content, existing_content, category_type="default"):
        """
        Geliştirilmiş içerik tekillik kontrolü.
        Korumalı kategoriler ve anahtar kelimeler için farklı eşikler kullanır.
        """
        logging.debug("BENZERSİZLİK KONTROLÜ BAŞLADI: %s", category_type)
        logging.debug("Mevcut içerik sayısı: %d", len(existing_content))
        
        base_threshold = config.get('similarity_threshold', 0.85)
        
        protected_categories = config.get('thinking_framework', {}).get('protected_categories', [])
        is_protected = category_type in protected_categories
        logging.debug("Korumalı kategori mi?: %s", is_protected)
        
        if is_protected:
            threshold = config.get('thinking_framework', {}).get('similarity_threshold_protected', 0.65)
            logging.debug("🛡️ Korumalı kategori için eşik: %.2f", threshold)
        else:
            threshold = base_threshold
            logging.debug("Normal kategori için eşik: %.2f", threshold)
            
        keywords = config.get('thinking_framework', {}).get('content_preservation_keywords', [])
        if any(keyword in content.lower() for keyword in keywords):
            threshold = threshold * 0.8
            logging.debug("🔑 Anahtar kelime koruması aktif. Yeni eşik: %.2f", threshold)
        
        if not self.sentence_model:
            logging.warning("Sentence transformer model yok. Kelime tabanlı kontrol kullanılıyor.")
            words1 = set(content.lower().split())
            for existing_item in existing_content:
                words2 = set(existing_item.lower().split())
                if len(words1) > 3 and len(words2) > 3:
                    overlap = len(words1 & words2)
                    similarity = overlap / max(len(words1), len(words2))
                    if similarity > threshold:
                        logging.debug("Kelime tabanlı benzerlik skoru: %.2f, BENZER", similarity)
                        return False
            logging.debug("Kelime tabanlı kontrol sonucu: BENZERSİZ")
            return True
        
        content_sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
        if not content_sentences:
            logging.debug("Cümle bulunamadı, BENZERSİZ olarak kabul edildi.")
            return True
        
        new_embeddings = self.sentence_model.encode(content_sentences, convert_to_tensor=True)
        max_similarity_score = 0
        
        for existing_item in existing_content:
            existing_sentences = [s.strip() for s in existing_item.split('.') if len(s.strip()) > 15]
            if not existing_sentences:
                continue
                
            existing_embeddings = self.sentence_model.encode(existing_sentences, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(new_embeddings, existing_embeddings)
            current_max_sim = torch.max(similarities)
            if current_max_sim > max_similarity_score:
                max_similarity_score = current_max_sim
            
            if current_max_sim > threshold:
                logging.debug("En yüksek benzerlik skoru: %.2f. Final karar: BENZER", current_max_sim)
                return False
                
        logging.debug("En yüksek benzerlik skoru: %.2f. Final karar: BENZERSİZ", max_similarity_score)
        return True

    def create_enhanced_final_synthesis(self, branches, original_question, attempt=1):
        """
        Gerçek Sentez Yaklaşımı:
        Thinking framework'ten üretilen TÜM içerikler toplanır ve LLM bunları
        düzgünce birleştirip tutarlı bir final yanıt üretir.
        """
        logging.info("🔄 SENTEZ BAŞLADI - Deneme No: %d", attempt)
        logging.info("📊 Gelen branch sayısı: %d", len(branches))

        # Branch'leri kategorilere ayır - TAM İÇERİK al (kısaltma yok!)
        kategoriler = {
            "köken_analizi": [],      # zero_point
            "karşıtlar": [],          # opposites
            "olasılıklar": [],        # quantum_field
            "bilgelik": [],           # wisdom
            "bütüncül_anlam": [],     # holistic_meaning
            "zaman_akışı": [],        # divine_order
            "derin_sorular": [],      # recursive_questioning
            "spekülatif_teori": [],   # speculative_theory
            "hakikat_boşluk": []      # truth_void
        }

        type_mapping = {
            "core_essence": "köken_analizi",
            "opposite_viewpoints": "karşıtlar",
            "perspective": "karşıtlar",
            "future_applications": "olasılıklar",
            "wisdom_insights": "bilgelik",
            "holistic_meaning": "bütüncül_anlam",
            "divine_order": "zaman_akışı",
            "recursive_questioning": "derin_sorular",
            "speculative_theory": "spekülatif_teori",
            "truth_void": "hakikat_boşluk",
            "truth_void_illusion": "hakikat_boşluk"
        }

        # Branch'leri kategorilere dağıt - TAM içerik!
        for branch in branches:
            content = branch.get("content", "")
            branch_type = branch.get("type", "default")

            if content and len(content) > 50:
                mapped_type = type_mapping.get(branch_type, "bütüncül_anlam")
                # TAM içeriği al, kısaltma YOK!
                kategoriler[mapped_type].append(content)

        # Her kategoriden EN ÖZGÜN içerikleri seç (semantic uniqueness)
        secilen_icerikler = []
        tum_secilen_embeddings = []  # Tüm seçilenlerin embedding'leri

        kategori_basliklar = {
            "köken_analizi": "KÖKEN VE TEMEL",
            "karşıtlar": "KARŞIT BAKIŞLAR",
            "olasılıklar": "OLASILIKLAR VE POTANSİYELLER",
            "bilgelik": "DERİN BİLGELİK",
            "bütüncül_anlam": "BÜTÜNCÜL ANLAM",
            "zaman_akışı": "ZAMAN VE EVRİM",
            "derin_sorular": "DERİNLEŞTİRİCİ SORULAR",
            "spekülatif_teori": "SPEKÜLATİF TEORİLER",
            "hakikat_boşluk": "HAKİKAT VE YANILGI"
        }

        # Ortak/tekrarlayan ifadeleri tespit et
        common_phrases = [
            "elektromanyetik radyasyon",
            "sembolik ve kültürel",
            "mitolojik ve dini",
            "fiziksel olgu",
            "temel yapı"
        ]

        for kat_key, kat_baslik in kategori_basliklar.items():
            icerikler = kategoriler[kat_key]
            if icerikler:
                en_iyi = None
                en_iyi_skor = -1
                
                for icerik in icerikler:
                    # Özgünlük skoru hesapla
                    skor = len(icerik)  # Uzunluk base skor
                    
                    # Ortak ifadeler varsa skor düşür
                    for phrase in common_phrases:
                        if phrase in icerik.lower():
                            skor -= 100
                    
                    # Daha önce seçilenlerle benzerlik kontrolü
                    if self.sentence_model and tum_secilen_embeddings:
                        try:
                            icerik_emb = self.sentence_model.encode(icerik[:500], convert_to_tensor=True)
                            for prev_emb in tum_secilen_embeddings:
                                sim = util.cos_sim(icerik_emb, prev_emb).item()
                                if sim > 0.7:
                                    skor -= 200  # Çok benzer, skor düşür
                        except Exception as e:
                            logging.debug(f"Embedding benzerlik hesaplama hatası: {e}")

                    if skor > en_iyi_skor:
                        en_iyi_skor = skor
                        en_iyi = icerik

                if en_iyi:
                    secilen_icerikler.append(f"[{kat_baslik}]\n{en_iyi}")
                    # Bu içeriğin embedding'ini kaydet
                    try:
                        if self.sentence_model:
                            emb = self.sentence_model.encode(en_iyi[:500], convert_to_tensor=True)
                            tum_secilen_embeddings.append(emb)
                    except Exception as e:
                        logging.debug(f"Embedding kaydetme hatası: {e}")

        logging.info("📦 Senteze dahil edilen kategori: %d / 9", len(secilen_icerikler))

        if not secilen_icerikler:
            logging.warning("⚠️ Sentezlenecek içerik bulunamadı!")
            return "Yeterli içerik üretilemedi."

        # PRE-PROCESSING: Tekrarlayan tanım cümlelerini tespit et ve sadece ilkini bırak
        tum_icerik_raw = "\n\n---\n\n".join(secilen_icerikler)
        tum_icerik = self.remove_repeated_definitions(tum_icerik_raw, original_question)

        # SENTEZ PROMPT - İçerikleri GERÇEKTEN sentezle
        synthesis_prompt = f"""Sen yardımsever ve zeki bir asistansın. Amacın, aşağıda verilen içerikleri sentezleyerek ÖZGÜN ve AKICI bir yanıt üretmektir.

SORU: "{original_question}"

İÇERİKLER:
{tum_icerik}

---

KRİTİK KURALLAR:

1. TEKRAR YASAĞI:
   - Aynı tanımı (örn: "X, Y'dir") sadece BİR KERE ver, sonra bir daha tekrarlama
   - Eğer bir kavram tanımlandıysa, sonraki paragraflarda o tanımı varsay, yeniden açıklama
   - "Işık elektromanyetik radyasyondur" gibi temel tanımlar SADECE başta bir kez söylenir

2. İLERİ GİT, GERİ DÖNME:
   - Her paragraf bir öncekinin üzerine inşa etmeli
   - Başa dönüp aynı şeyleri farklı kelimelerle söyleme
   - Akış: Temel → Derinlik → Bağlantılar → Sonuç

3. BENZERSİZ İÇERİK:
   - Her kategoriden SADECE o kategoriye özgü, benzersiz fikirleri al
   - Kategoriler arası ortak ifadeleri atla
   - "Sembolik anlam", "kültürel değer", "mitolojik" gibi genel ifadeler MAX 1 kez

4. YAPI:
   - Giriş: Konunun özü (kısa, 2-3 cümle)
   - Gelişme: Her perspektiften BİR özgün içgörü (tekrarsız)
   - Derinlik: Beklenmedik bağlantılar, özgün çıkarımlar
   - Kapanış: Açık uçlu düşündürücü bir soru veya içgörü

5. DİL:
   - Doğal, akıcı Türkçe
   - Akademik değil, sohbet tarzı
   - Felsefi derinlik ama anlaşılır

YANIT:"""

        # Sentez için model_prompt="" kullan - sadece bu talimatlar geçerli olsun
        synthesized, _ = self.llama_generate(synthesis_prompt, model_prompt="", force_json=False)

        # Temizlik
        final_text = self.remove_duplicate_sentences(synthesized)
        final_text = self.clean_meta_terms(final_text)

        logging.info("✅ Sentez tamamlandı. Final yanıt uzunluğu: %d karakter", len(final_text))

        return final_text

    def clean_meta_terms(self, text):
        """Framework'e ait meta-terimleri temizle"""
        meta_terms = [
            r'[Bb]akış [Aa]çısı:?\s*',
            r'[Pp]erspektif:?\s*',
            r'[Zz]ıtlık(lar)?:?\s*',
            r'[Bb]ilgelik:?\s*',
            r'[Hh]akikat[- ][Bb]oşluk:?\s*',
            r'[Ss]pekülatif [Tt]eori:?\s*',
            r'[Öö]z [Tt]anım:?\s*',
            r'AŞAMA \d+:?\s*',
            r'[Kk]ategori:?\s*',
            r'[Bb]ütüncül [Aa]nlam:?\s*'
        ]

        cleaned = text
        for term in meta_terms:
            cleaned = re.sub(term, '', cleaned)

        # Fazla boşlukları temizle
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)

        return cleaned.strip()

    def remove_duplicate_sentences(self, text):
        """
        Geliştirilmiş tekrar temizleme - hem kelime hem anlam bazlı
        """
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
        
        if not sentences:
            return text
            
        unique_sentences = []
        seen_fingerprints = set()
        seen_embeddings = []
        
        # Semantic similarity threshold
        SEMANTIC_THRESHOLD = 0.75
        
        for sentence in sentences:
            # 1. Kelime bazlı fingerprint kontrolü
            words = [w for w in sentence.lower().split() if len(w) > 3][:5]
            fingerprint = " ".join(words)
            
            if fingerprint in seen_fingerprints:
                continue
                
            # 2. Semantic similarity kontrolü (embedding ile)
            is_duplicate = False
            try:
                if self.sentence_model and seen_embeddings:
                    sent_embedding = self.sentence_model.encode(sentence, convert_to_tensor=True)
                    for prev_embedding in seen_embeddings:
                        similarity = util.cos_sim(sent_embedding, prev_embedding).item()
                        if similarity > SEMANTIC_THRESHOLD:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_embeddings.append(sent_embedding)
            except Exception as e:
                logging.warning(f"Semantic dedup error: {e}")
            
            if not is_duplicate and len(fingerprint) > 10:
                unique_sentences.append(sentence)
                seen_fingerprints.add(fingerprint)
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else text

    def remove_repeated_definitions(self, text, topic):
        """
        Sentez öncesi: Aynı konunun tekrarlayan tanım cümlelerini tespit et,
        sadece ilkini bırak, diğerlerini kaldır.
        """
        # Tanım pattern'leri (X, Y'dir / X Y olarak tanımlanır vs.)
        definition_patterns = [
            rf'{topic}[,\s]+(bir\s+)?[\w\s]{{5,50}}(dır|dir|tır|tir|dur|dür)',
            rf'{topic}[,\s]+[\w\s]{{5,50}}olarak\s+tanımlan',
            rf'{topic}[,\s]+[\w\s]{{5,50}}(şeklinde|biçiminde)',
            r'teknik\s+(açıdan|olarak|kökeni)',
            r'fiziksel\s+(açıdan|olarak)',
            r'elektromanyetik\s+radyasyon',
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        seen_definitions = set()
        
        for line in lines:
            line_lower = line.lower()
            is_repeated_def = False
            
            for pattern in definition_patterns:
                import re
                matches = re.findall(pattern, line_lower, re.IGNORECASE)
                if matches:
                    # Bu pattern daha önce görüldü mü?
                    pattern_key = pattern[:30]  # Pattern'in ilk kısmını key olarak kullan
                    if pattern_key in seen_definitions:
                        is_repeated_def = True
                        logging.info(f"🔄 Tekrarlayan tanım atlandı: {line[:80]}...")
                        break
                    else:
                        seen_definitions.add(pattern_key)
            
            if not is_repeated_def:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def beautify_synthesis(self, raw_synthesis, original_question):
        clean_text = re.sub(r'\(Score:.*?\)|\(Coherence:.*?\)|\$\$.*?\$\$', '', raw_synthesis)
        clean_text = re.sub(r'#+\s*[^:\n]*:', '', clean_text)
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'Wikipedia.*?:|PubMed.*?:|LLM.*?:|Gemma.*?:', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(re.escape(original_question), '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s{2,}', ' ', clean_text)
        clean_text = clean_text.replace("ý", "ı").replace("þ", "ş").replace("ð", "ğ").replace("Ý", "İ").replace("Þ", "Ş").replace("Ð", "Ğ")
        return clean_text.strip()
    
    def truth_filter(self, text):
        logging.info("ADİL HAVUZ SİSTEMİ BAŞLADI: '%s'", text)
        self.current_query_text = text
        data_pool = self.collect_all_relevant_data(text)
        ranked_pool = self.rank_by_relevance_only(data_pool)
        
        if ranked_pool:
            if len(ranked_pool) >= 2:
                first_source = ranked_pool[0]
                second_source = ranked_pool[1]
                
                thinking_content = f"""
KAYNAK 1 ({first_source['source']}, Skor: {first_source['final_score']:.2f}):
{first_source['content']}

KAYNAK 2 ({second_source['source']}, Skor: {second_source['final_score']:.2f}):
{second_source['content']}
"""
                thinking_source = f"{first_source['source']} + {second_source['source']}"
                logging.info("🧠 İKİLİ THİNKİNG FRAMEWORK: %s verisi kullanılıyor", thinking_source)
            else:
                best_data = ranked_pool[0]
                thinking_content = best_data['content']
                thinking_source = best_data['source']
                logging.info("🧠 TEK THİNKİNG FRAMEWORK: %s verisi kullanılıyor", thinking_source)
        else:
            logging.warning("❌ Hiç alakalı harici veri bulunamadı! LLM'i doğrudan kullanma fallback'i uygulanıyor.")
            thinking_content = text
            thinking_source = "User Query (Fallback)"

        thought_result = self.apply_thinking_framework(text, thinking_content)
        result = {"text": text, "status": "ThoughtProcessed", "branches": []}
        enhanced_branches = self.collect_branches_enhanced(thought_result)
        
        logging.info("Thinking framework sonrası elde edilen branch sayısı: %d", len(enhanced_branches))
        
        if enhanced_branches:
            final_synthesized_text = self.create_enhanced_final_synthesis(enhanced_branches, text)
            
            if final_synthesized_text and "Error" not in final_synthesized_text and len(final_synthesized_text) > 200:
                final_clean = self.beautify_synthesis(final_synthesized_text, text)
                result["final_response"] = final_clean
                logging.info("✨ QuantumTree temiz final_response üretildi.")
                logging.info("Final response uzunluğu: %d", len(final_clean))
            else:
                result["final_response"] = ranked_pool[0]['content'] if ranked_pool else "Uygun yanıt bulunamadı."
                logging.warning("⚠️ Sentez başarısız, en iyi harici kaynak kullanıldı.")
        else:
            result["final_response"] = ranked_pool[0]['content'] if ranked_pool else "Uygun yanıt bulunamadı."
        
        logging.info("ADİL HAVUZ SİSTEMİ BİTTİ: %d kaynak", len(ranked_pool[:3]))
        
        if ranked_pool:
            top_sources = ranked_pool[:3]
            avg_score = sum(data['final_score'] for data in top_sources) / len(top_sources) if top_sources else 0
            if avg_score > 0.7:
                result["status"] = "High Quality (Multi-Source)"
            elif avg_score > 0.4:
                result["status"] = "Good Quality (Mixed Sources)"
            else:
                result["status"] = "Low Quality (Limited Sources)"
        else:
            result["status"] = "LLM Fallback (No External Data)"

        result["branches"] = [f"{data['source']}: {data['content'][:300]}... (Final Score: {data['final_score']:.2f})" for data in ranked_pool[:3]]
        result["thought_process"] = thought_result["thought_process"]
        
        logging.info(f"⚖️ ADİL HAVUZ SİSTEMİ BİTTİ: {len(ranked_pool[:3])} kaynak kullanıldı, status={result['status']}")
        
        return result

    def get_best_branch(self, result):
        branches = result.get("branches", [])
        for branch in branches:
            if "Synthesized Insight" in branch:
                content = branch.split(":", 1)[1] if ":" in branch else branch
                content = content.split("(Coherence:")[0].strip()
                if len(content) > 100:
                    return self.beautify_synthesis(content, result.get("text", ""))
        
        for branch in branches:
            if any(llm_name in branch for llm_name in ["LLaMA", "LLM", "Gemma"]):
                content = branch.split(":", 1)[1] if ":" in branch else branch
                content = content.split("(Score:")[0].strip()
                if content.endswith("..."):
                    content = content[:-3]
                if len(content) > 100:
                    return self.beautify_synthesis(content, result.get("text", ""))
        
        for branch in branches:
            if "Wikipedia" in branch:
                content = branch.split(":", 1)[1] if ":" in branch else branch
                content = content.split("(Score:")[0].strip()
                if content.endswith("..."):
                    content = content[:-3]
                if len(content) > 100:
                    return self.beautify_synthesis(content, result.get("text", ""))
        
        return "Uygun yanıt bulunamadı."
        
    def collect_all_relevant_data(self, text):
        data_pool = []
        logging.info(f"🏊 PARALEL VERİ HAVUZU OLUŞTURULUYOR: '{text}'")
        
        # Paralel executor oluştur
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Tüm kaynakları paralel başlat
            future_neo4j = executor.submit(self.fetch_from_core_task, text)
            future_wiki = executor.submit(self.wikipedia_verify_task, text)
            future_llm = executor.submit(self.llama_generate_task, text)
            future_faiss = executor.submit(self.faiss_search_task, text)
            
            logging.info("🔄 4 kaynak paralel olarak çalışıyor...")
            
            # Neo4j sonucu
            neo4j_result = future_neo4j.result()
            if neo4j_result:
                core_data, core_source, core_score = neo4j_result
                if core_data:
                    ok, coherence_score = self.coherence_check(text, core_data)
                    if ok and coherence_score > 0.2:
                        data_pool.append({
                            "content": core_data, "source": core_source, "type": "neo4j",
                            "relevance_score": coherence_score, "quality_score": core_score,
                            "final_score": (coherence_score * 0.7) + (core_score * 0.3)
                        })
                        logging.info(f"🏊 NEO4J PARALEL HAVUZA EKLENDİ: {core_source}")
            
            # Wikipedia sonucu
            wiki_result = future_wiki.result()
            if wiki_result:
                valid_wiki, wiki_source, wiki_data, wiki_score = wiki_result
                if valid_wiki and wiki_data:
                    ok, coherence_score = self.coherence_check(text, wiki_data)
                    if ok and coherence_score > 0.2:
                        data_pool.append({
                            "content": wiki_data, "source": wiki_source, "type": "wikipedia",
                            "relevance_score": coherence_score, "quality_score": wiki_score,
                            "final_score": (coherence_score * 0.7) + (wiki_score * 0.3)
                        })
                        logging.info(f"🏊 WIKIPEDIA PARALEL HAVUZA EKLENDİ: {wiki_source}")
            
            # LLM sonucu
            llm_result = future_llm.result()
            if llm_result:
                llm_response, llm_score = llm_result
                if llm_response and "Error" not in llm_response:
                    ok, coherence_score = self.coherence_check(text, llm_response)
                    if ok and coherence_score > 0.1:
                        data_pool.append({
                            "content": llm_response, "source": f"LLM ({config.get('ollama_model_name', 'gemma3:27b')})",
                            "type": "llm", "relevance_score": coherence_score, "quality_score": llm_score,
                            "final_score": (coherence_score * 0.7) + (llm_score * 0.3)
                        })
                        logging.info(f"🏊 LLM PARALEL HAVUZA EKLENDİ")
            
            # Faiss sonucu
            faiss_result = future_faiss.result()
            if faiss_result:
                for res in faiss_result:
                    ok, coherence_score = self.coherence_check(text, res["text"], threshold=0.2)
                    print(f"DEBUG: ok={ok}, coherence_score={coherence_score}, threshold=0.2")
                    if ok:
                        logging.info(f"📚 FAISS VERİSİ LLM'E GÖNDERİLDİ: Coherence={coherence_score:.3f}")
                        logging.info(f"📖 Metin örneği: {res['text'][:150]}...")
                        data_pool.append({
                            "content": res["text"], "source": "FAISS-Database", "type": "faiss",
                            "relevance_score": coherence_score, "quality_score": res["score"],
                            "final_score": (coherence_score * 0.7) + (res["score"] * 0.3)
                        })
                        logging.info(f"🏊 FAISS PARALEL HAVUZA EKLENDİ")
                    else:
                        print(f"DEBUG: Neden elendi? ok={ok}")
                        logging.info(f"❌ FAISS VERİSİ ELENDİ: Coherence={coherence_score:.3f} (threshold: 0.2)")
                        logging.info(f"🗑️ Elenen metin: {res['text'][:100]}...")
        
        logging.info(f"✅ PARALEL HAVUZ TAMAMLANDI: {len(data_pool)} veri toplandı")
        
        # Library results (sıralı çalışabilir, yavaş değil)
        library_results = self.fetch_from_libraries(text)
        for result in library_results:
            ok, coherence_score = self.coherence_check(text, result["content"])
            if ok and coherence_score > 0.2:
                data_pool.append({
                    "content": result["content"], "source": result["source"], "type": "library",
                    "relevance_score": coherence_score, "quality_score": result["score"],
                    "final_score": (coherence_score * 0.7) + (result['score'] * 0.3)
                })
                logging.info(f"🏊 LIBRARY HAVUZA EKLENDİ: {result['source']}")
        
        logging.info(f"🏊 TOPLAM PARALEL HAVUZ: {len(data_pool)} veri")
        return data_pool

    def rank_by_relevance_only(self, data_pool):
        logging.info(f"⚖️ ADALET SIRALAMA BAŞLADI: {len(data_pool)} veri")
        ranked_pool = sorted(data_pool, key=lambda x: x["final_score"], reverse=True)
        
        logging.info(f"🏆 ADALET SIRALAMASINDA KAZANANLAR:")
        for i, data in enumerate(ranked_pool[:5]):
            logging.info(f"🥇 {i+1}. {data['source']} - Final Score: {data['final_score']:.2f} (Relevance: {data['relevance_score']:.2f}, Quality: {data['quality_score']:.2f})")
        
        return ranked_pool
    
    def discover_connections(self):
        if not self.core:
            logging.warning("Neo4j bağlantısı yok, connection discovery atlanıyor")
            return None
        try:
            with self.core.session() as session:
                nodes = session.run("MATCH (n:DeepKnowledge) RETURN n.topic_key, n.category LIMIT 20")
                node_list = [(record["n.topic_key"], record.get("n.category", "general")) for record in nodes]
                
                if len(node_list) < 2:
                    logging.info("Not enough DeepKnowledge nodes for connection discovery.")
                    return None
                    
                pair = random.sample(node_list, 2)
                node1, node2 = pair[0][0], pair[1][0]
                
                prompt = self.get_step_prompt("connection_discovery", f"'{node1[:100]}' and '{node2[:100]}'")
                insight, _ = self.llama_generate(prompt)
                
                if insight and "Error" not in insight:
                    with self.core.session() as session:
                        session.run(
                            "MERGE (n1:DeepKnowledge {topic_key: $text1}) "
                            "MERGE (n2:DeepKnowledge {topic_key: $text2}) "
                            "MERGE (c:Connection {insight: $insight}) "
                            "MERGE (n1)-[:CONNECTED_TO]->(c)<-[:CONNECTED_TO]-(n2)",
                            text1=node1.lower(), text2=node2.lower(), insight=insight.lower()
                        )
                    logging.info(f"Connection saved: {node1[:50]}... <-> {node2[:50]}...")
                    return {"node1": node1[:100], "node2": node2[:100], "insight": insight}
        except Exception as e:
            logging.error(f"Connection discovery error: {e}")
        
        return None

    def auto_fetch_background(self):
        auto_background = config.get('auto_background', {'enabled': False})
        if not auto_background.get('enabled', False):
            logging.info("Auto background disabled in config.")
            return
        
        while self.running:
            try:
                time.sleep(auto_background.get('interval_minutes', 10) * 60)
                connection = self.discover_connections()
            except Exception as e:
                logging.error(f"Background process error: {e}")

    def stop_background(self):
        self.running = False
        logging.info("Background thread stopped.")

    def fetch_from_core_task(self, text):
        """Neo4j task for parallel processing"""
        try:
            return self.fetch_from_core(text)
        except Exception as e:
            logging.error(f"Neo4j paralel task hatası: {e}")
            return None

    def wikipedia_verify_task(self, text):
        """Wikipedia task for parallel processing"""
        try:
            return self.wikipedia_verify(text)
        except Exception as e:
            logging.error(f"Wikipedia paralel task hatası: {e}")
            return None

    def llama_generate_task(self, text):
        """LLM task for parallel processing"""
        try:
            return self.llama_generate(text)
        except Exception as e:
            logging.error(f"LLM paralel task hatası: {e}")
            return None

    def faiss_search_task(self, text):
        """Faiss task for parallel processing"""
        try:
            return self.faiss_search(text, k=5)
        except Exception as e:
            logging.error(f"Faiss paralel task hatası: {e}")
            return None

    def extract_concepts(self, text):
        """Metinden kavramları çıkarır"""
        import re
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'were', 
                      'been', 'their', 'what', 'your', 'when', 'where', 'will', 
                      'there', 'would', 'could', 'should', 'think', 'about'}
        concepts = [word for word in important_words if word not in stop_words]
        return list(set(concepts))[:10]

    def smart_neo4j_search(self, query, max_results=5):
        """Akıllı Neo4j arama - kavram + embedding benzerliği"""
        
        query_concepts = self.extract_concepts(query)
        # embedding boşsa hata olmasın diye kontrol eklendi
        if self.embedder:
            query_embedding = self.embedder.encode(query).tolist()
        else:
            query_embedding = None
        
        with self.core.session() as session:
            # 1. Kavram bazlı arama
            concept_results = []
            for concept in query_concepts:
                results = session.run("""
                    MATCH (main:DeepKnowledge)
                    WHERE main.topic_key CONTAINS $concept 
                          OR main.content CONTAINS $concept
                    OPTIONAL MATCH (main)-[r:SIMILAR_TO]-(connected:DeepKnowledge)
                    RETURN main.topic_key as topic_key,
                            main.content as content,
                            main.quality_score as quality_score,
                            main.source as source,
                            collect(DISTINCT connected.topic_key) as connected_topics,
                            count(DISTINCT connected) as connection_count
                    ORDER BY main.quality_score DESC, connection_count DESC
                    LIMIT $limit
                """, concept=concept, limit=max_results)
                
                concept_results.extend([dict(record) for record in results])
            
            # 2. Embedding benzerliği ile arama
            all_nodes = session.run("""
                MATCH (main:DeepKnowledge)
                WHERE main.embedding IS NOT NULL
                OPTIONAL MATCH (main)-[r:SIMILAR_TO]-(connected:DeepKnowledge)
                RETURN main.topic_key as topic_key,
                        main.content as content,
                        main.quality_score as quality_score,
                        main.source as source,
                        main.embedding as embedding,
                        collect(DISTINCT connected.topic_key) as connected_topics,
                        count(DISTINCT connected) as connection_count
            """)
            
            similarity_results = []
            if query_embedding:
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
                for record in all_nodes:
                    if record['embedding'] and len(record['embedding']) > 0:  # Boş veya null kontrolü eklendi
                        try:
                            node_embedding = np.array(record['embedding']).reshape(1, -1)
                            similarity = cosine_similarity(query_embedding_np, node_embedding)[0][0]
                        
                            if similarity > 0.3:
                                result_dict = dict(record)
                                result_dict['similarity_score'] = float(similarity)
                                del result_dict['embedding']
                                similarity_results.append(result_dict)
                        except Exception as e:
                            logging.error(f"Embedding işleme hatası: {e} - Record: {record['topic_key']}")
                            continue
                    else:
                        logging.warning(f"Neo4j'de {record['topic_key']} için boş veya geçersiz embedding bulundu, karşılaştırma atlandı.")
            
            # Sonuçları birleştir ve sırala
            all_results = concept_results + similarity_results
            
            # Duplikat temizleme
            seen_topics = set()
            unique_results = []
            for result in all_results:
                if result['topic_key'] not in seen_topics:
                    seen_topics.add(result['topic_key'])
                    unique_results.append(result)
            
            # Skorlama - GÜNCELLENDİ
            for result in unique_results:
                quality_score = result.get('quality_score', 0)
                connection_count = result.get('connection_count', 0)
                similarity_score = result.get('similarity_score', 0)
                
                # None kontrolü ekle
                if quality_score is None:
                    quality_score = 0.0
                if connection_count is None:
                    connection_count = 0.0
                if similarity_score is None:
                    similarity_score = 0.0
                    
                score = float(quality_score) * 0.4
                score += float(connection_count) * 0.3
                score += float(similarity_score) * 0.3
                result['final_score'] = score
            
            unique_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            return unique_results[:max_results]

    def get_knowledge_graph(self, topic_key, depth=2):
        """Belirli bir konunun bilgi grafiğini getirir"""
        
        with self.core.session() as session:
            result = session.run("""
                MATCH path = (main:DeepKnowledge {topic_key: $topic_key})
                             -[:SIMILAR_TO*1..""" + str(depth) + """]-(connected:DeepKnowledge)
                WITH main, connected, 
                     relationships(path) as rels,
                     length(path) as path_length
                WHERE path_length <= $depth
                RETURN main.topic_key as main_topic,
                        main.content as main_content,
                        main.quality_score as main_quality,
                        collect(DISTINCT {
                            topic: connected.topic_key,
                            content: connected.content,
                            quality: connected.quality_score,
                            distance: path_length,
                            strength: CASE WHEN size(rels) > 0 THEN rels[0].strength ELSE 0.5 END
                        }) as connections
            """, topic_key=topic_key, depth=depth).single()
            
            if result:
                # Access count güncelle
                session.run("""
                    MATCH (main:DeepKnowledge {topic_key: $topic_key})
                    SET main.access_count = COALESCE(main.access_count, 0) + 1,
                        main.last_accessed = timestamp()
                """, topic_key=topic_key)
                
                return dict(result)
            
            return None

    def enrich_with_graph(self, main_content, connections):
        """Ana içeriği bağlantılarla zenginleştirir"""
        
        if not connections:
            return main_content
        
        enrichment = "\n\n--- İLİŞKİLİ BİLGİLER ---\n"
        
        # En güçlü bağlantıları ekle
        strong_connections = [c for c in connections 
                              if c.get('strength', 0) > 0.7][:3]
        
        for i, conn in enumerate(strong_connections, 1):
            enrichment += f"{i}. [{conn.get('topic', 'N/A')}] "
            enrichment += f"(Güç: {conn.get('strength', 0):.2f})\n"
            enrichment += f"    {conn.get('content', 'N/A')[:200]}...\n\n"
        
        return main_content + enrichment

    def save_to_core(self, info, source, is_valid=True, score=0, related_topics=None):
        if not self.core or not info or not info.strip():
            return
            
        topic_key = self.extract_topic_key(info)
        content_limited = info.lower()[:1000]
        source_limited = source[:100]
        
        # Kavramları çıkar
        concepts = self.extract_concepts(info)
        
        # Embedding hesapla
        if self.embedder:
            content_embedding = self.embedder.encode(info).tolist()
        else:
            content_embedding = None

        # GÜNCELLENMİŞ KOD: Score'u None kontrolü ile güvenli hale getir
        safe_score = float(score) if score is not None else 0.0

        try:
            with self.core.session() as session:
                # Cypher syntax hatası giderildi ve ON CREATE/ON MATCH ayrımları yapıldı
                session.run(
                    "MERGE (n:DeepKnowledge {topic_key: $topic_key}) "
                    "ON CREATE SET n.created_at = timestamp(), "
                    "            n.content = $content, "
                    "            n.quality_score = $score, "  # safe_score kullan
                    "            n.source = $source, "
                    "            n.embedding = $embedding, "
                    "            n.concepts = $concepts, "
                    "            n.access_count = 1, "
                    "            n.evolution_level = 1 "
                    "ON MATCH SET n.content = $content, "
                    "           n.quality_score = $score, "
                    "           n.source = $source, "
                    "           n.embedding = $embedding, "
                    "           n.concepts = $concepts, "
                    "           n.access_count = COALESCE(n.access_count, 0) + 1, "
                    "           n.evolution_level = COALESCE(n.evolution_level, 1) + 1, "
                    "           n.last_accessed = timestamp()",
                    topic_key=topic_key, content=content_limited, score=safe_score, 
                    source=source_limited, embedding=content_embedding, concepts=concepts
                )
                
                # Kavram node'larını oluştur ve bağla
                for concept in concepts:
                    session.run("""
                        MERGE (c:Concept {name: $concept})
                        WITH c
                        MATCH (main:DeepKnowledge {topic_key: $topic_key})
                        MERGE (main)-[:CONTAINS_CONCEPT]->(c)
                    """, concept=concept, topic_key=topic_key)
                
                # Benzer içeriklerle bağlantı oluştur
                if content_embedding:
                    self.create_similarity_connections(session, topic_key, content_embedding)

                logging.info(f"✅ Neo4j'ye eklendi: {topic_key} ({len(concepts)} kavram)")
            
        except Exception as e:
            logging.error(f"Enhanced save_to_core error: {e}")

    def create_similarity_connections(self, session, topic_key, new_embedding):
        """Benzer içeriklerle otomatik bağlantı oluşturur - NoneType korumalı"""
        
        # Yeni eklenen embedding None ise işlemi durdur
        if new_embedding is None:
            logging.warning("Yeni embedding boş, benzerlik bağlantısı oluşturulamadı.")
            return

        # GÜNCELLENMİŞ QUERY - quality_score ve embedding kontrolü eklendi
        existing_nodes = session.run("""
            MATCH (n:DeepKnowledge)
            WHERE n.topic_key <> $topic_key 
              AND n.embedding IS NOT NULL
              AND n.quality_score IS NOT NULL
              AND size(n.embedding) > 0
            RETURN n.topic_key as topic_key, 
                   n.embedding as embedding,
                   n.quality_score as quality_score
        """, topic_key=topic_key)
        
        new_embedding_np = np.array(new_embedding).reshape(1, -1)
        
        for record in existing_nodes:
            try:
                # Çifte güvenlik kontrolü
                if (record['embedding'] and 
                    len(record['embedding']) > 0 and 
                    record['quality_score'] is not None):
                        
                    existing_embedding = np.array(record['embedding']).reshape(1, -1)
                    similarity = cosine_similarity(new_embedding_np, existing_embedding)[0][0]
                
                    if not np.isnan(similarity) and similarity > self.similarity_threshold:
                        # Benzerlik bağlantısı oluştur
                        session.run("""
                            MATCH (a:DeepKnowledge {topic_key: $topic_a})
                            MATCH (b:DeepKnowledge {topic_key: $topic_b})
                            MERGE (a)-[r:SIMILAR_TO]->(b)
                            SET r.strength = $similarity,
                                r.created_at = timestamp()
                            MERGE (b)-[r2:SIMILAR_TO]->(a)
                            SET r2.strength = $similarity,
                                r2.created_at = timestamp()
                        """, topic_a=topic_key, topic_b=record['topic_key'], 
                            similarity=float(similarity))
                        
                        logging.info(f"🔗 Bağlantı oluşturuldu: {topic_key} ↔ {record['topic_key']} "
                                     f"(similarity: {similarity:.3f})")
            except (ValueError, TypeError, AttributeError) as e:
                logging.error(f"Embedding işleme hatası: {e} - Record: {record['topic_key']}")
                continue


# Global tree instance
tree = QuantumTree()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form.get('text')
        if not text or text.strip() == "":
            result = {
                "text": "No input", "status": "Error", "branches": ["Please enter a valid question."],
                "synthesized_insight": "", "synthesized_coherence": 0.0, "final_response": "Lütfen geçerli bir soru girin."
            }
        else:
            try:
                result = tree.truth_filter(text)
            except Exception as e:
                logging.error(f"Truth filter error: {e}")
                result = {
                    "text": text, "status": "Error", "branches": [f"System error: {str(e)}"],
                    "synthesized_insight": "", "synthesized_coherence": 0.0,
                    "final_response": f"Sistem hatası: {str(e)}"
                }
        
        if 'history' not in session:
            session['history'] = []
        session['history'].append({"text": text or "Empty Input", "status": result["status"]})
        session['history'] = session['history'][-2:]
        session.modified = True
    
    history = session.get('history', [])
    return render_template_string('''
    <html>
    <head>
        <title>QuantumTree Debug</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f4f7f6; color: #333; }
            .container { max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { color: #2e8b57; text-align: center; margin-bottom: 20px; }
            textarea { width: 100%; height: 100px; padding: 10px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; resize: vertical; box-sizing: border-box; }
            .button-group { display: flex; justify-content: space-between; margin-top: 15px; }
            button, .link-button { padding: 12px 24px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; text-decoration: none; color: white; display: inline-block; text-align: center; }
            button[type="submit"] { background-color: #ff6347; }
            .link-button.settings { background-color: #4682b4; }
            .link-button.discover { background-color: #8a2be2; }
            button:hover, .link-button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
            .debug-info { background: #f0f8ff; padding: 15px; border-radius: 8px; margin-top: 25px; border-left: 4px solid #4682b4; }
            .final-response-box { background: #e8f5e8; padding: 25px; border-radius: 8px; border-left: 5px solid #2e8b57; margin-top: 20px; }
            .final-response-box p { color: #333; line-height: 1.8; font-size: 17px; }
            details { margin: 20px 0; background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 15px; }
            summary { font-size: 1.2em; font-weight: bold; cursor: pointer; color: #555; }
            details div { padding: 10px 0; }
            ul { list-style-type: none; padding: 0; }
            li { background: #f9f9f9; border-left: 3px solid #ccc; padding: 10px; margin: 8px 0; border-radius: 5px; }
            .status-ok { color: green; font-weight: bold; }
            .status-warn { color: orange; font-weight: bold; }
            .status-error { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>QuantumTree: Hakikatin Ateşiyle Aydınlanmış</h1>
        <form method="POST" action="/">
            <textarea name="text" placeholder="Bir kıvılcım bırakın..."></textarea>
            <div class="button-group">
                <button type="submit">Ateşi Yay</button>
                <a href="/settings" class="link-button settings">Ayarlara Git</a>
                <a href="/discover" class="link-button discover">Bağlantıları Keşfet</a>
            </div>
        </form>
        {% if result %}
        <hr>
        <div class="debug-info">
            <h3>🔍 Hata Ayıklama Bilgisi:</h3>
            <h4>Ateş Durumu: <span class="status-{% if 'High Quality' in result['status'] %}ok{% elif 'Good Quality' in result['status'] %}warn{% else %}error{% endif %}">{{ result['status'] }}</span></h4>
            <p>Dal Sayısı: {{ result['branches']|length }}</p>
            <p>Düşünce Süreci Adım Sayısı: {{ result.get('thought_process', [])|length }}</p>
            <p>Sentez Var mı?: {{ 'Evet' if result.get('final_response') else 'Hayır' }}</p>
        </div>
        
        {% if result.get('final_response') %}
        <div class="final-response-box">
            <h3>💡 Nihai Yanıt:</h3>
            <p>{{ result['final_response'] }}</p>
        </div>
        {% endif %}
        
        <details>
            <summary><h3>🌳 Tüm Dallar ({{ result['branches']|length }})</h3></summary>
            {% if result['branches'] %}
            <ul>
                {% for branch in result['branches'] %}
                <li><strong>{{ loop.index }}.</strong> {{ branch }}</li>
                {% endfor %}
            </ul>
            {% else %}
            <p>Güvenilir bir yanıt bulunamadı, sistem öğreniyor...</p>
            {% endif %}
        </details>
        
        {% if result.get('thought_process') %}
        <details>
            <summary><h3>🧠 Düşünce Süreci Detayları ({{ result['thought_process']|length }} adım)</h3></summary>
            <div>
                {% for step in result['thought_process'] %}
                <details>
                    <summary style="font-weight: normal; color: #8a2be2;"><strong>{{ loop.index }}. {{ step['step']|title }}</strong></summary>
                    <div style="padding-left: 20px;">
                        {% if step['output'] %}
                            <pre>{{ step['output']|tojson(indent=2) }}</pre>
                        {% else %}
                            <p><em>Çıktı üretilmedi</em></p>
                        {% endif %}
                    </div>
                </details>
                {% endfor %}
            </div>
        </details>
        {% endif %}
        {% endif %}
        
        {% if history %}
        <h3>📝 Geçmiş Analizler:</h3>
        <ul>
            {% for item in history %}
            <li><strong>{{ item['text'] }}</strong>: <span class="status-{% if 'High Quality' in item['status'] %}ok{% elif 'Good Quality' in item['status'] %}warn{% else %}error{% endif %}">{{ item['status'] }}</span></li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    </body>
    </html>
    ''', result=result, history=history, config=config)

@app.route('/set', methods=['POST'])
def set_config():
    global config
    steps_enabled = {}
    for step_name in config['thinking_framework']['steps_enabled'].keys():
        steps_enabled[step_name] = f'step_{step_name}' in request.form
    
    new_config = {
        'ollama_model_name': request.form.get('ollama_model_name', 'gemma3:27b'),
        'similarity_threshold': float(request.form.get('similarity_threshold', 0.85)),
        'coherence_threshold': float(request.form.get('coherence_threshold', 0.60)),
        'llm_coherence_threshold': float(request.form.get('llm_coherence_threshold', 0.60)),
        'external_sources': {
            'wikipedia': 'wikipedia' in request.form, 'scholar': 'scholar' in request.form,
            'twitter': 'twitter' in request.form, 'pubmed': 'pubmed' in request.form,
            'ieee': 'ieee' in request.form, 'wikipedia_lang': config['external_sources'].get('wikipedia_lang', 'en')
        },
        'generation': {
            'enabled': 'generation_enabled' in request.form,
            'max_tokens': int(request.form.get('max_tokens', 1500)),
            'temperature': float(request.form.get('temperature', 0.85)),
            'confidence_threshold': float(request.form.get('confidence_threshold', 0.85)),
            'num_ctx': config.get('llm_generation', config.get('generation', {})).get('num_ctx', 32768)
        },
        'auto_background': {
            'enabled': 'auto_background_enabled' in request.form,
            'interval_minutes': int(request.form.get('auto_background_interval_minutes', 10)),
            'relation_generation_depth': int(request.form.get('auto_background_relation_generation_depth', 2)),
            'max_random_entity_pairs': int(request.form.get('auto_background_max_random_entity_pairs', 4)),
            'relation_similarity_threshold': float(request.form.get('auto_background_relation_similarity_threshold', 0.7))
        },
        'case_sensitive': 'case_sensitive' in request.form,
        'thinking_framework': {
            'enabled': 'thinking_enabled' in request.form,
            'synthesis_enabled': 'synthesis_enabled' in request.form,
            'steps_enabled': steps_enabled,
            'max_opposites': int(request.form.get('max_opposites', 4)),
            'max_potentials': int(request.form.get('max_potentials', 5)),
            'max_wisdom_insights': int(request.form.get('max_wisdom_insights', 4)),
            'max_timeline_stages': int(request.form.get('max_timeline_stages', 4)),
            'protected_categories': ['speculative_theory', 'holistic_meaning', 'truth_void'],
            'force_include_protected': 'force_include_protected' in request.form,
            'similarity_threshold_protected': float(request.form.get('similarity_threshold_protected', 0.65)),
            'content_preservation_keywords': [kw.strip() for kw in request.form.get('content_preservation_keywords', '').split(',') if kw.strip()]
        },
        'beautification': config.get('beautification', {}),
        'update_interval_days': int(request.form.get('update_interval_days', 7)),
        "neo4j": config.get('neo4j', {}),
        "file_paths": config.get('file_paths', {})
    }
    
    try:
        save_config(new_config)
        config = new_config
        logging.info("Config başarıyla güncellendi")
    except Exception as e:
        logging.error(f"Config güncellenirken hata: {e}")
    
    return redirect(url_for('settings'))

@app.route('/test')
def test():
    test_results = {"neo4j_connection": False, "sentence_model": False, "thinking_framework": False, "ollama_connection": False, "config_valid": False}
    try:
        if tree.core:
            with tree.core.session() as session:
                session.run("RETURN 1")
            test_results["neo4j_connection"] = True
    except Exception as e:
        logging.error(f"Neo4j test failed: {e}")
    test_results["sentence_model"] = tree.sentence_model is not None
    test_results["thinking_framework"] = tree.thinking_framework is not None
    try:
        response = ollama.generate(model=config.get("ollama_model_name", "gemma3:27b"), prompt="Test", options={"num_predict": 10})
        test_results["ollama_connection"] = True
    except Exception as e:
        logging.error(f"Ollama test failed: {e}")
    test_results["config_valid"] = bool(config and isinstance(config, dict))
    return f"""
    <html>
    <head><title>Sistem Testi</title></head>
    <body style="font-family: Arial; padding: 20px;">
    <h1>Sistem Testi Sonuçları</h1>
    <ul>
    {''.join([f'<li style="color: {"green" if v else "red"};">{k}: {"✅" if v else "❌"}</li>' for k, v in test_results.items()])}
    </ul>
    <a href="/">Ana Sayfaya Dön</a>
    </body>
    </html>
    """
    
@app.route('/discover')
def discover():
    connection = tree.discover_connections()
    return f"""
    <html>
    <head><title>Bağlantıları Keşfet</title></head>
    <body style="font-family: Arial; padding: 20px;">
    <h1>Bağlantı Keşfi</h1>
    {f'<p><strong>Bağlantı:</strong> {connection["insight"]}</p>' if connection else '<p>Bağlantı bulunamadı.</p>'}
    <a href="/">Ana Sayfaya Dön</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    try:
        if config.get('auto_background', {}).get('enabled', False):
            tree.background_thread.start()
        logging.info("Background thread started")
        app.run(host='0.0.0.0', port=8080, debug=True)
    except Exception as e:
        logging.error(f"App start error: {e}")
        print(f"KRITIK HATA: {e}")
        print("Lütfen Neo4j ve Ollama'nın çalıştığından emin olun.")
        exit(1)