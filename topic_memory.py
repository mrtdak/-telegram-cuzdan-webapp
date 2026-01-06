"""
Topic Memory - Uzun Dönem Hafıza Sistemi
Konu bazlı, semantik benzerlikle gruplandırma

Her konu AYRI dosya olarak saklanır (spesifik konular, genel kategoriler DEĞİL)

Dosya Yapısı:
user_data/user_{id}/topic_memory/
  ├── topics_index.json    # Konu listesi + embeddings
  └── categories/
      ├── hazine_adasi_kitabi.json
      ├── kuru_fasulye_tarifi.json
      ├── allahin_sifatlari.json
      └── python_list_comprehension.json

Kullanım:
    memory = TopicMemory(user_id="murat")

    memory.save_topic(messages, topic_hint="Hazine Adası")

    categories = memory.get_categories()

    context = memory.get_context_for_query("Hazine Adası kitabı")
"""

import json
import os
import time
import hashlib
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import requests

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("TopicMemory: sentence_transformers yok, keyword araması kullanılacak")


class TopicMemory:
    """
    Uzun Dönem Hafıza Sistemi

    - Kategori bazlı organizasyon
    - Semantic similarity ile benzerlik kontrolü
    - Aynı gün = güncelle, farklı gün = yeni session
    - Minimum kalite kontrolü
    """

    MIN_MEANINGFUL_MESSAGES = 3  # En az 3 anlamlı mesaj
    MIN_TOTAL_CHARS = 100  # En az 100 karakter toplam içerik
    SIMILARITY_THRESHOLD = 0.70  # %70 benzerlik eşiği - aynı konunun farklı yönleri birleşsin

    TRIVIAL_PATTERNS = [
        r'^(merhaba|selam|hey|hi|hello)[\s!?.]*$',
        r'^(teşekkür|sağol|eyvallah|mersi|thanks)[\s!?.]*$',
        r'^(tamam|ok|okay|oldu|anladım|evet|hayır)[\s!?.]*$',
        r'^(görüşürüz|hoşçakal|bye|bay bay|iyi günler)[\s!?.]*$',
        r'^(hmm|hm|aa|oh|vay|wow)[\s!?.]*$',
        r'^[!?.]+$',  # Sadece noktalama
    ]

    def __init__(
        self,
        user_id: str = "default",
        base_dir: str = "user_data",
        together_api_key: str = None,
        together_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        embedding_model: str = "BAAI/bge-m3"
    ):
        self.user_id = user_id
        self.together_api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
        self.together_model = together_model

        self.memory_dir = os.path.join(base_dir, f"user_{user_id}", "topic_memory")
        self.categories_dir = os.path.join(self.memory_dir, "categories")
        self.index_file = os.path.join(self.memory_dir, "topics_index.json")

        os.makedirs(self.categories_dir, exist_ok=True)

        self.index: Dict[str, Any] = self._load_index()

        self._embedder = None
        self._embedding_model_name = embedding_model

        print(f"TopicMemory baslatildi - {len(self.index.get('categories', {}))} kategori")

    @property
    def embedder(self):
        """Lazy load embedding model"""
        if self._embedder is None and EMBEDDINGS_AVAILABLE:
            print("Embedding modeli yukleniyor (TopicMemory)...")
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder


    def _load_index(self) -> Dict[str, Any]:
        """Index dosyasını yükle"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Index yükleme hatası: {e}")

        return {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "categories": {},  # category_id -> {name, embedding, last_updated, session_count}
        }

    def _save_index(self):
        """Index'i kaydet"""
        try:
            self.index["last_updated"] = datetime.now().isoformat()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Index kaydetme hatasi: {e}")

    def _load_category(self, category_id: str) -> Dict[str, Any]:
        """Kategori dosyasını yükle"""
        filepath = os.path.join(self.categories_dir, f"{category_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Kategori yükleme hatası ({category_id}): {e}")

        return {
            "category_id": category_id,
            "sessions": []  # [{date, summary, messages_count, key_points}]
        }

    def _save_category(self, category_id: str, data: Dict[str, Any]):
        """Kategori dosyasını kaydet"""
        filepath = os.path.join(self.categories_dir, f"{category_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Kategori kaydetme hatasi: {e}")


    def _is_trivial_message(self, content: str) -> bool:
        """Mesaj anlamsız mı? (merhaba, teşekkürler vb.)"""
        if not content:
            return True

        content_lower = content.lower().strip()

        if len(content_lower) < 5:
            return True

        for pattern in self.TRIVIAL_PATTERNS:
            if re.match(pattern, content_lower, re.IGNORECASE):
                return True

        return False

    def _count_meaningful_messages(self, messages: List[Dict]) -> int:
        """Anlamlı mesaj sayısını hesapla"""
        count = 0
        for msg in messages:
            content = msg.get("content", "")
            if not self._is_trivial_message(content):
                count += 1
        return count

    def _calculate_total_content_length(self, messages: List[Dict]) -> int:
        """Toplam içerik uzunluğunu hesapla (trivial mesajlar hariç)"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if not self._is_trivial_message(content):
                total += len(content)
        return total

    def is_worth_saving(self, messages: List[Dict]) -> Tuple[bool, str]:
        """
        Bu konuşma kaydedilmeye değer mi?

        Returns:
            (is_worth, reason)
        """
        if not messages:
            return False, "Mesaj yok"

        meaningful_count = self._count_meaningful_messages(messages)
        if meaningful_count < self.MIN_MEANINGFUL_MESSAGES:
            return False, f"Yetersiz anlamli mesaj ({meaningful_count}/{self.MIN_MEANINGFUL_MESSAGES})"

        total_length = self._calculate_total_content_length(messages)
        if total_length < self.MIN_TOTAL_CHARS:
            return False, f"Yetersiz icerik ({total_length}/{self.MIN_TOTAL_CHARS} karakter)"

        return True, "Kaydedilmeye deger"


    def _generate_category_id(self, name: str) -> str:
        """Kategori adından ID oluştur (türkçe karakter temizleme)"""
        tr_map = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'c', 'Ğ': 'g', 'İ': 'i', 'Ö': 'o', 'Ş': 's', 'Ü': 'u'
        }

        result = name.lower()
        for tr_char, en_char in tr_map.items():
            result = result.replace(tr_char, en_char)

        result = re.sub(r'[^a-z0-9]+', '_', result)
        result = result.strip('_')

        return result[:50]  # Max 50 karakter

    def _detect_category_with_llm(self, messages: List[Dict], topic_hint: str = "") -> Tuple[str, str]:
        """
        LLM ile kategori tespit et

        Returns:
            (category_name, summary)
        """
        try:
            conversation = []
            for m in messages[-8:]:  # Son 8 mesaj
                role = "Kullanici" if m.get("role") == "user" else "AI"
                content = (m.get("content") or "")[:250]
                if content and not self._is_trivial_message(content):
                    conversation.append(f"{role}: {content}")

            if not conversation:
                return "genel_sohbet", "Genel konusma"

            conv_text = "\n".join(conversation)

            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Asagidaki konusmayi analiz et.

KONUSMA:
{conv_text}

GOREV:
1. Bu konusmanin SPESIFIK KONUSU nedir? (2-4 kelime, NET ve OZGUN)
2. Konusmanin kisa ozeti nedir? (1-2 cumle)

ONEMLI: Genel kategori DEGIL, spesifik konu adi ver!

DOGRU ORNEKLER:
- "Hazine Adasi Kitabi" (DOGRU - spesifik)
- "Kuru Fasulye Tarifi" (DOGRU - spesifik)
- "Allah'in Sifatlari" (DOGRU - spesifik)
- "Python List Comprehension" (DOGRU - spesifik)
- "Antarktika Buzullari" (DOGRU - spesifik)

YANLIS ORNEKLER:
- "Edebiyat" (YANLIS - cok genel)
- "Yemek" (YANLIS - cok genel)
- "Din" (YANLIS - cok genel)
- "Programlama" (YANLIS - cok genel)

FORMAT:
KONU: [spesifik konu adi]
OZET: [kisa ozet]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.together_model,
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.2,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                },
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["text"].strip()
                result = result.replace("<|eot_id|>", "").strip()

                topic_name = "Genel Sohbet"
                summary = ""

                for line in result.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("KONU:"):
                        topic_name = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("OZET:"):
                        summary = line.split(":", 1)[1].strip()

                if not summary:
                    summary = f"{topic_name} hakkinda konusma"

                return topic_name, summary

        except Exception as e:
            print(f"Kategori tespiti hatasi: {e}")

        if topic_hint:
            return topic_hint, f"{topic_hint} hakkinda konusma"
        return "Genel Sohbet", "Genel konusma"

    def _find_similar_category(self, category_name: str) -> Optional[str]:
        """
        Mevcut kategorilerde benzer var mı?

        Returns:
            category_id if found (similarity >= 70%), else None
        """
        if not self.embedder or not self.index.get("categories"):
            return None

        try:
            new_embedding = self.embedder.encode(category_name)

            best_match = None
            best_score = 0

            for cat_id, cat_info in self.index["categories"].items():
                if cat_info.get("embedding"):
                    similarity = cosine_similarity(
                        [new_embedding],
                        [cat_info["embedding"]]
                    )[0][0]

                    if similarity >= self.SIMILARITY_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = cat_id

            if best_match:
                print(f"Benzer konu bulundu: {best_match} (skor: {best_score:.2f})")

            return best_match

        except Exception as e:
            print(f"Benzerlik kontrolu hatasi: {e}")
            return None

    def _get_or_create_category(self, category_name: str) -> str:
        """
        Kategori al veya oluştur (benzerlik kontrolü ile)

        Returns:
            category_id
        """
        similar_cat = self._find_similar_category(category_name)
        if similar_cat:
            return similar_cat

        category_id = self._generate_category_id(category_name)

        if category_id in self.index["categories"]:
            return category_id

        embedding = None
        if self.embedder:
            try:
                embedding = self.embedder.encode(category_name).tolist()
            except Exception as e:
                print(f"Embedding oluşturma hatası ({category_name}): {e}")

        self.index["categories"][category_id] = {
            "name": category_name,
            "embedding": embedding,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "session_count": 0
        }

        self._save_index()
        print(f"Yeni konu olusturuldu: {category_name} ({category_id})")

        return category_id


    def _is_same_day(self, timestamp1: str, timestamp2: str) -> bool:
        """İki tarih aynı gün mü?"""
        try:
            date1 = datetime.fromisoformat(timestamp1).date()
            date2 = datetime.fromisoformat(timestamp2).date()
            return date1 == date2
        except ValueError as e:
            print(f"Tarih parse hatası: {e}")
            return False

    def _add_session_to_category(
        self,
        category_id: str,
        summary: str,
        messages: List[Dict]
    ):
        """Kategoriye yeni session ekle veya güncelle"""
        category_data = self._load_category(category_id)
        today = datetime.now().isoformat()
        today_date = datetime.now().strftime("%Y-%m-%d")

        existing_session = None
        for i, session in enumerate(category_data["sessions"]):
            if self._is_same_day(session.get("date", ""), today):
                existing_session = i
                break

        if existing_session is not None:
            old_session = category_data["sessions"][existing_session]
            old_session["summary"] = summary  # Eski: ekleme yapıyordu, şimdi üstüne yazıyor
            old_session["messages_count"] = old_session.get("messages_count", 0) + len(messages)
            old_session["last_updated"] = today

            print(f"Session guncellendi: {today_date} ({category_id})")
        else:
            new_session = {
                "date": today,
                "date_display": today_date,
                "summary": summary,
                "messages_count": len(messages)
            }
            category_data["sessions"].append(new_session)

            if len(category_data["sessions"]) > 50:
                category_data["sessions"] = category_data["sessions"][-50:]

            print(f"Yeni session eklendi: {today_date} ({category_id})")

        self._save_category(category_id, category_data)

        if category_id in self.index["categories"]:
            self.index["categories"][category_id]["last_updated"] = today
            self.index["categories"][category_id]["session_count"] = len(category_data["sessions"])
            self._save_index()


    def save_topic(
        self,
        messages: List[Dict],
        topic_hint: str = "",
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Konuşmayı uzun dönem hafızaya kaydet

        Args:
            messages: Konuşma mesajları [{"role": "user/assistant", "content": "..."}]
            topic_hint: Opsiyonel konu ipucu
            force: True ise kalite kontrolünü atla

        Returns:
            Kayıt bilgisi veya None
        """
        if not messages:
            return None

        if not force:
            is_worth, reason = self.is_worth_saving(messages)
            if not is_worth:
                print(f"Kayit atildi: {reason}")
                return None

        print(f"Konu kaydediliyor ({len(messages)} mesaj)...")

        category_name, summary = self._detect_category_with_llm(messages, topic_hint)

        category_id = self._get_or_create_category(category_name)

        self._add_session_to_category(category_id, summary, messages)

        result = {
            "category_id": category_id,
            "category_name": category_name,
            "summary": summary,
            "messages_count": len(messages),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        print(f"Kaydedildi: [{category_name}] - {summary[:50]}...")

        self._trigger_archive_worker()

        return result

    def _trigger_archive_worker(self):
        """Arka planda arşiv işçisini tetikle"""
        try:
            import threading
            from archive_worker import ArchiveWorker

            def run_worker():
                try:
                    worker = ArchiveWorker(
                        user_id=self.user_id,
                        base_dir=os.path.dirname(os.path.dirname(self.memory_dir)),
                        dry_run=False
                    )
                    worker.run()
                except Exception as e:
                    print(f"[!] Archive worker hatasi: {e}")

            thread = threading.Thread(target=run_worker, daemon=True)
            thread.start()
            print(f"[Archive Worker] arka planda tetiklendi")

        except Exception as e:
            print(f"[!] Archive worker tetikleme hatasi: {e}")

    def _extract_key_points(self, messages: List[Dict], max_points: int = 5) -> List[str]:
        """Mesajlardan anahtar noktaları çıkar (basit)"""
        key_points = []

        for msg in messages:
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content and len(content) > 15 and not self._is_trivial_message(content):
                    if "?" in content or any(w in content.lower() for w in ["nedir", "nasıl", "neden", "ne"]):
                        point = content[:200]
                        if point not in key_points:
                            key_points.append(point)

        return key_points[:max_points]

    def get_context_for_query(self, query: str, max_sessions: int = 3) -> str:
        """
        Sorgu için ilgili hafıza bağlamını getir

        NOT: Bu bilgiyi zorla söyleme, sadece LLM context'e ekle
        """
        if not self.embedder or not self.index.get("categories"):
            return ""

        if len(query.strip()) < 15:
            return ""

        try:
            query_embedding = self.embedder.encode(query)
            query_lower = query.lower()

            category_scores = []
            for cat_id, cat_info in self.index["categories"].items():
                if cat_info.get("embedding"):
                    similarity = cosine_similarity(
                        [query_embedding],
                        [cat_info["embedding"]]
                    )[0][0]

                    cat_name = cat_info.get("name", "").lower()
                    cat_keywords = cat_name.replace("/", " ").replace("-", " ").split()
                    keyword_match = any(kw in query_lower for kw in cat_keywords if len(kw) > 3)

                    threshold = 0.65 if keyword_match else 0.75

                    if similarity >= threshold:
                        category_scores.append((cat_id, cat_info, similarity))

            if not category_scores:
                return ""

            category_scores.sort(key=lambda x: x[2], reverse=True)
            top_categories = category_scores[:2]  # En fazla 2 kategori

            context_parts = []

            for cat_id, cat_info, score in top_categories:
                cat_data = self._load_category(cat_id)
                sessions = cat_data.get("sessions", [])[-max_sessions:]  # Son N session

                if sessions:
                    cat_name = cat_info.get("name", cat_id)
                    session_summaries = []

                    for s in sessions:
                        date = s.get("date_display", "")
                        summary = s.get("summary", "")[:400]
                        session_summaries.append(f"  - {date}: {summary}")

                    context_parts.append(f"[{cat_name}]\n" + "\n".join(session_summaries))

            if context_parts:
                return "Gecmis konusmalar (referans):\n" + "\n\n".join(context_parts)

            return ""

        except Exception as e:
            print(f"Context getirme hatasi: {e}")
            return ""

    def get_categories(self) -> List[Dict[str, Any]]:
        """Tüm kategorileri listele"""
        categories = []
        for cat_id, cat_info in self.index.get("categories", {}).items():
            categories.append({
                "id": cat_id,
                "name": cat_info.get("name", cat_id),
                "session_count": cat_info.get("session_count", 0),
                "last_updated": cat_info.get("last_updated", "")
            })

        categories.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        return categories

    def get_category_sessions(self, category_id: str) -> List[Dict[str, Any]]:
        """Belirli kategorinin session'larını getir"""
        cat_data = self._load_category(category_id)
        return cat_data.get("sessions", [])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tüm hafızada ara

        Returns:
            En alakalı session'lar
        """
        if not self.embedder:
            return []

        try:
            query_embedding = self.embedder.encode(query)
            all_results = []

            for cat_id, cat_info in self.index.get("categories", {}).items():
                cat_data = self._load_category(cat_id)

                for session in cat_data.get("sessions", []):
                    summary = session.get("summary", "")
                    if summary:
                        summary_embedding = self.embedder.encode(summary)
                        similarity = cosine_similarity(
                            [query_embedding],
                            [summary_embedding]
                        )[0][0]

                        if similarity >= 0.70:  # Sadece gerçekten aynı konu olduğunda gelsin
                            all_results.append({
                                "category_id": cat_id,
                                "category_name": cat_info.get("name", cat_id),
                                "date": session.get("date_display", ""),
                                "summary": summary,
                                "score": float(similarity)
                            })

            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            print(f"Arama hatasi: {e}")
            return []

    def clear_all(self):
        """Tüm hafızayı temizle"""
        self.index = {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "categories": {}
        }
        self._save_index()

        import shutil
        if os.path.exists(self.categories_dir):
            shutil.rmtree(self.categories_dir)
        os.makedirs(self.categories_dir, exist_ok=True)

        print("Tum hafiza temizlendi")

    def get_stats(self) -> Dict[str, Any]:
        """Hafıza istatistikleri"""
        total_sessions = 0
        for cat_id in self.index.get("categories", {}):
            cat_data = self._load_category(cat_id)
            total_sessions += len(cat_data.get("sessions", []))

        return {
            "category_count": len(self.index.get("categories", {})),
            "total_sessions": total_sessions,
            "version": self.index.get("version", "unknown"),
            "created": self.index.get("created", ""),
            "last_updated": self.index.get("last_updated", "")
        }



class TopicArchiverCompat:
    """
    TopicArchiver uyumluluk wrapper'ı

    Eski TopicArchiver API'sini korur, ama yeni TopicMemory'yi kullanır
    """

    def __init__(
        self,
        user_id: str = "default",
        archive_dir: str = "user_data",
        together_api_key: str = None,
        together_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        embedding_model: str = "BAAI/bge-m3"
    ):
        self.memory = TopicMemory(
            user_id=user_id,
            base_dir=archive_dir,
            together_api_key=together_api_key,
            together_model=together_model,
            embedding_model=embedding_model
        )

    def archive_topic(
        self,
        messages: List[Dict],
        topic_hint: str = "",
        force: bool = False
    ) -> Optional[Dict]:
        """Eski API: archive_topic → save_topic"""
        return self.memory.save_topic(messages, topic_hint, force)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Eski API: search"""
        return self.memory.search(query, top_k)

    def get_summary(self, query: str) -> Optional[str]:
        """Eski API: get_summary → get_context_for_query"""
        context = self.memory.get_context_for_query(query, max_sessions=1)
        return context if context else None

    def get_recent(self, count: int = 5) -> List[Dict]:
        """Eski API: get_recent"""
        results = self.memory.search("", top_k=count)
        return results

    def clear(self):
        """Eski API: clear"""
        self.memory.clear_all()



if __name__ == "__main__":
    print("=" * 60)
    print("TopicMemory Test")
    print("=" * 60)

    memory = TopicMemory(user_id="test_user")

    print("\n--- Test 1: Kalite Kontrolu ---")

    trivial_messages = [
        {"role": "user", "content": "Merhaba"},
        {"role": "assistant", "content": "Merhaba!"},
        {"role": "user", "content": "Tesekkurler"},
    ]

    is_worth, reason = memory.is_worth_saving(trivial_messages)
    print(f"Trivial mesajlar: {is_worth} - {reason}")

    print("\n--- Test 2: Anlamli Konusma ---")

    meaningful_messages = [
        {"role": "user", "content": "Python'da list comprehension nasil kullanilir?"},
        {"role": "assistant", "content": "List comprehension, Python'da listeleri tek satirda olusturmanin zarif bir yoludur. Ornek: [x*2 for x in range(10)]"},
        {"role": "user", "content": "Peki nested list comprehension nasil yapilir?"},
        {"role": "assistant", "content": "Nested list comprehension icin ic ice donguler kullanabilirsin: [[j for j in range(3)] for i in range(3)]"},
    ]

    is_worth, reason = memory.is_worth_saving(meaningful_messages)
    print(f"Anlamli mesajlar: {is_worth} - {reason}")

    print("\n--- Test 3: Kaydetme ---")
    result = memory.save_topic(meaningful_messages)
    if result:
        print(f"Kaydedildi: {result}")

    print("\n--- Test 4: Arama ---")
    search_results = memory.search("Python list")
    print(f"Arama sonuclari: {len(search_results)} sonuc")
    for r in search_results:
        print(f"  - [{r['category_name']}] {r['summary'][:50]}... (skor: {r['score']:.2f})")

    print("\n--- Test 5: Istatistikler ---")
    stats = memory.get_stats()
    print(f"Stats: {stats}")

    print("\n" + "=" * 60)
    print("Test tamamlandi!")
    print("=" * 60)
