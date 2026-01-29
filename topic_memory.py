"""
Long Term Memory - Uzun Dönem Hafıza Sistemi
Tüm mesajları saklar, semantic search ile sorgular

Dosya Yapısı:
user_data/user_{id}/topic_memory/
  ├── messages.json    # Tüm mesajlar [{id, timestamp, role, content}, ...]
  └── embeddings.npy   # Numpy array [n_messages, embedding_dim]

Kullanım:
    memory = TopicMemory(user_id="murat")

    # Mesaj kaydet
    memory.save_message("user", "Python nedir?")
    memory.save_message("assistant", "Python bir programlama dilidir...")

    # Veya toplu kaydet
    memory.save_messages([
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ])

    # Arama
    results = memory.search("Python programlama")

    # Context getir (prompt için)
    context = memory.get_context("Python nedir?")
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("TopicMemory: sentence_transformers yok")


class TopicMemory:
    """
    Uzun Dönem Hafıza - Tüm mesajları saklar, semantic search ile sorgular
    """

    SIMILARITY_THRESHOLD = 0.60  # Minimum benzerlik eşiği

    def __init__(
        self,
        user_id: str = "default",
        base_dir: str = "user_data",
        embedding_model: str = "BAAI/bge-m3",
        **kwargs  # Eski API uyumluluğu için
    ):
        self.user_id = user_id

        self.memory_dir = os.path.join(base_dir, f"user_{user_id}", "topic_memory")
        self.messages_file = os.path.join(self.memory_dir, "messages.json")
        self.embeddings_file = os.path.join(self.memory_dir, "embeddings.npy")

        os.makedirs(self.memory_dir, exist_ok=True)

        # Mesajları yükle
        self.messages: List[Dict] = self._load_messages()

        # Embedding'leri yükle
        self.embeddings: Optional[np.ndarray] = self._load_embeddings()

        # Lazy load embedder
        self._embedder = None
        self._embedding_model_name = embedding_model

        print(f"TopicMemory baslatildi - {len(self.messages)} mesaj")

    @property
    def embedder(self):
        """Lazy load embedding model"""
        if self._embedder is None and EMBEDDINGS_AVAILABLE:
            print("Embedding modeli yukleniyor...")
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _load_messages(self) -> List[Dict]:
        """Mesajları yükle"""
        if os.path.exists(self.messages_file):
            try:
                with open(self.messages_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Mesaj yukleme hatasi: {e}")
        return []

    def _save_messages(self):
        """Mesajları kaydet"""
        try:
            with open(self.messages_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Mesaj kaydetme hatasi: {e}")

    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Embedding'leri yükle"""
        if os.path.exists(self.embeddings_file):
            try:
                return np.load(self.embeddings_file)
            except Exception as e:
                print(f"Embedding yukleme hatasi: {e}")
        return None

    def _save_embeddings(self):
        """Embedding'leri kaydet"""
        if self.embeddings is not None:
            try:
                np.save(self.embeddings_file, self.embeddings)
            except Exception as e:
                print(f"Embedding kaydetme hatasi: {e}")

    def save_message(self, role: str, content: str) -> bool:
        """
        Tek mesaj kaydet

        Args:
            role: "user" veya "assistant"
            content: Mesaj içeriği

        Returns:
            Başarılı mı
        """
        if not content or not content.strip():
            return False

        # Mesaj oluştur
        msg = {
            "id": len(self.messages),
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content.strip()
        }

        self.messages.append(msg)

        # Embedding oluştur
        if self.embedder:
            try:
                new_embedding = self.embedder.encode(content)

                if self.embeddings is None:
                    self.embeddings = np.array([new_embedding])
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embedding])

                self._save_embeddings()
            except Exception as e:
                print(f"Embedding olusturma hatasi: {e}")

        self._save_messages()
        return True

    def save_messages(self, messages: List[Dict]) -> int:
        """
        Toplu mesaj kaydet

        Args:
            messages: [{"role": "user/assistant", "content": "..."}]

        Returns:
            Kaydedilen mesaj sayısı
        """
        if not messages:
            return 0

        saved = 0
        new_embeddings = []

        # Son 20 mesajın content'lerini al (duplicate kontrolü için)
        recent_contents = set()
        for m in self.messages[-20:]:
            recent_contents.add(m.get("content", "").strip().lower())

        for msg in messages:
            content = msg.get("content", "")
            if content:
                content = content.strip()
            role = msg.get("role", "user")

            if not content:
                continue

            # Duplicate kontrolü - aynı mesaj son 20'de varsa atla
            if content.lower() in recent_contents:
                continue

            # Mesaj oluştur
            new_msg = {
                "id": len(self.messages),
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content
            }

            self.messages.append(new_msg)
            recent_contents.add(content.lower())  # Yeni eklenenler için de kontrol
            saved += 1

            # Embedding oluştur
            if self.embedder:
                try:
                    embedding = self.embedder.encode(content)
                    new_embeddings.append(embedding)
                except Exception as e:
                    print(f"Embedding hatasi: {e}")
                    new_embeddings.append(np.zeros(self.embedder.get_sentence_embedding_dimension()))

        # Embedding'leri birleştir
        if new_embeddings and self.embedder:
            new_emb_array = np.array(new_embeddings)

            if self.embeddings is None:
                self.embeddings = new_emb_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb_array])

            self._save_embeddings()

        self._save_messages()
        print(f"{saved} mesaj kaydedildi (toplam: {len(self.messages)})")
        return saved

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Semantic search

        Args:
            query: Arama sorgusu
            top_k: Maksimum sonuç sayısı

        Returns:
            En alakalı mesajlar [{message, score}, ...]
        """
        if not query or not self.embedder or self.embeddings is None or len(self.messages) == 0:
            return []

        try:
            # Sorgu embedding'i
            query_embedding = self.embedder.encode(query)

            # Cosine similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

            # En yüksek skorlu mesajları bul
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= self.SIMILARITY_THRESHOLD:
                    results.append({
                        "message": self.messages[idx],
                        "score": score
                    })

            return results

        except Exception as e:
            print(f"Arama hatasi: {e}")
            return []

    def get_context(self, query: str, max_messages: int = 5) -> str:
        """
        Sorgu için context oluştur (prompt'a eklenecek)

        Args:
            query: Kullanıcının sorusu
            max_messages: Maksimum mesaj sayısı

        Returns:
            Context string
        """
        if len(query.strip()) < 10:
            return ""

        results = self.search(query, top_k=max_messages * 2)

        if not results:
            return ""

        # Mesajları grupla
        context_parts = []
        seen_ids = set()

        for r in results[:max_messages]:
            msg = r["message"]
            msg_id = msg["id"]

            if msg_id in seen_ids:
                continue

            seen_ids.add(msg_id)

            # Tarih formatla
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                date_str = dt.strftime("%d.%m.%Y")
            except:
                date_str = ""

            role = "Kullanici" if msg["role"] == "user" else "Sen"
            content = msg["content"][:300]  # Max 300 karakter

            context_parts.append(f"[{date_str}] {role}: {content}")

        if context_parts:
            return "\n".join(context_parts)

        return ""
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikler"""
        user_count = sum(1 for m in self.messages if m.get("role") == "user")
        assistant_count = sum(1 for m in self.messages if m.get("role") == "assistant")

        return {
            "total_messages": len(self.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "has_embeddings": self.embeddings is not None,
            "embedding_count": len(self.embeddings) if self.embeddings is not None else 0
        }

    def clear_all(self):
        """Tüm hafızayı temizle"""
        self.messages = []
        self.embeddings = None

        if os.path.exists(self.messages_file):
            os.remove(self.messages_file)
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)

        print("Hafiza temizlendi")


if __name__ == "__main__":
    print("=" * 60)
    print("TopicMemory Test (Yeni Mimari)")
    print("=" * 60)

    memory = TopicMemory(user_id="test_new")

    # Test mesajları
    test_messages = [
        {"role": "user", "content": "Python'da list comprehension nasıl kullanılır?"},
        {"role": "assistant", "content": "List comprehension, Python'da listeleri tek satırda oluşturmanın zarif bir yoludur."},
        {"role": "user", "content": "Örnek verir misin?"},
        {"role": "assistant", "content": "Tabii, örnek: [x*2 for x in range(10)] - bu 0'dan 9'a kadar sayıların 2 katını içeren liste oluşturur."},
    ]

    print("\n--- Mesaj Kaydetme ---")
    memory.save_messages(test_messages)

    print("\n--- Arama ---")
    results = memory.search("Python liste")
    for r in results:
        print(f"  [{r['score']:.2f}] {r['message']['content'][:50]}...")

    print("\n--- Context ---")
    ctx = memory.get_context("Python'da liste nasıl yapılır?")
    print(ctx)

    print("\n--- İstatistikler ---")
    print(memory.get_stats())

    print("\n" + "=" * 60)
    print("Test tamamlandi!")
