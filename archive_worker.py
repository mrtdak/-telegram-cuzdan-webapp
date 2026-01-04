"""
Archive Worker - Arşiv İşçisi
Bağımsız çalışan, topic_memory'yi temizleyen/düzenleyen script.

Görevler:
1. Yedek al (topic_memory_backup/)
2. Duplicate kategorileri birleştir
3. Karışık konuları ayır
4. Temizlenmiş haliyle kaydet

Kullanım:
    python archive_worker.py --user murat
    python archive_worker.py --user murat --dry-run  (sadece rapor, değişiklik yok)
"""

import json
import os
import shutil
import argparse
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


@dataclass
class CategoryInfo:
    """Kategori bilgisi"""
    category_id: str
    name: str
    file_path: str
    sessions: List[Dict]
    summary: str
    total_messages: int


class ArchiveWorker:
    """
    Arşiv İşçisi - Topic Memory'yi temizler ve düzenler
    """

    def __init__(
        self,
        user_id: str,
        base_dir: str = "user_data",
        together_api_key: str = None,
        together_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        dry_run: bool = False
    ):
        self.user_id = user_id
        self.base_dir = base_dir
        self.together_api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
        self.together_model = together_model
        self.dry_run = dry_run

        # Dizinler
        self.topic_memory_dir = os.path.join(base_dir, f"user_{user_id}", "topic_memory")
        self.categories_dir = os.path.join(self.topic_memory_dir, "categories")
        self.backup_dir = os.path.join(base_dir, f"user_{user_id}", "topic_memory_backup")

        print(f"{'='*60}")
        print(f"ARŞİV İŞÇİSİ BAŞLATILDI")
        print(f"{'='*60}")
        print(f"Kullanıcı: {user_id}")
        print(f"Dizin: {self.categories_dir}")
        print(f"Mod: {'DRY-RUN (değişiklik yok)' if dry_run else 'GERÇEK'}")
        print(f"{'='*60}\n")

    # ==================== YEDEKLEME ====================

    def backup(self) -> bool:
        """Topic memory'yi yedekle"""
        if not os.path.exists(self.categories_dir):
            print(f"[!] Kategori dizini bulunamadı: {self.categories_dir}")
            return False

        # Yedek dizini oluştur (tarih damgalı)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")

        try:
            os.makedirs(backup_path, exist_ok=True)

            # Tüm dosyaları kopyala
            file_count = 0
            for filename in os.listdir(self.categories_dir):
                if filename.endswith('.json'):
                    src = os.path.join(self.categories_dir, filename)
                    dst = os.path.join(backup_path, filename)
                    shutil.copy2(src, dst)
                    file_count += 1

            # Index dosyasını da yedekle
            index_file = os.path.join(self.topic_memory_dir, "topics_index.json")
            if os.path.exists(index_file):
                shutil.copy2(index_file, os.path.join(backup_path, "topics_index.json"))

            print(f"[OK] Yedek alındı: {backup_path}")
            print(f"     {file_count} kategori dosyası yedeklendi")

            # Eski yedekleri temizle (dry-run modunda değilse)
            if not self.dry_run:
                self._cleanup_old_backups(keep=3)
            else:
                print("[DRY-RUN] Eski yedek temizleme atlandı\n")

            return True

        except Exception as e:
            print(f"[X] Yedekleme hatası: {e}")
            return False

    def _cleanup_old_backups(self, keep: int = 3):
        """Eski yedekleri temizle, sadece son N tanesini tut"""
        try:
            if not os.path.exists(self.backup_dir):
                return

            # Tüm yedek klasörlerini listele
            backups = []
            for name in os.listdir(self.backup_dir):
                path = os.path.join(self.backup_dir, name)
                if os.path.isdir(path) and name.startswith("backup_"):
                    backups.append(name)

            # Tarihe göre sırala (yeniden eskiye)
            backups.sort(reverse=True)

            # Silinecekleri bul
            to_delete = backups[keep:]

            if to_delete:
                print(f"[...] Eski yedekler temizleniyor ({len(to_delete)} adet)...")
                for backup_name in to_delete:
                    backup_path = os.path.join(self.backup_dir, backup_name)
                    shutil.rmtree(backup_path)
                    print(f"     - {backup_name} silindi")
                print(f"[OK] {len(backups) - len(to_delete)} yedek kaldı\n")
            else:
                print(f"[OK] Temizlenecek eski yedek yok ({len(backups)} mevcut)\n")

        except Exception as e:
            print(f"[!] Yedek temizleme hatası: {e}\n")

    # ==================== KATEGORİ OKUMA ====================

    def load_all_categories(self) -> List[CategoryInfo]:
        """Tüm kategorileri oku"""
        categories = []

        if not os.path.exists(self.categories_dir):
            return categories

        for filename in os.listdir(self.categories_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.categories_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Session'lardan özet çıkar
                sessions = data.get("sessions", [])
                summaries = [s.get("summary", "") for s in sessions if s.get("summary")]
                combined_summary = " | ".join(summaries) if summaries else data.get("consolidated_summary", "")

                total_msgs = sum(s.get("messages_count", 0) for s in sessions)
                if not total_msgs:
                    total_msgs = data.get("total_messages", 0)

                cat_info = CategoryInfo(
                    category_id=data.get("category_id", filename.replace(".json", "")),
                    name=data.get("category_name", data.get("category_id", "")),
                    file_path=filepath,
                    sessions=sessions,
                    summary=combined_summary,
                    total_messages=total_msgs
                )
                categories.append(cat_info)

            except Exception as e:
                print(f"[!] Dosya okuma hatası ({filename}): {e}")

        print(f"[OK] {len(categories)} kategori yüklendi\n")
        return categories

    # ==================== LLM ANALİZ ====================

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Together.ai API çağrısı"""
        if not self.together_api_key:
            print("[X] TOGETHER_API_KEY bulunamadı! Environment variable set edin.")
            return ""

        try:
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.together_model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["text"].strip()
                return result
            else:
                print(f"[X] API hata kodu: {response.status_code}")
                print(f"    Yanıt: {response.text[:200]}")

        except Exception as e:
            print(f"[X] LLM hatası: {e}")

        return ""

    def find_duplicates(self, categories: List[CategoryInfo]) -> List[Tuple[CategoryInfo, CategoryInfo, float]]:
        """Duplicate/benzer kategorileri bul"""
        if len(categories) < 2:
            return []

        print("[...] Duplicate analizi yapılıyor...")

        # Kategori listesi oluştur
        cat_list = "\n".join([
            f"{i+1}. [{c.category_id}] {c.name}: {c.summary[:100]}..."
            for i, c in enumerate(categories)
        ])

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Aşağıdaki kategorileri analiz et ve SADECE GERÇEKTEN AYNI KONUYU içerenleri bul.

KATEGORİLER:
{cat_list}

⚠️ ÖNEMLİ KURALLAR:
1. SADECE aynı ana konuyu (AI, define avcılığı, programlama vb.) içeren kategorileri eşleştir
2. "AI rekabeti" ile "define avcılığı" FARKLI konulardır - BİRLEŞTİRME!
3. Kategori adlarındaki anahtar kelimelere dikkat et
4. Emin değilsen BİRLEŞTİRME - yanlış birleştirme tehlikelidir
5. Sadece %90+ benzerlik varsa birleştir

ÖRNEKLER:
- "ai_rekabeti" + "ai_sistemleri_rekabeti" = AYNI KONU (ikisi de AI hakkında) ✓
- "define_avciligi" + "define_yasal" = AYNI KONU (ikisi de define hakkında) ✓
- "ai_rekabeti" + "define_avciligi" = FARKLI KONU (birleştirme!) ✗

FORMAT (JSON):
{{
    "duplicates": [
        {{"ids": [1, 2], "reason": "İkisi de AI/yapay zeka hakkında"}}
    ]
}}

Eğer kesin duplicate yoksa: {{"duplicates": []}}

SADECE JSON döndür.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        result = self._call_llm(prompt)

        # Debug: LLM yanıtını göster
        print(f"[DEBUG] LLM yanıtı: {result[:300]}..." if result else "[DEBUG] LLM yanıt vermedi!")

        duplicates = []
        try:
            import re
            # Daha esnek JSON çıkarma - nested objeleri destekler
            # Önce direkt parse dene
            data = None

            # Yöntem 1: Direkt JSON parse
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                pass  # Diğer yöntemleri dene

            # Yöntem 2: JSON bloğunu bul (```json ... ``` veya { ... })
            if not data:
                # Code block içinde mi?
                code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
                if code_match:
                    try:
                        data = json.loads(code_match.group(1))
                    except json.JSONDecodeError:
                        pass  # Diğer yöntemleri dene

            # Yöntem 3: İlk { ile son } arasını al
            if not data:
                start = result.find('{')
                end = result.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(result[start:end+1])
                    except json.JSONDecodeError:
                        pass  # JSON parse edilemedi

            if data and "duplicates" in data:
                for dup in data.get("duplicates", []):
                    ids = dup.get("ids", [])
                    if len(ids) >= 2:
                        # Index'ler 1'den başlıyor, 0'a çevir
                        idx1, idx2 = ids[0] - 1, ids[1] - 1
                        if 0 <= idx1 < len(categories) and 0 <= idx2 < len(categories):
                            duplicates.append((categories[idx1], categories[idx2], dup.get("reason", "")))
            else:
                print(f"[!] JSON'da 'duplicates' bulunamadı")

        except Exception as e:
            print(f"[!] JSON parse hatası: {e}")

        print(f"[OK] {len(duplicates)} duplicate bulundu\n")
        return duplicates

    def analyze_mixed_category(self, category: CategoryInfo) -> List[Dict]:
        """Karışık kategoriyi analiz et, farklı konuları ayır"""
        if not category.summary or len(category.summary) < 50:
            return []

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Bu kategori içinde FARKLI konular var mı? Analiz et.

KATEGORİ: {category.name}
İÇERİK: {category.summary[:500]}

GÖREV:
1. Bu kategoride kaç FARKLI konu var?
2. Her biri için ayrı kategori adı ve özet ver

FORMAT (JSON):
{{
    "is_mixed": true/false,
    "topics": [
        {{"name": "Konu 1 adı", "summary": "Konu 1 özeti"}},
        {{"name": "Konu 2 adı", "summary": "Konu 2 özeti"}}
    ]
}}

Eğer tek konu varsa: {{"is_mixed": false, "topics": []}}

SADECE JSON döndür.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        result = self._call_llm(prompt)

        try:
            import re
            json_match = re.search(r'\{.*"is_mixed".*"topics".*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("is_mixed") and data.get("topics"):
                    return data["topics"]
        except json.JSONDecodeError as e:
            print(f"Mixed kategori JSON parse hatası: {e}")

        return []

    # ==================== TEMİZLİK İŞLEMLERİ ====================

    def merge_categories(self, cat1: CategoryInfo, cat2: CategoryInfo, reason: str) -> bool:
        """İki kategoriyi birleştir"""
        print(f"[...] Birleştiriliyor: {cat1.name} + {cat2.name}")
        print(f"      Sebep: {reason}")

        if self.dry_run:
            print(f"[DRY-RUN] Birleştirme yapılmadı\n")
            return True

        try:
            # cat1'e cat2'yi ekle
            with open(cat1.file_path, 'r', encoding='utf-8') as f:
                data1 = json.load(f)

            with open(cat2.file_path, 'r', encoding='utf-8') as f:
                data2 = json.load(f)

            # Session'ları birleştir
            sessions1 = data1.get("sessions", [])
            sessions2 = data2.get("sessions", [])
            data1["sessions"] = sessions1 + sessions2

            # Mesaj sayısını güncelle
            total = data1.get("total_messages", 0) + data2.get("total_messages", 0)
            data1["total_messages"] = total

            # Kaydet
            with open(cat1.file_path, 'w', encoding='utf-8') as f:
                json.dump(data1, f, ensure_ascii=False, indent=2)

            # cat2'yi sil
            os.remove(cat2.file_path)

            print(f"[OK] Birleştirildi -> {cat1.name}\n")
            return True

        except Exception as e:
            print(f"[X] Birleştirme hatası: {e}\n")
            return False

    def split_category(self, category: CategoryInfo, topics: List[Dict]) -> bool:
        """Karışık kategoriyi ayır"""
        print(f"[...] Ayrılıyor: {category.name} -> {len(topics)} konu")

        if self.dry_run:
            for t in topics:
                print(f"      - {t.get('name')}")
            print(f"[DRY-RUN] Ayırma yapılmadı\n")
            return True

        try:
            # Her konu için yeni dosya oluştur
            for topic in topics:
                topic_name = topic.get("name", "Bilinmeyen")
                topic_summary = topic.get("summary", "")

                # Kategori ID oluştur
                cat_id = topic_name.lower()
                for tr, en in [('ı','i'),('ğ','g'),('ü','u'),('ş','s'),('ö','o'),('ç','c')]:
                    cat_id = cat_id.replace(tr, en)
                import re
                cat_id = re.sub(r'[^a-z0-9]+', '_', cat_id)[:50]

                new_file = os.path.join(self.categories_dir, f"{cat_id}.json")

                # Yeni kategori verisi
                new_data = {
                    "category_id": cat_id,
                    "category_name": topic_name,
                    "sessions": [{
                        "date": datetime.now().isoformat(),
                        "date_display": datetime.now().strftime("%Y-%m-%d"),
                        "summary": topic_summary,
                        "messages_count": category.total_messages // len(topics)
                    }]
                }

                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

                print(f"      + {topic_name} -> {cat_id}.json")

            # Eski kategoriyi sil
            os.remove(category.file_path)

            print(f"[OK] Ayrıldı\n")
            return True

        except Exception as e:
            print(f"[X] Ayırma hatası: {e}\n")
            return False

    # ==================== INDEX GÜNCELLEME ====================

    def update_index(self) -> bool:
        """topics_index.json dosyasını güncel kategorilere göre yeniden oluştur"""
        print("[...] Index güncelleniyor...")

        if self.dry_run:
            print("[DRY-RUN] Index güncellenmedi\n")
            return True

        index_file = os.path.join(self.topic_memory_dir, "topics_index.json")

        try:
            # Embedding model'i yükle
            embedder = None
            try:
                from sentence_transformers import SentenceTransformer
                print("[...] Embedding modeli yükleniyor...")
                embedder = SentenceTransformer("BAAI/bge-m3")
            except Exception as e:
                print(f"[!] Embedding modeli yüklenemedi: {e}")
                print("    Embedding'ler boş kalacak, ilk kullanımda oluşturulacak.")

            # Mevcut index'i oku (embedding'leri korumak için)
            old_index = {}
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    old_index = old_data.get("categories", {})

            # Yeni kategorileri yükle
            new_categories = {}
            for filename in os.listdir(self.categories_dir):
                if not filename.endswith('.json'):
                    continue

                cat_id = filename.replace('.json', '')
                filepath = os.path.join(self.categories_dir, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cat_name = data.get("category_name", cat_id)

                # Eski index'te embedding varsa kullan
                embedding = []
                if cat_id in old_index and old_index[cat_id].get("embedding"):
                    embedding = old_index[cat_id]["embedding"]
                elif embedder:
                    # Yeni embedding oluştur
                    print(f"    + Embedding: {cat_name}")
                    embedding = embedder.encode(cat_name).tolist()

                new_categories[cat_id] = {
                    "name": cat_name,
                    "embedding": embedding
                }

            # Yeni index'i yaz
            new_index = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "categories": new_categories
            }

            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(new_index, f, ensure_ascii=False, indent=2)

            print(f"[OK] Index güncellendi: {len(new_categories)} kategori\n")
            return True

        except Exception as e:
            print(f"[X] Index güncelleme hatası: {e}\n")
            return False

    # ==================== ANA FONKSİYON ====================

    def run(self):
        """Arşiv işçisini çalıştır"""
        print("\n" + "="*60)
        print("ADIM 1: YEDEKLEME")
        print("="*60)

        if not self.backup():
            print("[X] Yedekleme başarısız, işlem durduruluyor.")
            return

        print("\n" + "="*60)
        print("ADIM 2: KATEGORİLERİ YÜKLE")
        print("="*60)

        categories = self.load_all_categories()
        if not categories:
            print("[!] Kategori bulunamadı.")
            return

        for cat in categories:
            print(f"  [{cat.category_id}] {cat.name}")
            print(f"      Mesaj: {cat.total_messages} | Özet: {cat.summary[:50]}...")

        print("\n" + "="*60)
        print("ADIM 3: DUPLICATE ANALİZİ")
        print("="*60)

        duplicates = self.find_duplicates(categories)

        for cat1, cat2, reason in duplicates:
            print(f"  ! {cat1.name} <=> {cat2.name}")
            print(f"    Sebep: {reason}")
            self.merge_categories(cat1, cat2, reason)

        print("\n" + "="*60)
        print("ADIM 4: KARIŞIK KONU ANALİZİ")
        print("="*60)

        # Kategorileri yeniden yükle (birleştirmeden sonra)
        categories = self.load_all_categories()

        for cat in categories:
            topics = self.analyze_mixed_category(cat)
            if topics and len(topics) > 1:
                print(f"  ! {cat.name} karışık ({len(topics)} farklı konu)")
                self.split_category(cat, topics)

        print("\n" + "="*60)
        print("ADIM 5: INDEX GÜNCELLEME")
        print("="*60)

        self.update_index()

        print("\n" + "="*60)
        print("ARŞİV İŞÇİSİ TAMAMLANDI")
        print("="*60)

        if self.dry_run:
            print("\n[!] DRY-RUN modunda çalıştı, değişiklik yapılmadı.")
            print("    Gerçek çalıştırmak için: python archive_worker.py --user murat")


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arşiv İşçisi - Topic Memory Temizleyici")
    parser.add_argument("--user", type=str, default="murat", help="Kullanıcı ID")
    parser.add_argument("--dry-run", action="store_true", help="Sadece rapor ver, değişiklik yapma")
    parser.add_argument("--base-dir", type=str, default="user_data", help="Veri dizini")

    args = parser.parse_args()

    worker = ArchiveWorker(
        user_id=args.user,
        base_dir=args.base_dir,
        dry_run=args.dry_run
    )

    worker.run()
