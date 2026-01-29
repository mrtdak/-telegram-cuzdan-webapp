"""
Belge Asistanı - PDF/Doküman Yükleme, Embedding ve Arama Sistemi
Mini prototype - bağımsız test edilebilir modül
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import requests

# Lazy imports for dependencies
faiss = None
SentenceTransformer = None

def _load_dependencies():
    """Bağımlılıkları lazy load et"""
    global faiss, SentenceTransformer
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST

# Text extraction functions
def extract_text_from_pdf(file_path: str) -> str:
    """PDF'den metin çıkar"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        # PyMuPDF alternatifi
        try:
            import fitz  # PyMuPDF
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except ImportError:
            raise ImportError("PDF okumak için 'PyPDF2' veya 'PyMuPDF' gerekli: pip install PyPDF2")

def extract_text_from_docx(file_path: str) -> str:
    """DOCX'den metin çıkar"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise ImportError("DOCX okumak için 'python-docx' gerekli: pip install python-docx")

def extract_text_from_txt(file_path: str) -> str:
    """TXT dosyasından metin çıkar"""
    encodings = ['utf-8', 'utf-8-sig', 'cp1254', 'iso-8859-9', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Dosya okunamadı: {file_path}")

def extract_text(file_path: str) -> str:
    """Dosya türüne göre metin çıkar"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Metni parçalara böl"""
    if not text:
        return []

    # Önce paragraflara böl
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Paragraf chunk_size'dan küçükse birleştir
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += (" " if current_chunk else "") + para
        else:
            # Mevcut chunk'ı kaydet
            if current_chunk:
                chunks.append(current_chunk)

            # Paragraf çok uzunsa böl
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) < chunk_size:
                        current_chunk += (" " if current_chunk else "") + word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = para

    # Son chunk'ı ekle
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class BelgeAsistani:
    """Belge yönetimi ve arama sınıfı"""

    def __init__(self, data_dir: str = "belge_data"):
        _load_dependencies()

        self.data_dir = data_dir
        self.index_file = os.path.join(data_dir, "faiss_index.bin")
        self.meta_file = os.path.join(data_dir, "belgeler.json")
        self.chunks_file = os.path.join(data_dir, "chunks.json")

        os.makedirs(data_dir, exist_ok=True)

        # Embedding modeli
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.dimension = 1024  # bge-m3 dimension

        # FAISS index ve metadata yükle
        self.index = None
        self.belgeler = {}  # {belge_id: {dosya_adi, yukleme_tarihi, chunk_sayisi}}
        self.chunks = {}    # {vector_id: {belge_id, chunk_index, text}} - artık dict
        self.next_id = 0    # Sonraki vector ID

        # Aktif belge (soru-cevap modu)
        self.aktif_belge_id = None
        self.aktif_belge_baslangic = None  # Timestamp
        self.aktif_belge_mesaj_sayisi = 0
        self.TIMEOUT_MESAJ = 15  # 15 mesaj sonra kapat
        self.TIMEOUT_DAKIKA = 30  # 30 dakika sonra kapat

        self._load()

    def _load(self):
        """Index ve metadata yükle"""
        # Metadata yükle
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.belgeler = json.load(f)

        # Chunks yükle (dict olarak)
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Eski format (list) ise dict'e çevir
                if isinstance(data, list):
                    self.chunks = {}
                    for i, chunk in enumerate(data):
                        if not chunk.get("deleted"):
                            self.chunks[str(i)] = chunk
                    self.next_id = len(data)
                else:
                    self.chunks = data
                    # next_id'yi mevcut maksimum ID'den hesapla
                    if self.chunks:
                        self.next_id = max(int(k) for k in self.chunks.keys()) + 1

        # FAISS index yükle veya oluştur (IndexIDMap)
        needs_rebuild = False
        if os.path.exists(self.index_file):
            loaded_index = faiss.read_index(self.index_file)
            # IndexIDMap mı kontrol et (tip adına bak)
            index_type = type(loaded_index).__name__
            if 'IDMap' in index_type:
                self.index = loaded_index
            else:
                print(f"[BELGE] Eski index formatı ({index_type}), IndexIDMap'e geçiliyor...")
                needs_rebuild = True
        else:
            needs_rebuild = True

        if needs_rebuild:
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            # Mevcut chunk'lar varsa yeniden indexle
            if self.chunks:
                self._rebuild_from_chunks()

    def _rebuild_from_chunks(self):
        """Mevcut chunk'lardan index'i yeniden oluştur"""
        if not self.chunks:
            return

        texts = []
        ids = []
        for vid, chunk in self.chunks.items():
            texts.append(chunk["text"])
            ids.append(int(vid))

        if texts:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype('float32')
            self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
            print(f"[BELGE] {len(texts)} chunk yeniden indexlendi")

    def _save(self):
        """Index ve metadata kaydet"""
        # Metadata kaydet
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.belgeler, f, ensure_ascii=False, indent=2)

        # Chunks kaydet
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # FAISS index kaydet (boş olsa bile)
        faiss.write_index(self.index, self.index_file)

    def _generate_id(self, filename: str) -> str:
        """Dosya için unique ID oluştur"""
        hash_input = f"{filename}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def _olustur_ozet(self, text: str) -> str:
        """LLM ile belge özeti oluştur"""
        try:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                return "Özet oluşturulamadı (API key yok)"

            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Sen bir belge özetleme asistanısın. Verilen metni 2-3 cümleyle özetle. Türkçe yaz. Sadece özeti yaz, başka bir şey ekleme."
                        },
                        {
                            "role": "user",
                            "content": f"Bu belgeyi özetle:\n\n{text}"
                        }
                    ],
                    "max_tokens": 200,
                    "temperature": 0.3
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                ozet = data["choices"][0]["message"]["content"].strip()
                return ozet
            else:
                print(f"[ÖZET] API hatası: {response.status_code}")
                return "Özet oluşturulamadı"

        except Exception as e:
            print(f"[ÖZET] Hata: {e}")
            return "Özet oluşturulamadı"

    def get_ozet(self, belge_id: str) -> Optional[str]:
        """Belgenin özetini döndür"""
        if belge_id not in self.belgeler:
            return None
        return self.belgeler[belge_id].get("ozet")

    def belge_yukle(self, file_path: str) -> Dict:
        """Belge yükle ve indexle"""
        if not os.path.exists(file_path):
            return {"success": False, "error": "Dosya bulunamadı"}

        filename = os.path.basename(file_path)

        # Metin çıkar
        try:
            text = extract_text(file_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

        if not text:
            return {"success": False, "error": "Dosyadan metin çıkarılamadı"}

        # Chunk'lara böl
        text_chunks = split_into_chunks(text)

        if not text_chunks:
            return {"success": False, "error": "Metin parçalanamadı"}

        # Belge ID oluştur
        belge_id = self._generate_id(filename)

        # Chunk'ları kaydet ve ID'leri topla
        vector_ids = []
        for i, chunk_text in enumerate(text_chunks):
            vid = self.next_id
            self.next_id += 1
            vector_ids.append(vid)
            self.chunks[str(vid)] = {
                "belge_id": belge_id,
                "chunk_index": i,
                "text": chunk_text
            }

        # Embedding oluştur
        embeddings = self.model.encode(text_chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')

        # FAISS'e ID'lerle ekle
        ids = np.array(vector_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        # Özet oluştur (ilk 3000 karakter ile)
        ozet = self._olustur_ozet(text[:3000])

        # Metadata kaydet
        self.belgeler[belge_id] = {
            "dosya_adi": filename,
            "yukleme_tarihi": datetime.now().isoformat(),
            "chunk_sayisi": len(text_chunks),
            "karakter_sayisi": len(text),
            "vector_ids": vector_ids,  # Silme için gerekli
            "ozet": ozet  # Otomatik özet
        }

        self._save()

        return {
            "success": True,
            "belge_id": belge_id,
            "dosya_adi": filename,
            "chunk_sayisi": len(text_chunks),
            "karakter_sayisi": len(text),
            "ozet": ozet
        }

    def ara(self, sorgu: str, k: int = 5) -> List[Dict]:
        """Belgelerde arama yap"""
        if self.index.ntotal == 0:
            return []

        # Sorgu embedding
        query_embedding = self.model.encode([sorgu], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')

        # Arama
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # idx artık vector ID (int64)
            if idx < 0:
                continue

            chunk = self.chunks.get(str(idx))
            if not chunk:
                continue

            belge = self.belgeler.get(chunk["belge_id"], {})

            results.append({
                "belge_id": chunk["belge_id"],
                "dosya_adi": belge.get("dosya_adi", "Bilinmiyor"),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "skor": float(score)
            })

        return results

    def listele(self) -> List[Dict]:
        """Tüm belgeleri listele"""
        belgeler = []
        for belge_id, meta in self.belgeler.items():
            belgeler.append({
                "belge_id": belge_id,
                "dosya_adi": meta["dosya_adi"],
                "yukleme_tarihi": meta["yukleme_tarihi"],
                "chunk_sayisi": meta["chunk_sayisi"]
            })
        return belgeler

    def get_icerik(self, belge_id: str) -> Dict:
        """Belgenin tüm içeriğini getir"""
        if belge_id not in self.belgeler:
            return {"success": False, "error": "Belge bulunamadı"}

        meta = self.belgeler[belge_id]

        # Chunk'ları sıralı şekilde birleştir
        chunk_list = []
        for vid, chunk in self.chunks.items():
            if chunk["belge_id"] == belge_id:
                chunk_list.append((chunk["chunk_index"], chunk["text"]))

        chunk_list.sort(key=lambda x: x[0])
        icerik = "\n\n".join([text for _, text in chunk_list])

        return {
            "success": True,
            "belge_id": belge_id,
            "dosya_adi": meta["dosya_adi"],
            "icerik": icerik.strip(),
            "chunk_sayisi": meta["chunk_sayisi"]
        }

    def sil(self, belge_id: str) -> Dict:
        """Belge sil - FAISS'ten direkt silme (IndexIDMap sayesinde)"""
        if belge_id not in self.belgeler:
            return {"success": False, "error": "Belge bulunamadı"}

        meta = self.belgeler[belge_id]
        dosya_adi = meta["dosya_adi"]

        # FAISS'ten sil (vector_ids varsa)
        if "vector_ids" in meta:
            ids_to_remove = np.array(meta["vector_ids"], dtype=np.int64)
            self.index.remove_ids(ids_to_remove)

        # Chunk'ları sil
        vids_to_delete = [vid for vid, chunk in self.chunks.items()
                         if chunk["belge_id"] == belge_id]
        for vid in vids_to_delete:
            del self.chunks[vid]

        # Metadata'dan sil
        del self.belgeler[belge_id]

        self._save()

        return {"success": True, "dosya_adi": dosya_adi}

    def rebuild_index(self):
        """Index'i yeniden oluştur (eski versiyon migration veya bozulma için)"""
        if not self.chunks:
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            self._save()
            return

        # Chunk'ları topla
        texts = []
        ids = []
        for vid, chunk in self.chunks.items():
            texts.append(chunk["text"])
            ids.append(int(vid))

        # Yeni embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')

        # Yeni index (IndexIDMap)
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

        self._save()

    # ============ AKTİF BELGE (SORU-CEVAP MODU) ============

    def set_aktif(self, belge_id: str) -> Dict:
        """Belgeyi aktif yap - soru-cevap modu başlat"""
        if belge_id not in self.belgeler:
            return {"success": False, "error": "Belge bulunamadı"}

        self.aktif_belge_id = belge_id
        self.aktif_belge_baslangic = datetime.now()
        self.aktif_belge_mesaj_sayisi = 0

        meta = self.belgeler[belge_id]
        return {
            "success": True,
            "belge_id": belge_id,
            "dosya_adi": meta["dosya_adi"]
        }

    def clear_aktif(self):
        """Aktif belgeyi temizle"""
        self.aktif_belge_id = None
        self.aktif_belge_baslangic = None
        self.aktif_belge_mesaj_sayisi = 0

    def increment_mesaj(self):
        """Mesaj sayısını artır ve timeout kontrolü yap"""
        if not self.aktif_belge_id:
            return None

        self.aktif_belge_mesaj_sayisi += 1

        # Timeout kontrolü
        timeout_reason = self.check_timeout()
        return timeout_reason

    def check_timeout(self) -> Optional[str]:
        """Timeout oldu mu kontrol et. Olduysa sebebini döndür."""
        if not self.aktif_belge_id:
            return None

        # Süre limiti (30 dk) - otomatik kapat
        if self.aktif_belge_baslangic:
            gecen_dakika = (datetime.now() - self.aktif_belge_baslangic).total_seconds() / 60
            if gecen_dakika >= self.TIMEOUT_DAKIKA:
                dosya_adi = self.belgeler[self.aktif_belge_id]["dosya_adi"]
                self.clear_aktif()
                return f"sure_limit:{dosya_adi}"

        # Mesaj limiti (15 mesaj) - soru sor, kapatma
        if self.aktif_belge_mesaj_sayisi >= self.TIMEOUT_MESAJ:
            return "mesaj_limit_sor"  # Sadece uyarı, kapatma yok

        return None

    def reset_mesaj_sayaci(self):
        """Mesaj sayacını sıfırla (devam et seçildiğinde)"""
        self.aktif_belge_mesaj_sayisi = 0

    def get_aktif(self) -> Optional[Dict]:
        """Aktif belge bilgisini döndür"""
        if not self.aktif_belge_id:
            return None

        if self.aktif_belge_id not in self.belgeler:
            self.aktif_belge_id = None
            return None

        meta = self.belgeler[self.aktif_belge_id]
        return {
            "belge_id": self.aktif_belge_id,
            "dosya_adi": meta["dosya_adi"]
        }

    def get_context(self, sorgu: str, k: int = 3) -> Optional[str]:
        """
        Aktif belge üzerinden sorguya uygun context döndür.
        Soru-cevap için LLM'e verilecek metin.
        """
        if not self.aktif_belge_id:
            return None

        if self.index.ntotal == 0:
            return None

        # Sorgu embedding
        query_embedding = self.model.encode([sorgu], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')

        # Daha fazla sonuç al, sonra filtrele (sadece aktif belge)
        search_k = min(k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        # Aktif belgeye ait chunk'ları filtrele
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.chunks.get(str(idx))
            if not chunk:
                continue

            # Sadece aktif belgenin chunk'ları
            if chunk["belge_id"] != self.aktif_belge_id:
                continue

            relevant_chunks.append({
                "text": chunk["text"],
                "skor": float(score),
                "chunk_index": chunk["chunk_index"]
            })

            if len(relevant_chunks) >= k:
                break

        if not relevant_chunks:
            return None

        # Context metni oluştur
        meta = self.belgeler[self.aktif_belge_id]
        context_parts = [f"📄 Belge: {meta['dosya_adi']}\n"]

        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(f"[Bölüm {chunk['chunk_index'] + 1}]\n{chunk['text']}")

        return "\n\n".join(context_parts)


# Test fonksiyonları
def test_basic():
    """Basit test"""
    print("=" * 50)
    print("Belge Asistani Test")
    print("=" * 50)

    # Test dizini
    test_dir = "belge_test_data"

    # Asistan oluştur
    asistan = BelgeAsistani(data_dir=test_dir)
    print(f"\n[OK] Asistan olusturuldu (data_dir: {test_dir})")

    # Test TXT dosyası oluştur
    test_txt = os.path.join(test_dir, "test_belge.txt")
    with open(test_txt, 'w', encoding='utf-8') as f:
        f.write("""
        Python Programlama Dili

        Python, 1991 yilinda Guido van Rossum tarafindan gelistirilmis yuksek seviyeli
        bir programlama dilidir. Okunabilirligi ve basit sozdizimi ile bilinir.

        Python'un Ozellikleri:
        - Dinamik tip sistemi
        - Otomatik bellek yonetimi
        - Genis standart kutuphane
        - Coklu programlama paradigmasi destegi

        Python, web gelistirme, veri analizi, yapay zeka ve otomasyon gibi
        bircok alanda yaygin olarak kullanilmaktadir.

        Populer Python kutuphaneleri arasinda NumPy, Pandas, Django, Flask,
        TensorFlow ve PyTorch bulunmaktadir.
        """)
    print(f"[OK] Test dosyasi olusturuldu: {test_txt}")

    # Belge yükle
    result = asistan.belge_yukle(test_txt)
    print(f"\n[BELGE] Belge yukleme sonucu:")
    print(f"   - Basarili: {result['success']}")
    if result['success']:
        print(f"   - Belge ID: {result['belge_id']}")
        print(f"   - Chunk sayisi: {result['chunk_sayisi']}")
        print(f"   - Karakter sayisi: {result['karakter_sayisi']}")

    # Arama yap
    print("\n[ARAMA] Sorgu: 'Python ozellikleri nelerdir?'")
    sonuclar = asistan.ara("Python ozellikleri nelerdir?", k=3)
    for i, sonuc in enumerate(sonuclar, 1):
        print(f"\n   Sonuc {i} (skor: {sonuc['skor']:.3f}):")
        print(f"   {sonuc['text'][:200]}...")

    # Listele
    print("\n[LISTE] Belgeler:")
    for belge in asistan.listele():
        print(f"   - [{belge['belge_id']}] {belge['dosya_adi']} ({belge['chunk_sayisi']} chunk)")

    print("\n" + "=" * 50)
    print("Test tamamlandi!")
    print("=" * 50)


if __name__ == "__main__":
    test_basic()
