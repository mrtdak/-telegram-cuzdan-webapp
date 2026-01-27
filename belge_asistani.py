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
        self.chunks = []    # [{belge_id, chunk_index, text}]

        self._load()

    def _load(self):
        """Index ve metadata yükle"""
        # Metadata yükle
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.belgeler = json.load(f)

        # Chunks yükle
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

        # FAISS index yükle veya oluştur
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

    def _save(self):
        """Index ve metadata kaydet"""
        # Metadata kaydet
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.belgeler, f, ensure_ascii=False, indent=2)

        # Chunks kaydet
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # FAISS index kaydet
        if self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_file)

    def _generate_id(self, filename: str) -> str:
        """Dosya için unique ID oluştur"""
        hash_input = f"{filename}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

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

        # Mevcut chunk sayısı (yeni chunk'ların başlangıç indexi)
        start_idx = len(self.chunks)

        # Chunk'ları kaydet
        for i, chunk_text in enumerate(text_chunks):
            self.chunks.append({
                "belge_id": belge_id,
                "chunk_index": i,
                "text": chunk_text
            })

        # Embedding oluştur
        embeddings = self.model.encode(text_chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')

        # FAISS'e ekle
        self.index.add(embeddings)

        # Metadata kaydet
        self.belgeler[belge_id] = {
            "dosya_adi": filename,
            "yukleme_tarihi": datetime.now().isoformat(),
            "chunk_sayisi": len(text_chunks),
            "karakter_sayisi": len(text),
            "start_idx": start_idx
        }

        self._save()

        return {
            "success": True,
            "belge_id": belge_id,
            "dosya_adi": filename,
            "chunk_sayisi": len(text_chunks),
            "karakter_sayisi": len(text)
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
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
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
        # Chunk'ları birleştir
        icerik = ""
        for chunk in self.chunks:
            if chunk["belge_id"] == belge_id and not chunk.get("deleted"):
                icerik += chunk["text"] + "\n\n"

        return {
            "success": True,
            "belge_id": belge_id,
            "dosya_adi": meta["dosya_adi"],
            "icerik": icerik.strip(),
            "chunk_sayisi": meta["chunk_sayisi"]
        }

    def sil(self, belge_id: str) -> Dict:
        """Belge sil (FAISS'ten silme karmaşık, sadece metadata'dan sil)"""
        if belge_id not in self.belgeler:
            return {"success": False, "error": "Belge bulunamadı"}

        dosya_adi = self.belgeler[belge_id]["dosya_adi"]
        del self.belgeler[belge_id]

        # Chunk'ları işaretle (silmek yerine)
        # Not: Gerçek silme için index rebuild gerekir
        for chunk in self.chunks:
            if chunk["belge_id"] == belge_id:
                chunk["deleted"] = True

        self._save()

        return {"success": True, "dosya_adi": dosya_adi}

    def rebuild_index(self):
        """Index'i yeniden oluştur (silinen belgeler için)"""
        # Silinmemiş chunk'ları filtrele
        active_chunks = [c for c in self.chunks if not c.get("deleted")]

        if not active_chunks:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self._save()
            return

        # Yeni embeddings
        texts = [c["text"] for c in active_chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')

        # Yeni index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Chunk'ları güncelle
        self.chunks = active_chunks

        self._save()


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
