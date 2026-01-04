"""
Conversation Context Manager - LLM Tabanlı Konuşma Bağlamı Yönetimi

Problem: Embedding tabanlı benzerlik, semantik olarak ilişkili konuları
(örn: "Allah'ın ilmi" → "kader" → "irade") farklı konu olarak algılıyor.

Çözüm: LLM tabanlı özet ve konu devamı tespiti
- Her N mesajda bir LLM'den özet al
- Yeni mesaj geldiğinde "Bu konu devamı mı?" sor
- Devamsa özeti güncelle, değilse arşivle

Kullanım:
    context_manager = ConversationContextManager(user_id="murat")

    # Her mesajda çağır
    result = context_manager.process_message(user_message, ai_response, chat_history)

    # Context al (LLM'e eklemek için)
    context = context_manager.get_current_context()
"""

import json
import os
import time
import requests
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class ConversationSession:
    """Aktif konuşma oturumu"""
    session_id: str
    topic_summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    message_count: int = 0
    started_at: str = ""
    last_updated: str = ""
    messages_buffer: List[Dict] = field(default_factory=list)


class ConversationContextManager:
    """
    LLM Tabanlı Konuşma Bağlamı Yöneticisi

    Özellikler:
    - LLM ile akıllı konu devamı tespiti
    - Dinamik özet oluşturma ve güncelleme
    - Otomatik arşivleme (FAISS veya dosya)
    - Her LLM çağrısına sessiz context enjeksiyonu
    """

    # Ayarlar
    SUMMARY_INTERVAL = 5  # Her 5 mesajda bir özet al
    MAX_BUFFER_SIZE = 10  # Buffer'da max mesaj sayısı

    def __init__(
        self,
        user_id: str = "default",
        base_dir: str = "user_data",
        together_api_key: str = None,
        together_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        archive_to_faiss: bool = False,
        faiss_manager = None
    ):
        self.user_id = user_id
        self.together_api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
        self.together_model = together_model
        self.archive_to_faiss = archive_to_faiss
        self.faiss_manager = faiss_manager

        # Dizin yapısı
        self.context_dir = os.path.join(base_dir, f"user_{user_id}", "conversation_context")
        self.archive_dir = os.path.join(self.context_dir, "archive")
        self.session_file = os.path.join(self.context_dir, "current_session.json")

        os.makedirs(self.archive_dir, exist_ok=True)

        # Aktif session'ı yükle veya oluştur
        self.current_session = self._load_or_create_session()

        # 🆕 Çift kontrol önleme flag'i
        # check_topic_before_response çağrıldığında True olur
        # process_message bu flag'i görürse topic check'i atlar
        self._topic_already_checked = False

        print(f"ConversationContextManager başlatıldı - user: {user_id}")

    # ==================== SESSION YÖNETİMİ ====================

    def _generate_session_id(self) -> str:
        """Benzersiz session ID oluştur"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_or_create_session(self) -> ConversationSession:
        """Mevcut session'ı yükle veya yeni oluştur"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ConversationSession(**data)
            except Exception as e:
                print(f"Session yükleme hatası: {e}")

        return self._create_new_session()

    def _create_new_session(self) -> ConversationSession:
        """Yeni session oluştur"""
        session = ConversationSession(
            session_id=self._generate_session_id(),
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        self._save_session(session)
        return session

    def _save_session(self, session: ConversationSession = None):
        """Session'ı kaydet"""
        session = session or self.current_session
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Session kaydetme hatası: {e}")

    def clear(self):
        """Tüm konuşma bağlamını temizle - sohbet sıfırlama için"""
        # Mevcut session'ı arşivle (varsa içerik)
        if self.current_session.topic_summary or self.current_session.message_count > 0:
            self._archive_session(self.current_session)

        # Yeni boş session oluştur
        self.current_session = self._create_new_session()
        self._topic_already_checked = False
        print("✅ ConversationContext temizlendi - yeni session başlatıldı")

    # ==================== LLM İLETİŞİMİ ====================

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Together.ai API çağrısı"""
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
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["text"].strip()
                # Gemma formatını temizle
                result = result.replace("<|eot_id|>", "").strip()
                return result

        except Exception as e:
            print(f"LLM çağrısı hatası: {e}")

        return ""

    # ==================== KONU DEVAMI TESPİTİ ====================

    def _check_topic_continuation(self, new_message: str, chat_history: List[Dict]) -> Tuple[bool, str]:
        """
        LLM ile konu devamı kontrolü

        Returns:
            (is_continuation, updated_summary)
        """
        # Mevcut özet ve son mesajları al
        current_summary = self.current_session.topic_summary

        # Son 3 mesajı al
        recent_messages = []
        for msg in chat_history[-6:]:
            role = "Kullanıcı" if msg.get("role") == "user" else "AI"
            content = (msg.get("content") or "")[:200]
            if content:
                recent_messages.append(f"{role}: {content}")

        recent_text = "\n".join(recent_messages[-4:])

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

GÖREV: Bu yeni mesajın mevcut konuşmanın devamı mı yoksa tamamen yeni bir konu mu olduğunu belirle.

MEVCUT KONU ÖZETİ:
{current_summary if current_summary else "(Henüz özet yok - ilk mesajlar)"}

SON MESAJLAR:
{recent_text}

YENİ MESAJ:
{new_message}

ANALİZ:
1. Yeni mesaj mevcut konuyla ilgili mi? (alt konu, derinleştirme, devam sorusu sayılır)
2. Tamamen alakasız yeni bir konu mu?

ÖNEMLİ:
- "Allah'ın ilmi" → "kader" → "irade" gibi İLİŞKİLİ konular DEVAM sayılır
- Bir konuyu derinleştirmek DEVAM sayılır
- "Peki ya X?" şeklinde bağlantılı sorular DEVAM sayılır
- Sadece tamamen alakasız konular YENİ KONU sayılır

CEVAP FORMAT:
KARAR: DEVAM veya YENİ_KONU
ÖZET: [Güncellenmiş konu özeti - 1-2 cümle]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        result = self._call_llm(prompt, max_tokens=150)

        # Parse et
        is_continuation = True
        updated_summary = current_summary

        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("KARAR:"):
                decision = line.split(":", 1)[1].strip().upper()
                is_continuation = "YENİ" not in decision
            elif line.upper().startswith("ÖZET:") or line.upper().startswith("OZET:"):
                updated_summary = line.split(":", 1)[1].strip()

        return is_continuation, updated_summary

    # ==================== ÖZET YÖNETİMİ ====================

    def _generate_summary(self, messages: List[Dict]) -> Tuple[str, List[str]]:
        """
        Mesajlardan özet ve anahtar konular çıkar

        Returns:
            (summary, key_topics)
        """
        # Mesajları text'e çevir
        conversation = []
        for m in messages[-8:]:
            role = "Kullanıcı" if m.get("role") == "user" else "AI"
            content = (m.get("content") or "")[:300]
            if content:
                conversation.append(f"{role}: {content}")

        if not conversation:
            return "", []

        conv_text = "\n".join(conversation)

        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Bu konuşmayı analiz et ve özet çıkar.

KONUŞMA:
{conv_text}

GÖREV:
1. Ana konu nedir? (1-2 cümle özet)
2. Hangi alt konular tartışıldı? (max 5 anahtar kelime)

FORMAT:
ÖZET: [ana konu özeti]
KONULAR: [konu1, konu2, konu3]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        result = self._call_llm(prompt, max_tokens=150)

        summary = ""
        topics = []

        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("ÖZET:") or line.upper().startswith("OZET:"):
                summary = line.split(":", 1)[1].strip()
            elif line.upper().startswith("KONULAR:"):
                topics_str = line.split(":", 1)[1].strip()
                topics = [t.strip() for t in topics_str.split(",") if t.strip()]

        return summary, topics[:5]

    # ==================== ARŞİVLEME ====================

    def _archive_session(self, session: ConversationSession):
        """Session'ı arşivle"""
        if not session.topic_summary:
            return

        archive_data = {
            "session_id": session.session_id,
            "summary": session.topic_summary,
            "topics": session.key_topics,
            "message_count": session.message_count,
            "started_at": session.started_at,
            "archived_at": datetime.now().isoformat()
        }

        # Dosyaya kaydet
        archive_file = os.path.join(
            self.archive_dir,
            f"{session.session_id}.json"
        )

        try:
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, ensure_ascii=False, indent=2)
            print(f"Session arşivlendi: {session.session_id}")
        except Exception as e:
            print(f"Arşivleme hatası: {e}")

        # FAISS'e de ekle (opsiyonel)
        if self.archive_to_faiss and self.faiss_manager:
            try:
                archive_text = f"Konu: {session.topic_summary}\nAnahtar konular: {', '.join(session.key_topics)}"
                self.faiss_manager.add(archive_text, metadata={
                    "type": "archived_session",
                    "session_id": session.session_id
                })
            except Exception as e:
                print(f"FAISS arşivleme hatası: {e}")

    def get_archived_context(self, query: str, max_results: int = 2) -> str:
        """Arşivden ilgili bağlamı getir"""
        if not self.archive_to_faiss or not self.faiss_manager:
            return ""

        try:
            results = self.faiss_manager.search(query, top_k=max_results)
            if results:
                contexts = [r.get("text", "")[:200] for r in results if r.get("type") == "archived_session"]
                if contexts:
                    return "Geçmiş konuşmalardan:\n" + "\n".join(contexts)
        except:
            pass

        return ""

    # ==================== ANA FONKSİYONLAR ====================

    def check_topic_before_response(self, user_message: str, chat_history: List[Dict]) -> bool:
        """
        🔑 MESAJ İŞLENMEDEN ÖNCE konu değişimini kontrol et.
        Bu metod, context almadan ÖNCE çağrılmalı.

        Args:
            user_message: Yeni kullanıcı mesajı
            chat_history: Mevcut sohbet geçmişi

        Returns:
            bool: True ise yeni session başlatıldı
        """
        # 🆕 Flag'i set et - process_message'de tekrar kontrol yapılmasın
        self._topic_already_checked = True

        # 🔑 KISA MESAJ - LLM karar verecek (keyword yerine)
        word_count = len(user_message.split())
        if word_count <= 4:
            print(f"   🔍 Kısa mesaj ({word_count} kelime) - LLM konu değişimi kontrol edecek")

        # Özet yoksa kontrol etmeye gerek yok
        if not self.current_session.topic_summary:
            return False

        # En az 2 mesaj varsa konu kontrolü yap
        if self.current_session.message_count < 2:
            return False

        try:
            is_continuation, updated_summary = self._check_topic_continuation(
                user_message, chat_history
            )

            if not is_continuation:
                # Yeni konu tespit edildi - eski session'ı arşivle
                print(f"🔄 Konu değişimi tespit edildi (pre-check)")
                self._archive_session(self.current_session)
                self.current_session = self._create_new_session()
                self._save_session()
                return True
            else:
                # Aynı konu devam ediyor, özeti güncelle
                if updated_summary and updated_summary != self.current_session.topic_summary:
                    self.current_session.topic_summary = updated_summary
                    self._save_session()

        except Exception as e:
            print(f"⚠️ Pre-check hatası: {e}")

        return False

    def process_message(
        self,
        user_message: str,
        ai_response: str,
        chat_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Her mesaj sonrası çağrılacak ana fonksiyon

        Args:
            user_message: Kullanıcı mesajı
            ai_response: AI yanıtı
            chat_history: Tüm sohbet geçmişi

        Returns:
            {
                "is_continuation": bool,
                "summary_updated": bool,
                "new_session_started": bool,
                "current_summary": str
            }
        """
        result = {
            "is_continuation": True,
            "summary_updated": False,
            "new_session_started": False,
            "current_summary": self.current_session.topic_summary
        }

        # Buffer'a ekle
        self.current_session.messages_buffer.append({
            "role": "user",
            "content": user_message
        })
        self.current_session.messages_buffer.append({
            "role": "assistant",
            "content": ai_response
        })
        self.current_session.message_count += 1
        self.current_session.last_updated = datetime.now().isoformat()

        # Buffer boyutu kontrolü
        if len(self.current_session.messages_buffer) > self.MAX_BUFFER_SIZE * 2:
            self.current_session.messages_buffer = self.current_session.messages_buffer[-self.MAX_BUFFER_SIZE * 2:]

        # Konu devamı kontrolü (en az 2 mesaj varsa)
        # 🆕 Eğer check_topic_before_response zaten kontrol ettiyse ATLA!
        if self._topic_already_checked:
            print("   ⏩ Konu kontrolü atlandı (check_topic_before_response zaten kontrol etti)")
            self._topic_already_checked = False  # Flag'i sıfırla
        elif self.current_session.message_count >= 2 and self.current_session.topic_summary:
            is_continuation, updated_summary = self._check_topic_continuation(
                user_message, chat_history
            )

            result["is_continuation"] = is_continuation

            if is_continuation:
                # Özeti güncelle
                if updated_summary and updated_summary != self.current_session.topic_summary:
                    self.current_session.topic_summary = updated_summary
                    result["summary_updated"] = True
                    result["current_summary"] = updated_summary
            else:
                # Yeni konu - eski session'ı arşivle
                self._archive_session(self.current_session)
                self.current_session = self._create_new_session()
                result["new_session_started"] = True
                result["current_summary"] = ""

        # Periyodik özet alma
        if self.current_session.message_count % self.SUMMARY_INTERVAL == 0:
            summary, topics = self._generate_summary(chat_history)
            if summary:
                self.current_session.topic_summary = summary
                self.current_session.key_topics = topics
                result["summary_updated"] = True
                result["current_summary"] = summary

        # İlk özet (3 mesaj sonra)
        if self.current_session.message_count == 3 and not self.current_session.topic_summary:
            summary, topics = self._generate_summary(chat_history)
            if summary:
                self.current_session.topic_summary = summary
                self.current_session.key_topics = topics
                result["summary_updated"] = True
                result["current_summary"] = summary

        # Kaydet
        self._save_session()

        return result

    def get_current_context(self) -> str:
        """
        Mevcut bağlamı getir (LLM'e eklemek için)

        Returns:
            Context string (boş olabilir)
        """
        if not self.current_session.topic_summary:
            return ""

        context_parts = []

        # Ana konu özeti
        context_parts.append(f"Mevcut konu: {self.current_session.topic_summary}")

        # Anahtar konular
        if self.current_session.key_topics:
            topics_str = ", ".join(self.current_session.key_topics)
            context_parts.append(f"Alt konular: {topics_str}")

        return "\n".join(context_parts)

    def get_context_for_prompt(self) -> str:
        """
        LLM prompt'una eklenecek formatlanmış bağlam
        """
        context = self.get_current_context()
        if not context:
            return ""

        return f"""[KONUŞMA BAĞLAMI]
{context}
[/KONUŞMA BAĞLAMI]

"""

    def force_new_session(self):
        """Manuel olarak yeni session başlat"""
        if self.current_session.topic_summary:
            self._archive_session(self.current_session)
        self.current_session = self._create_new_session()
        print("Yeni session başlatıldı")

    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri getir"""
        # Arşivdeki session sayısı
        archive_count = 0
        if os.path.exists(self.archive_dir):
            archive_count = len([f for f in os.listdir(self.archive_dir) if f.endswith('.json')])

        return {
            "current_session_id": self.current_session.session_id,
            "message_count": self.current_session.message_count,
            "has_summary": bool(self.current_session.topic_summary),
            "current_summary": self.current_session.topic_summary[:100] if self.current_session.topic_summary else "",
            "key_topics": self.current_session.key_topics,
            "archived_sessions": archive_count
        }


# ==================== ENTEGRASYON HELPER ====================

class ContextInjector:
    """
    Mevcut sisteme kolay entegrasyon için helper sınıf

    Kullanım:
        injector = ContextInjector(context_manager)

        # Prompt'a context ekle
        enhanced_prompt = injector.inject_context(original_prompt)

        # Mesaj işlendikten sonra
        injector.after_message(user_msg, ai_response, chat_history)
    """

    def __init__(self, context_manager: ConversationContextManager):
        self.context_manager = context_manager

    def inject_context(self, prompt: str) -> str:
        """Prompt'a context ekle"""
        context = self.context_manager.get_context_for_prompt()
        if context:
            return context + prompt
        return prompt

    def after_message(self, user_message: str, ai_response: str, chat_history: List[Dict]) -> Dict:
        """Mesaj sonrası işlem"""
        return self.context_manager.process_message(user_message, ai_response, chat_history)

    def get_context(self) -> str:
        """Mevcut context'i al"""
        return self.context_manager.get_current_context()


# ==================== TEST ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ConversationContextManager Test")
    print("=" * 60)

    # Test instance
    manager = ConversationContextManager(user_id="test_user")

    # Simüle mesajlar
    test_messages = [
        ("Allah'ın ilmi hakkında ne düşünüyorsun?", "Allah'ın ilmi sonsuz ve kuşatıcıdır..."),
        ("Peki bu kaderle nasıl ilişkili?", "Kader, Allah'ın ezeli ilmiyle doğrudan bağlantılıdır..."),
        ("İnsan iradesi bu durumda ne anlama geliyor?", "İnsan iradesi, kader içinde bir tercih alanıdır..."),
        ("Tamamen farklı bir konu: Python'da list nasıl kullanılır?", "Python'da list kullanımı..."),
    ]

    chat_history = []

    for user_msg, ai_resp in test_messages:
        print(f"\n--- Mesaj: {user_msg[:50]}... ---")

        # Chat history güncelle
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": ai_resp})

        # İşle
        result = manager.process_message(user_msg, ai_resp, chat_history)

        print(f"Devam mı: {result['is_continuation']}")
        print(f"Özet güncellendi: {result['summary_updated']}")
        print(f"Yeni session: {result['new_session_started']}")
        print(f"Özet: {result['current_summary'][:100]}...")

    print("\n--- İstatistikler ---")
    print(manager.get_stats())

    print("\n--- Context ---")
    print(manager.get_context_for_prompt())

    print("\n" + "=" * 60)
    print("Test tamamlandı!")
