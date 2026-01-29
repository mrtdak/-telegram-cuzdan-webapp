"""
Profile Manager - Kullanıcı Profil Yöneticisi
Her kullanıcı için kişisel profil bilgilerini saklar ve yönetir.

Kullanım:
    from profile_manager import ProfileManager

    pm = ProfileManager(user_id="murat")
    profile = pm.get_profile()
    pm.update_last_session("Bugün GUI geliştirdik")
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


class ProfileManager:
    """
    Kullanıcı profil yöneticisi
    - Profil oluşturma/okuma/güncelleme
    - Son sohbet özeti kaydetme
    - Prompt için profil özeti oluşturma
    """

    DEFAULT_PROFILE = {
        "name": "",
        "interests": [],
        "important_facts": [],
        "personality_notes": "",
        "last_session_summary": "",
        "last_session_date": "",  # Son sohbet tarihi (ISO format)
        "session_count": 0,
        "created_at": "",
        "updated_at": ""
    }

    def __init__(self, user_id: str, base_dir: str = "user_data"):
        self.user_id = user_id
        self.base_dir = base_dir
        self.user_dir = os.path.join(base_dir, f"user_{user_id}")
        self.profile_path = os.path.join(self.user_dir, "profile.json")

        # Dizin yoksa oluştur
        os.makedirs(self.user_dir, exist_ok=True)

        # Profili yükle veya oluştur
        self.profile = self._load_or_create_profile()

    def _load_or_create_profile(self) -> Dict[str, Any]:
        """Profili yükle, yoksa varsayılan oluştur"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    # Eksik alanları ekle
                    for key, value in self.DEFAULT_PROFILE.items():
                        if key not in profile:
                            profile[key] = value
                    return profile
            except Exception as e:
                print(f"[ProfileManager] Profil okuma hatası: {e}")

        # Yeni profil oluştur
        profile = self.DEFAULT_PROFILE.copy()
        profile["created_at"] = datetime.now().isoformat()
        profile["updated_at"] = datetime.now().isoformat()
        self._save_profile(profile)
        return profile

    def _save_profile(self, profile: Dict[str, Any] = None) -> bool:
        """Profili kaydet"""
        if profile is None:
            profile = self.profile

        try:
            profile["updated_at"] = datetime.now().isoformat()
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[ProfileManager] Profil kaydetme hatası: {e}")
            return False

    def get_profile(self) -> Dict[str, Any]:
        """Profili döndür"""
        return self.profile

    def get_name(self) -> str:
        """Kullanıcı adını döndür"""
        return self.profile.get("name", "")

    def set_name(self, name: str) -> bool:
        """Kullanıcı adını ayarla"""
        self.profile["name"] = name
        return self._save_profile()

    def get_interests(self) -> List[str]:
        """İlgi alanlarını döndür"""
        return self.profile.get("interests", [])

    def add_interest(self, interest: str) -> bool:
        """İlgi alanı ekle"""
        interests = self.profile.get("interests", [])
        if interest not in interests:
            interests.append(interest)
            self.profile["interests"] = interests
            return self._save_profile()
        return True

    def get_important_facts(self) -> List[str]:
        """Önemli gerçekleri döndür"""
        return self.profile.get("important_facts", [])

    def add_important_fact(self, fact: str) -> bool:
        """Önemli gerçek ekle"""
        facts = self.profile.get("important_facts", [])
        if fact not in facts:
            facts.append(fact)
            self.profile["important_facts"] = facts
            return self._save_profile()
        return True

    def get_last_session_summary(self) -> str:
        """Son sohbet özetini döndür"""
        return self.profile.get("last_session_summary", "")

    def update_last_session(self, summary: str) -> bool:
        """Son sohbet özetini ve tarihini güncelle"""
        self.profile["last_session_summary"] = summary
        self.profile["last_session_date"] = datetime.now().isoformat()
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1
        return self._save_profile()

    def get_prompt_context(self) -> str:
        """
        Prompt'a eklenecek profil bağlamını oluştur
        Bu bilgi AI'a sessizce verilir, kullanıcıya söylemez
        """
        parts = []

        name = self.get_name()
        if name:
            parts.append(f"  Kullanıcının adı: {name}")

        if not parts:
            return ""

        return "\n".join(parts)

    def has_profile(self) -> bool:
        """Profil dolu mu kontrol et"""
        return bool(self.get_name() or self.get_interests() or self.get_important_facts())

    def increment_session(self) -> int:
        """Oturum sayısını artır ve döndür"""
        self.profile["session_count"] = self.profile.get("session_count", 0) + 1
        self._save_profile()
        return self.profile["session_count"]

    def update_profile(self, updates: Dict[str, Any]) -> bool:
        """Profili toplu güncelle"""
        for key, value in updates.items():
            if key in self.DEFAULT_PROFILE:
                self.profile[key] = value
        return self._save_profile()

    def __repr__(self):
        return f"ProfileManager(user_id={self.user_id}, name={self.get_name()})"


# Test
if __name__ == "__main__":
    pm = ProfileManager(user_id="test_profile")

    print("Profil oluşturuldu:", pm.profile_path)
    print("Mevcut profil:", pm.get_profile())

    # Test güncellemeleri
    pm.set_name("Test Kullanıcı")
    pm.add_interest("yazılım")
    pm.add_interest("AI")
    pm.add_important_fact("Python biliyor")
    pm.update_last_session("Test sohbeti yapıldı")

    print("\nGüncellenmiş profil:", pm.get_profile())
    print("\nPrompt context:\n", pm.get_prompt_context())
