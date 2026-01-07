"""
PersonalAI - Sadeleştirilmiş LLM Cevap Üretici
Telegram → HafizaAsistani (prompt) → PersonalAI (LLM) → Telegram
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional
import torch
import aiohttp
from zoneinfo import ZoneInfo

from hafiza_asistani import HafizaAsistani


class SystemConfig:
    """Sistem ayarları"""

    SYSTEM_NAME = "PersonalAI"
    VERSION = "3.0.0"
    DEFAULT_USER_ID = "murat"
    USER_DATA_BASE_DIR = "user_data"

    LOG_FULL_PROMPT = True  # Debug için

    # LLM Ayarları
    LLM_PROVIDER = "together"  # "ollama" veya "together"

    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gemma3:27b"

    TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

    MODEL_NAME = TOGETHER_MODEL if LLM_PROVIDER == "together" else OLLAMA_MODEL

    # Model Parametreleri
    TEMPERATURE = 0.5
    TOP_P = 0.90
    REPEAT_PENALTY = 1.15
    MAX_TOKENS = 1500

    ENABLE_VISION = True

    TIMEZONE = ZoneInfo("Europe/Istanbul")

    @classmethod
    def get_gemma3_params(cls) -> Dict[str, Any]:
        return {
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "repeat_penalty": cls.REPEAT_PENALTY,
            "max_tokens": cls.MAX_TOKENS,
        }


class LocalLLM:
    """
    LLM API wrapper - Together.ai veya Ollama
    """

    def __init__(self, user_id: str = SystemConfig.DEFAULT_USER_ID):
        self.user_id = user_id
        self.provider = SystemConfig.LLM_PROVIDER
        self.ollama_url = SystemConfig.OLLAMA_URL
        self.model_name = SystemConfig.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.together_api_key = os.getenv("TOGETHER_API_KEY", "")

        provider_name = "Together.ai" if self.provider == "together" else "Ollama"
        print(f"✅ LLM başlatıldı: {self.model_name} ({provider_name}, {self.device})")

    async def generate(self, prompt: str, image_data: Optional[bytes] = None) -> str:
        """LLM yanıt üret"""
        try:
            if image_data:
                return await self._generate_with_vision(prompt, image_data)
            else:
                return await self._generate_text_only(prompt)
        except Exception as e:
            print(f"❌ LLM hatası: {e}")
            return "Üzgünüm, yanıt oluşturulurken bir hata oluştu."

    async def _generate_with_vision(self, prompt: str, image_data: str) -> str:
        """Vision ile yanıt üret (Ollama)"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "raw": True,
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
                        return "Görseli analiz edemedim."
        except Exception as e:
            print(f"⚠️ Vision hatası: {e}")
            return "Görsel analizi sırasında hata oluştu."

    async def _generate_text_only(self, prompt: str) -> str:
        """Text LLM çağrısı"""
        if SystemConfig.LOG_FULL_PROMPT:
            print("\n" + "=" * 70)
            print(f"📋 LLM PROMPT ({self.provider.upper()}):")
            print("=" * 70)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("=" * 70 + "\n")

        if self.provider == "together":
            return await self._generate_together(prompt)
        else:
            return await self._generate_ollama(prompt)

    async def _generate_together(self, prompt: str) -> str:
        """Together.ai API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": SystemConfig.TOGETHER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
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
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        error_text = await resp.text()
                        print(f"⚠️ Together.ai hatası: {resp.status} - {error_text[:200]}")
                        return "API hatası oluştu."
        except asyncio.TimeoutError:
            return "Zaman aşımı."
        except Exception as e:
            print(f"⚠️ Together.ai hatası: {e}")
            return "Bağlantı hatası."

    async def _generate_ollama(self, prompt: str) -> str:
        """Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,
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
                        return "Ollama hatası."
        except Exception as e:
            print(f"⚠️ Ollama hatası: {e}")
            return "Bağlantı hatası."


class PersonalAI:
    """
    PersonalAI - LLM Cevap Üretici

    Akış:
    Telegram → HafizaAsistani (prompt) → PersonalAI (LLM) → HafizaAsistani (kayıt) → Telegram
    """

    def __init__(self, user_id: str = None):
        self.user_id = user_id or SystemConfig.DEFAULT_USER_ID

        print("=" * 60)
        print(f"🚀 PersonalAI Başlatılıyor... (user: {self.user_id})")
        print("=" * 60)

        # 1. LLM - Cevap üretici
        self.llm = LocalLLM(self.user_id)

        # 2. HafizaAsistani - Merkezi beyin
        self.memory = HafizaAsistani(
            saat_limiti=48,
            esik=0.50,
            max_mesaj=20,
            model_adi="BAAI/bge-m3",
            use_decision_llm=True,
            decision_model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        )

        # 3. LLM'i HafizaAsistani'ya ver
        self.memory.set_llm(self.llm)

        print("✅ PersonalAI hazır!")
        print("=" * 60 + "\n")

    async def generate(self, prompt: str, image_data=None) -> str:
        """LLM cevap üret"""
        return await self.llm.generate(prompt, image_data)

    def close(self):
        """Kapat"""
        print("🛑 PersonalAI kapatılıyor...")
        if hasattr(self.memory, 'profile_manager'):
            try:
                self.memory.profile_manager.update_last_session("Sohbet yapıldı")
            except:
                pass
        print("✅ Tamamlandı.")
