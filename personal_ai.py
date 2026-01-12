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

# HafizaAsistani artık telegram_bot.py'de yönetiliyor


class SystemConfig:
    """Sistem ayarları"""

    SYSTEM_NAME = "PersonalAI"
    VERSION = "3.0.0"
    DEFAULT_USER_ID = "murat"
    USER_DATA_BASE_DIR = "user_data"

    LOG_FULL_PROMPT = True  # Debug için

    # LLM Ayarları
    LLM_PROVIDER = "openrouter"  # "ollama", "together" veya "openrouter"

    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gemma3:27b"

    TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

    # OpenRouter (Claude)
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "google/gemma-3-27b-it"  # Test: Gemma 3 27B (açık kaynak)

    MODEL_NAME = OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else (TOGETHER_MODEL if LLM_PROVIDER == "together" else OLLAMA_MODEL)

    # Model Parametreleri (Gemma 3 - tutarlı ayar)
    TEMPERATURE = 0.72  # Dengeli: doğal ve akıcı
    TOP_P = 0.95       # Gemma resmi
    TOP_K = 64         # Gemma resmi
    MAX_TOKENS = 4000

    ENABLE_VISION = True

    TIMEZONE = ZoneInfo("Europe/Istanbul")

    @classmethod
    def get_gemma3_params(cls) -> Dict[str, Any]:
        return {
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
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
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

        provider_names = {"together": "Together.ai", "ollama": "Ollama", "openrouter": "OpenRouter (Claude)"}
        provider_name = provider_names.get(self.provider, self.provider)
        print(f"✅ LLM başlatıldı: {self.model_name} ({provider_name}, {self.device})")

    async def generate(self, prompt: str, image_data: Optional[bytes] = None, messages: list = None) -> str:
        """LLM yanıt üret - messages formatı destekler"""
        try:
            if image_data:
                return await self._generate_with_vision(prompt, image_data)
            elif messages:
                # Yeni: Messages formatı (sohbet bağlamı korunur)
                return await self._generate_with_messages(messages)
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

        if self.provider == "openrouter":
            return await self._generate_openrouter(prompt)
        elif self.provider == "together":
            return await self._generate_together(prompt)
        else:
            return await self._generate_ollama(prompt)

    async def _generate_with_messages(self, messages: list) -> str:
        """Messages formatı ile LLM çağrısı - sohbet bağlamı korunur"""
        if SystemConfig.LOG_FULL_PROMPT:
            print("\n" + "=" * 70)
            print(f"📋 LLM MESSAGES ({self.provider.upper()}):")
            print("=" * 70)
            for msg in messages[-5:]:  # Son 5 mesajı göster
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                # System message tam göster, diğerleri kısa
                if role == 'system':
                    print(f"[{role}]: {content}")
                else:
                    print(f"[{role}]: {content}")  # Tam göster, kesme yok
            print("=" * 70 + "\n")

        if self.provider == "openrouter":
            return await self._generate_openrouter_messages(messages)
        elif self.provider == "together":
            return await self._generate_together_messages(messages)
        else:
            # Ollama için messages'ı prompt'a çevir
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return await self._generate_ollama(prompt)

    async def _generate_together_messages(self, messages: list) -> str:
        """Together.ai API - Messages formatı"""
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": SystemConfig.TOGETHER_MODEL,
                "messages": messages,
                "max_tokens": SystemConfig.MAX_TOKENS,
                "temperature": SystemConfig.TEMPERATURE,
                "top_p": SystemConfig.TOP_P,
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

    async def _generate_openrouter(self, prompt: str) -> str:
        """OpenRouter API (Claude)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/personal-ai",
                "X-Title": "PersonalAI"
            }
            payload = {
                "model": SystemConfig.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": SystemConfig.MAX_TOKENS,
                "temperature": SystemConfig.TEMPERATURE,
                "top_p": SystemConfig.TOP_P,
                "top_k": SystemConfig.TOP_K
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SystemConfig.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        error_text = await resp.text()
                        print(f"⚠️ OpenRouter hatası: {resp.status} - {error_text[:200]}")
                        return "API hatası oluştu."
        except asyncio.TimeoutError:
            return "Zaman aşımı."
        except Exception as e:
            print(f"⚠️ OpenRouter hatası: {e}")
            return "Bağlantı hatası."

    async def _generate_openrouter_messages(self, messages: list) -> str:
        """OpenRouter API - Messages formatı (Claude)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/personal-ai",
                "X-Title": "PersonalAI"
            }
            payload = {
                "model": SystemConfig.OPENROUTER_MODEL,
                "messages": messages,
                "max_tokens": SystemConfig.MAX_TOKENS,
                "temperature": SystemConfig.TEMPERATURE,
                "top_p": SystemConfig.TOP_P,
                "top_k": SystemConfig.TOP_K
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SystemConfig.OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        error_text = await resp.text()
                        print(f"⚠️ OpenRouter hatası: {resp.status} - {error_text[:200]}")
                        return "API hatası oluştu."
        except asyncio.TimeoutError:
            return "Zaman aşımı."
        except Exception as e:
            print(f"⚠️ OpenRouter hatası: {e}")
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
    PersonalAI - SADECE Cevap Üretici (LLM)

    Akış:
    Telegram → HafizaAsistani.prepare() → PersonalAI.generate() → HafizaAsistani.save() → Telegram
    """

    def __init__(self, user_id: str = None):
        self.user_id = user_id or SystemConfig.DEFAULT_USER_ID

        print(f"🤖 PersonalAI başlatıldı (user: {self.user_id})")

        # Sadece LLM - Cevap üretici
        self.llm = LocalLLM(self.user_id)

    async def generate(self, messages: list = None, prompt: str = None, image_data: bytes = None) -> str:
        """
        LLM ile cevap üret

        Args:
            messages: Chat messages formatı (öncelikli)
            prompt: Düz metin prompt (eski format)
            image_data: Görsel verisi (vision için)

        Returns:
            str: LLM cevabı
        """
        return await self.llm.generate(prompt=prompt, image_data=image_data, messages=messages)
