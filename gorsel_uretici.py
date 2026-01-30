"""
Replicate API ile FLUX görsel üretim modülü.
Sohbet akışına entegre çalışır.
"""

import os
import asyncio
import aiohttp
from typing import Optional


class GorselUretici:
    """Replicate FLUX API ile görsel üretim"""

    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        self.api_url = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        # flux-schnell: $0.003/görsel, hızlı
        # flux-dev: $0.03/görsel, daha kaliteli
        # flux-pro: $0.055/görsel, en kaliteli

    async def uret(self, prompt: str) -> Optional[bytes]:
        """
        Prompt'tan görsel üret (FLUX modeli).

        Args:
            prompt: Görsel için açıklama (Türkçe veya İngilizce)

        Returns:
            bytes: Görsel verisi (PNG/JPEG)
            None: Hata durumunda
        """
        if not self.api_token:
            print("❌ REPLICATE_API_TOKEN bulunamadı")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait"  # Sonucu bekle (sync mode)
        }

        payload = {
            "input": {
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Prediction başlat
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status in [200, 201]:
                        data = await resp.json()

                        # Sonuç URL'sini al
                        output = data.get("output")
                        if output:
                            # output liste olabilir
                            image_url = output[0] if isinstance(output, list) else output
                            print(f"✅ Görsel üretildi: {prompt[:50]}...")

                            # Görseli indir
                            async with session.get(image_url) as img_resp:
                                if img_resp.status == 200:
                                    return await img_resp.read()
                                else:
                                    print(f"❌ Görsel indirilemedi: {img_resp.status}")
                        else:
                            # Henüz hazır değil, polling yap
                            prediction_id = data.get("id")
                            if prediction_id:
                                return await self._poll_result(session, prediction_id, headers)
                            print("❌ Görsel çıktısı bulunamadı")
                    else:
                        error = await resp.text()
                        print(f"❌ Replicate hatası: {resp.status} - {error[:200]}")

        except asyncio.TimeoutError:
            print("❌ Replicate timeout (120s)")
        except Exception as e:
            print(f"❌ Replicate bağlantı hatası: {e}")

        return None

    async def _poll_result(self, session, prediction_id: str, headers: dict) -> Optional[bytes]:
        """Sonuç hazır olana kadar bekle"""
        poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"

        for _ in range(60):  # Max 60 deneme (60 saniye)
            await asyncio.sleep(1)

            async with session.get(poll_url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status = data.get("status")

                    if status == "succeeded":
                        output = data.get("output")
                        if output:
                            image_url = output[0] if isinstance(output, list) else output
                            async with session.get(image_url) as img_resp:
                                if img_resp.status == 200:
                                    return await img_resp.read()
                        return None

                    elif status == "failed":
                        error = data.get("error", "Bilinmeyen hata")
                        print(f"❌ Görsel üretim başarısız: {error}")
                        return None

                    # processing/starting - devam et

        print("❌ Polling timeout")
        return None


# Test için
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    async def test():
        uretici = GorselUretici()
        result = await uretici.uret("A cute orange cat sitting on a windowsill, digital art")
        if result:
            with open("test_gorsel.webp", "wb") as f:
                f.write(result)
            print("✅ test_gorsel.webp kaydedildi")
        else:
            print("❌ Test başarısız")

    asyncio.run(test())
