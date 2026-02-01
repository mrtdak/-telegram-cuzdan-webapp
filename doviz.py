"""
Döviz ve Kripto Kur Modülü
- Döviz: Frankfurter API (ücretsiz, API key gerektirmez)
- Kripto: CoinGecko API (ücretsiz, API key gerektirmez)
- Altın: Frankfurter XAU/TRY
"""

import requests
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

class DovizKur:
    """Döviz ve kripto kur sorgulama sınıfı"""

    def __init__(self):
        # Frankfurter API - Avrupa Merkez Bankası verileri
        self.doviz_api = "https://api.frankfurter.app"
        # CoinGecko API - Kripto
        self.kripto_api = "https://api.coingecko.com/api/v3"
        self.timeout = 10

        # Cache - 5 dakika geçerli
        self._cache = {}
        self._cache_suresi = timedelta(minutes=5)

        # Desteklenen para birimleri
        self.doviz_map = {
            "dolar": "USD",
            "euro": "EUR",
            "sterlin": "GBP",
            "pound": "GBP",
            "frank": "CHF",
            "isviçre frangı": "CHF",
            "yen": "JPY",
            "japon yeni": "JPY",
            "yuan": "CNY",
            "çin yuanı": "CNY",
            "ruble": "RUB",
            "rus rublesi": "RUB",
            "riyal": "SAR",
            "suudi riyali": "SAR",
            "dirhem": "AED",
            "kanada doları": "CAD",
            "avustralya doları": "AUD",
        }

        # Kripto haritası
        self.kripto_map = {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "ethereum": "ethereum",
            "eth": "ethereum",
            "ether": "ethereum",
            "solana": "solana",
            "sol": "solana",
            "ripple": "ripple",
            "xrp": "ripple",
            "dogecoin": "dogecoin",
            "doge": "dogecoin",
            "cardano": "cardano",
            "ada": "cardano",
            "bnb": "binancecoin",
            "binance": "binancecoin",
            "tether": "tether",
            "usdt": "tether",
            "avax": "avalanche-2",
            "avalanche": "avalanche-2",
            "polkadot": "polkadot",
            "dot": "polkadot",
            "matic": "matic-network",
            "polygon": "matic-network",
            "shiba": "shiba-inu",
            "shib": "shiba-inu",
            "litecoin": "litecoin",
            "ltc": "litecoin",
        }

        # Emojiler
        self.emoji_map = {
            "USD": "💵",
            "EUR": "💶",
            "GBP": "💷",
            "JPY": "💴",
            "bitcoin": "₿",
            "ethereum": "⟠",
            "altin": "🥇",
            "gumus": "🥈",
        }

    def _cache_kontrol(self, anahtar: str) -> Optional[Dict]:
        """Cache kontrolü"""
        if anahtar in self._cache:
            veri, zaman = self._cache[anahtar]
            if datetime.now() - zaman < self._cache_suresi:
                return veri
        return None

    def _cache_kaydet(self, anahtar: str, veri: Dict):
        """Cache'e kaydet"""
        self._cache[anahtar] = (veri, datetime.now())

    def doviz_getir(self, birim: str = "USD") -> Dict:
        """
        Döviz kuru getir (TL karşılığı)

        Args:
            birim: Para birimi kodu (USD, EUR, GBP vs.)

        Returns:
            {"birim": "USD", "kur": 38.45, "emoji": "💵"}
        """
        birim = birim.upper()
        cache_key = f"doviz_{birim}"

        # Cache kontrol
        cached = self._cache_kontrol(cache_key)
        if cached:
            return cached

        try:
            # Frankfurter API'den TRY karşılığı al
            response = requests.get(
                f"{self.doviz_api}/latest",
                params={"from": birim, "to": "TRY"},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            kur = data.get("rates", {}).get("TRY", 0)

            sonuc = {
                "birim": birim,
                "kur": round(kur, 4),
                "emoji": self.emoji_map.get(birim, "💱"),
                "tarih": data.get("date", ""),
                "basarili": True
            }

            self._cache_kaydet(cache_key, sonuc)
            return sonuc

        except Exception as e:
            return {
                "birim": birim,
                "kur": 0,
                "emoji": "❌",
                "hata": str(e),
                "basarili": False
            }

    def kripto_getir(self, kripto: str = "bitcoin") -> Dict:
        """
        Kripto para fiyatı getir (USD ve TRY)

        Args:
            kripto: Kripto adı (bitcoin, ethereum vs.)

        Returns:
            {"kripto": "bitcoin", "usd": 97450, "try": 3750000, "degisim_24h": -2.5}
        """
        kripto = kripto.lower()
        kripto_id = self.kripto_map.get(kripto, kripto)
        cache_key = f"kripto_{kripto_id}"

        # Cache kontrol
        cached = self._cache_kontrol(cache_key)
        if cached:
            return cached

        try:
            response = requests.get(
                f"{self.kripto_api}/simple/price",
                params={
                    "ids": kripto_id,
                    "vs_currencies": "usd,try",
                    "include_24hr_change": "true"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            kripto_data = data.get(kripto_id, {})

            sonuc = {
                "kripto": kripto_id,
                "usd": kripto_data.get("usd", 0),
                "try": kripto_data.get("try", 0),
                "degisim_24h": round(kripto_data.get("usd_24h_change", 0), 2),
                "emoji": self.emoji_map.get(kripto_id, "🪙"),
                "basarili": True
            }

            self._cache_kaydet(cache_key, sonuc)
            return sonuc

        except Exception as e:
            return {
                "kripto": kripto_id,
                "usd": 0,
                "try": 0,
                "degisim_24h": 0,
                "emoji": "❌",
                "hata": str(e),
                "basarili": False
            }

    def altin_getir(self) -> Dict:
        """
        Altın fiyatı getir (gram, TL)
        CoinGecko'dan altın fiyatı (XAU)
        """
        cache_key = "altin"

        cached = self._cache_kontrol(cache_key)
        if cached:
            return cached

        try:
            # CoinGecko'dan altın fiyatı
            response = requests.get(
                f"{self.kripto_api}/simple/price",
                params={
                    "ids": "tether-gold",  # XAUT - altın destekli token
                    "vs_currencies": "try"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Ons fiyatı / 31.1035 = gram fiyatı
            ons_try = data.get("tether-gold", {}).get("try", 0)
            gram_try = ons_try  # XAUT zaten yaklaşık 1 ons

            if gram_try == 0:
                # Fallback: USD altın fiyatından hesapla
                response2 = requests.get(
                    f"{self.kripto_api}/simple/price",
                    params={"ids": "tether-gold", "vs_currencies": "usd"},
                    timeout=self.timeout
                )
                usd_data = response2.json()
                altin_usd = usd_data.get("tether-gold", {}).get("usd", 2700)

                # Dolar kurunu al
                dolar = self.doviz_getir("USD")
                dolar_kur = dolar.get("kur", 38)

                # 1 ons = 31.1035 gram
                gram_try = (altin_usd * dolar_kur) / 31.1035

            sonuc = {
                "tip": "gram_altin",
                "fiyat": round(gram_try, 2),
                "emoji": "🥇",
                "basarili": True
            }

            self._cache_kaydet(cache_key, sonuc)
            return sonuc

        except Exception as e:
            return {
                "tip": "gram_altin",
                "fiyat": 0,
                "emoji": "❌",
                "hata": str(e),
                "basarili": False
            }

    def tum_kurlar(self) -> str:
        """Tüm önemli kurları listele"""
        lines = ["📊 **GÜNCEL KURLAR**\n"]

        # Dövizler
        for isim, kod in [("Dolar", "USD"), ("Euro", "EUR"), ("Sterlin", "GBP")]:
            kur = self.doviz_getir(kod)
            if kur["basarili"]:
                lines.append(f"{kur['emoji']} {isim}: {kur['kur']:,.2f} ₺")
            else:
                lines.append(f"❌ {isim}: Alınamadı")

        lines.append("")

        # Altın
        altin = self.altin_getir()
        if altin["basarili"]:
            lines.append(f"🥇 Gram Altın: {altin['fiyat']:,.2f} ₺")

        lines.append("")

        # Kripto
        for isim, kod in [("Bitcoin", "bitcoin"), ("Ethereum", "ethereum")]:
            kripto = self.kripto_getir(kod)
            if kripto["basarili"]:
                degisim = kripto["degisim_24h"]
                trend = "📈" if degisim > 0 else "📉" if degisim < 0 else "➡️"
                lines.append(f"{kripto['emoji']} {isim}: ${kripto['usd']:,.0f} ({trend} %{degisim:+.1f})")

        return "\n".join(lines)

    def kur_sorgula(self, mesaj: str) -> Optional[str]:
        """
        Mesajdan kur sorgusu çıkar ve cevapla.
        DecisionLLM tarafından kullanılacak.

        Args:
            mesaj: Kullanıcı mesajı (örn: "dolar kaç", "bitcoin fiyatı")

        Returns:
            Kur bilgisi string veya None
        """
        mesaj_lower = mesaj.lower().strip()

        # Döviz kontrolü
        for isim, kod in self.doviz_map.items():
            if isim in mesaj_lower:
                kur = self.doviz_getir(kod)
                if kur["basarili"]:
                    return f"{kur['emoji']} {isim.title()}: {kur['kur']:,.4f} ₺"
                else:
                    return f"❌ {isim.title()} kuru alınamadı: {kur.get('hata', 'Bilinmeyen hata')}"

        # Kripto kontrolü
        for isim, kod in self.kripto_map.items():
            if isim in mesaj_lower:
                kripto = self.kripto_getir(kod)
                if kripto["basarili"]:
                    degisim = kripto["degisim_24h"]
                    trend = "📈" if degisim > 0 else "📉" if degisim < 0 else "➡️"
                    return (
                        f"{kripto['emoji']} {isim.title()}\n"
                        f"💵 ${kripto['usd']:,.2f}\n"
                        f"💰 {kripto['try']:,.0f} ₺\n"
                        f"{trend} 24s: %{degisim:+.1f}"
                    )
                else:
                    return f"❌ {isim.title()} fiyatı alınamadı"

        # Altın kontrolü
        if any(k in mesaj_lower for k in ["altın", "altin", "gram altın", "gram altin"]):
            altin = self.altin_getir()
            if altin["basarili"]:
                return f"🥇 Gram Altın: {altin['fiyat']:,.2f} ₺"
            else:
                return f"❌ Altın fiyatı alınamadı"

        # Tüm kurlar
        if any(k in mesaj_lower for k in ["tüm kurlar", "kurlar", "döviz kurları", "doviz kurlari"]):
            return self.tum_kurlar()

        return None


# Test
if __name__ == "__main__":
    doviz = DovizKur()

    print("=== DOVİZ TESTİ ===")
    print(doviz.doviz_getir("USD"))
    print(doviz.doviz_getir("EUR"))

    print("\n=== KRİPTO TESTİ ===")
    print(doviz.kripto_getir("bitcoin"))
    print(doviz.kripto_getir("ethereum"))

    print("\n=== ALTIN TESTİ ===")
    print(doviz.altin_getir())

    print("\n=== TÜM KURLAR ===")
    print(doviz.tum_kurlar())

    print("\n=== SORGU TESTİ ===")
    print(doviz.kur_sorgula("dolar kaç"))
    print(doviz.kur_sorgula("bitcoin fiyatı"))
