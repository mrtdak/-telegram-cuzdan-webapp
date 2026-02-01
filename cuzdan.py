"""
Kişisel Cüzdan/Banka Modülü
Telegram butonlarıyla gelir/gider takibi
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class Cuzdan:
    """Kullanıcı cüzdan yönetimi"""

    # Gelir kategorileri
    GELIR_KATEGORILERI = {
        "maas": ("💼", "Maaş"),
        "ek_gelir": ("💵", "Ek Gelir"),
        "yatirim": ("📈", "Yatırım"),
        "hediye": ("🎁", "Hediye"),
        "iade": ("🔄", "İade"),
        "diger_gelir": ("💰", "Diğer"),
    }

    # Gider kategorileri
    GIDER_KATEGORILERI = {
        "market": ("🛒", "Market"),
        "yemek": ("🍔", "Yemek"),
        "fatura": ("📄", "Fatura"),
        "yakit": ("⛽", "Yakıt"),
        "ulasim": ("🚌", "Ulaşım"),
        "saglik": ("💊", "Sağlık"),
        "giyim": ("👕", "Giyim"),
        "eglence": ("🎮", "Eğlence"),
        "egitim": ("📚", "Eğitim"),
        "kira": ("🏠", "Kira"),
        "aidat": ("🏢", "Aidat"),
        "diger_gider": ("💸", "Diğer"),
    }

    def __init__(self, user_id: str, base_dir: str = "user_data"):
        self.user_id = user_id
        self.cuzdan_dir = os.path.join(base_dir, f"user_{user_id}", "cuzdan")
        self.cuzdan_file = os.path.join(self.cuzdan_dir, "islemler.json")

        # Klasör oluştur
        os.makedirs(self.cuzdan_dir, exist_ok=True)

        # Verileri yükle
        self.veriler = self._yukle()

    def _yukle(self) -> Dict:
        """Cüzdan verilerini yükle"""
        if os.path.exists(self.cuzdan_file):
            try:
                with open(self.cuzdan_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # Varsayılan yapı
        return {
            "islemler": [],
            "baslangic_bakiye": 0,
            "olusturma": datetime.now().isoformat()
        }

    def _kaydet(self):
        """Verileri kaydet"""
        with open(self.cuzdan_file, 'w', encoding='utf-8') as f:
            json.dump(self.veriler, f, ensure_ascii=False, indent=2)

    def gelir_ekle(self, tutar: float, kategori: str, aciklama: str = "") -> Dict:
        """Gelir ekle"""
        islem = {
            "id": len(self.veriler["islemler"]) + 1,
            "tip": "gelir",
            "tutar": tutar,
            "kategori": kategori,
            "aciklama": aciklama,
            "tarih": datetime.now().isoformat(),
            "gun": datetime.now().strftime("%d.%m.%Y"),
            "saat": datetime.now().strftime("%H:%M")
        }
        self.veriler["islemler"].append(islem)
        self._kaydet()
        return islem

    def gider_ekle(self, tutar: float, kategori: str, aciklama: str = "") -> Dict:
        """Gider ekle"""
        islem = {
            "id": len(self.veriler["islemler"]) + 1,
            "tip": "gider",
            "tutar": tutar,
            "kategori": kategori,
            "aciklama": aciklama,
            "tarih": datetime.now().isoformat(),
            "gun": datetime.now().strftime("%d.%m.%Y"),
            "saat": datetime.now().strftime("%H:%M")
        }
        self.veriler["islemler"].append(islem)
        self._kaydet()
        return islem

    def islem_sil(self, islem_id: int) -> bool:
        """İşlem sil"""
        for i, islem in enumerate(self.veriler["islemler"]):
            if islem["id"] == islem_id:
                del self.veriler["islemler"][i]
                self._kaydet()
                return True
        return False

    def bakiye_hesapla(self) -> float:
        """Toplam bakiyeyi hesapla"""
        bakiye = self.veriler.get("baslangic_bakiye", 0)
        for islem in self.veriler["islemler"]:
            if islem["tip"] == "gelir":
                bakiye += islem["tutar"]
            else:
                bakiye -= islem["tutar"]
        return bakiye

    def baslangic_bakiye_ayarla(self, tutar: float):
        """Başlangıç bakiyesi ayarla"""
        self.veriler["baslangic_bakiye"] = tutar
        self._kaydet()

    def son_islemler(self, limit: int = 10) -> List[Dict]:
        """Son işlemleri getir"""
        return list(reversed(self.veriler["islemler"]))[:limit]

    def bu_ay_ozet(self) -> Dict:
        """Bu ayın özeti"""
        now = datetime.now()
        ay_basi = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        toplam_gelir = 0
        toplam_gider = 0
        gelir_kategorileri = defaultdict(float)
        gider_kategorileri = defaultdict(float)

        for islem in self.veriler["islemler"]:
            islem_tarih = datetime.fromisoformat(islem["tarih"])
            if islem_tarih >= ay_basi:
                if islem["tip"] == "gelir":
                    toplam_gelir += islem["tutar"]
                    gelir_kategorileri[islem["kategori"]] += islem["tutar"]
                else:
                    toplam_gider += islem["tutar"]
                    gider_kategorileri[islem["kategori"]] += islem["tutar"]

        return {
            "ay": now.strftime("%B %Y"),
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net": toplam_gelir - toplam_gider,
            "gelir_kategorileri": dict(gelir_kategorileri),
            "gider_kategorileri": dict(gider_kategorileri)
        }

    def bu_hafta_ozet(self) -> Dict:
        """Bu haftanın özeti"""
        now = datetime.now()
        hafta_basi = now - timedelta(days=now.weekday())
        hafta_basi = hafta_basi.replace(hour=0, minute=0, second=0, microsecond=0)

        toplam_gelir = 0
        toplam_gider = 0

        for islem in self.veriler["islemler"]:
            islem_tarih = datetime.fromisoformat(islem["tarih"])
            if islem_tarih >= hafta_basi:
                if islem["tip"] == "gelir":
                    toplam_gelir += islem["tutar"]
                else:
                    toplam_gider += islem["tutar"]

        return {
            "toplam_gelir": toplam_gelir,
            "toplam_gider": toplam_gider,
            "net": toplam_gelir - toplam_gider
        }

    def format_bakiye_mesaj(self) -> str:
        """Bakiye mesajını formatla"""
        bakiye = self.bakiye_hesapla()
        hafta = self.bu_hafta_ozet()
        ay = self.bu_ay_ozet()

        emoji = "💰" if bakiye >= 0 else "🔴"

        mesaj = f"{emoji} *CÜZDANIM*\n\n"
        mesaj += f"💵 Bakiye: *{bakiye:,.2f}₺*\n\n"
        mesaj += f"📅 Bu Hafta:\n"
        mesaj += f"   ➕ Gelir: {hafta['toplam_gelir']:,.2f}₺\n"
        mesaj += f"   ➖ Gider: {hafta['toplam_gider']:,.2f}₺\n\n"
        mesaj += f"📆 Bu Ay:\n"
        mesaj += f"   ➕ Gelir: {ay['toplam_gelir']:,.2f}₺\n"
        mesaj += f"   ➖ Gider: {ay['toplam_gider']:,.2f}₺"

        return mesaj

    def format_rapor_mesaj(self) -> str:
        """Aylık rapor mesajı"""
        ay = self.bu_ay_ozet()
        bakiye = self.bakiye_hesapla()

        # Bu ayın "Diğer" kategorisindeki işlemlerini al (açıklama detayı için)
        now = datetime.now()
        ay_basi = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        diger_gelir_detay = defaultdict(float)
        diger_gider_detay = defaultdict(float)

        for islem in self.veriler["islemler"]:
            islem_tarih = datetime.fromisoformat(islem["tarih"])
            if islem_tarih >= ay_basi:
                aciklama = islem.get("aciklama", "").strip()
                if islem["kategori"] == "diger_gelir" and aciklama:
                    diger_gelir_detay[aciklama] += islem["tutar"]
                elif islem["kategori"] == "diger_gider" and aciklama:
                    diger_gider_detay[aciklama] += islem["tutar"]

        mesaj = f"📊 *AYLIK RAPOR - {ay['ay'].upper()}*\n\n"
        mesaj += f"💵 Mevcut Bakiye: *{bakiye:,.2f}₺*\n\n"

        mesaj += f"➕ *Toplam Gelir:* {ay['toplam_gelir']:,.2f}₺\n"
        if ay['gelir_kategorileri']:
            for kat, tutar in sorted(ay['gelir_kategorileri'].items(), key=lambda x: -x[1]):
                emoji, isim = self.GELIR_KATEGORILERI.get(kat, ("💰", kat))
                mesaj += f"   {emoji} {isim}: {tutar:,.2f}₺\n"
                # Diğer kategorisinde detay göster
                if kat == "diger_gelir" and diger_gelir_detay:
                    for acik, tut in sorted(diger_gelir_detay.items(), key=lambda x: -x[1]):
                        mesaj += f"      • {acik}: {tut:,.2f}₺\n"

        mesaj += f"\n➖ *Toplam Gider:* {ay['toplam_gider']:,.2f}₺\n"
        if ay['gider_kategorileri']:
            for kat, tutar in sorted(ay['gider_kategorileri'].items(), key=lambda x: -x[1]):
                emoji, isim = self.GIDER_KATEGORILERI.get(kat, ("💸", kat))
                mesaj += f"   {emoji} {isim}: {tutar:,.2f}₺\n"
                # Diğer kategorisinde detay göster
                if kat == "diger_gider" and diger_gider_detay:
                    for acik, tut in sorted(diger_gider_detay.items(), key=lambda x: -x[1]):
                        mesaj += f"      • {acik}: {tut:,.2f}₺\n"

        net = ay['toplam_gelir'] - ay['toplam_gider']
        net_emoji = "📈" if net >= 0 else "📉"
        mesaj += f"\n{net_emoji} *Net:* {net:+,.2f}₺"

        return mesaj

    def format_son_islemler_mesaj(self, limit: int = 10) -> str:
        """Son işlemler mesajı"""
        islemler = self.son_islemler(limit)

        if not islemler:
            return "📋 Henüz işlem yok."

        mesaj = f"📋 *SON {len(islemler)} İŞLEM*\n\n"

        for islem in islemler:
            if islem["tip"] == "gelir":
                kat_bilgi = self.GELIR_KATEGORILERI.get(islem["kategori"], ("💰", islem["kategori"]))
                emoji = "➕"
                isaret = "+"
            else:
                kat_bilgi = self.GIDER_KATEGORILERI.get(islem["kategori"], ("💸", islem["kategori"]))
                emoji = "➖"
                isaret = "-"

            mesaj += f"{emoji} {isaret}{islem['tutar']:,.2f}₺ | {kat_bilgi[0]} {kat_bilgi[1]}\n"
            mesaj += f"   📅 {islem['gun']} {islem['saat']}"
            if islem.get("aciklama"):
                mesaj += f" - {islem['aciklama']}"
            mesaj += f" `#{islem['id']}`\n\n"

        return mesaj.strip()

    def format_islem_onay(self, tip: str, tutar: float, kategori: str, aciklama: str = "") -> str:
        """İşlem onay mesajı"""
        if tip == "gelir":
            kat_bilgi = self.GELIR_KATEGORILERI.get(kategori, ("💰", kategori))
            mesaj = f"✅ *Gelir Eklendi*\n\n➕ +{tutar:,.2f}₺\n{kat_bilgi[0]} Kategori: {kat_bilgi[1]}"
        else:
            kat_bilgi = self.GIDER_KATEGORILERI.get(kategori, ("💸", kategori))
            mesaj = f"✅ *Gider Eklendi*\n\n➖ -{tutar:,.2f}₺\n{kat_bilgi[0]} Kategori: {kat_bilgi[1]}"

        if aciklama:
            mesaj += f"\n📝 {aciklama}"

        return mesaj


# Test
if __name__ == "__main__":
    c = Cuzdan("test_user")

    # Test işlemleri
    c.baslangic_bakiye_ayarla(5000)
    c.gelir_ekle(25000, "maas", "Ocak maaşı")
    c.gider_ekle(500, "market", "Haftalık alışveriş")
    c.gider_ekle(150, "yemek", "Dışarıda yemek")
    c.gider_ekle(1200, "fatura", "Elektrik + Doğalgaz")

    print(c.format_bakiye_mesaj())
    print("\n" + "="*50 + "\n")
    print(c.format_rapor_mesaj())
    print("\n" + "="*50 + "\n")
    print(c.format_son_islemler_mesaj())
