"""
Türkçe Sohbet Zekası - Turkish Conversation Intelligence

İki Türk arkadaş gibi doğal sohbet akışını anlayan katman.
DecisionLLM'den ÖNCE çalışır - hızlı ve kültürel.

Amaç: "Wooow mükemel bayıldım tamam" gibi mesajları
sadece "onay" değil, duygu + niyet + beklenti olarak analiz etmek.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SohbetEnerjisi(Enum):
    """Sohbetin enerji/tempo seviyesi"""
    COK_YUKSEK = "çok_yüksek"  # "OHAAA", "VAY BE!!!"
    YUKSEK = "yüksek"          # "Wooow", "Harika!"
    NORMAL = "normal"          # Normal sohbet
    DUSUK = "düşük"            # "tamam", "ok"
    KAPANIYOR = "kapanıyor"    # "görüşürüz", "hadi bb"


class BeklenenCevap(Enum):
    """AI'dan beklenen cevap tipi"""
    KISA_POZITIF = "kısa_pozitif"       # 1-2 cümle, pozitif
    KISA_NOTR = "kısa_nötr"             # 1-2 cümle, nötr
    KISA_VEDA = "kısa_veda"             # Samimi vedalaşma
    ACIKLAMA = "açıklama"               # Detaylı açıklama
    SORU_CEVAP = "soru_cevap"           # Direkt cevap
    DEVAM_ANLAT = "devam_anlat"         # Hikaye/anlatı devamı
    EMPATI = "empati"                   # Anlayış göster
    MERAK_GOSTER = "merak_göster"       # Karşı soru sor
    NORMAL = "normal"                   # Standart cevap


@dataclass
class SohbetAnalizi:
    """Sohbet analiz sonucu"""
    durumlar: List[str]
    kombinasyon: Optional[str]
    beklenen_cevap: BeklenenCevap
    ortuk_istek: Optional[str]
    sohbet_enerjisi: SohbetEnerjisi
    devam_beklentisi: bool
    duygu: Optional[str]
    konu_degisimi: bool
    onceki_konuyu_kapat: bool
    guven_skoru: float  # 0.0 - 1.0
    emoji_duygular: List[str] = None  # Emoji'lerden çıkarılan duygular
    ogrenme_modu: bool = False  # Ardarda soru = öğrenme/araştırma modu


class TurkishConversationIntelligence:
    """
    Türkçe sohbetin doğal akışını anlayan ana sınıf.

    Özellikler:
    - Türkçe sohbet kalıplarını tanır
    - Duygu + niyet kombinasyonlarını algılar
    - Örtük (implicit) istekleri çıkarır
    - Sohbet enerjisini takip eder
    - Konu geçişlerini algılar
    """

    # ═══════════════════════════════════════════════════════════════
    # TÜRKÇE SOHBET KALIPLARI
    # ═══════════════════════════════════════════════════════════════

    SOHBET_DURUMLARI = {
        # Selamlaşma
        "selamlasma": [
            "selam", "merhaba", "mrb", "slm", "selamın aleyküm", "selamun aleyküm",
            "günaydın", "iyi akşamlar", "iyi geceler", "hayırlı sabahlar",
            "naber", "nbr", "nabıon", "napıyon", "nasılsın", "nasıl gidiyor",
            "ne haber", "n'aber", "noldu", "ne var ne yok"
        ],

        # Heyecan / Pozitif şaşırma
        "heyecan": [
            "wooow", "wow", "woow", "vay", "vay be", "vay canına", "oha", "ohaa",
            "harika", "süper", "muhteşem", "mükemmel", "mükemel", "mükeemmel",
            "efsane", "çok iyi", "bayıldım", "aşırı iyi", "off", "offf",
            "yuppi", "yeey", "yaşasın", "heleyy"
        ],

        # Onay / Kabul
        "onay": [
            "tamam", "tamam tamam", "tamamdır", "ok", "okay", "oke", "okey",
            "olur", "oldu", "peki", "hay hay", "baş üstüne",
            "aynen", "aynen öyle", "kesinlikle", "evet", "he", "hee", "hı hı",
            "anladım", "anladım anladım", "anlaşıldı", "kabul"
        ],

        # Teşekkür
        "tesekkur": [
            "teşekkürler", "teşekkür ederim", "teşekkür", "tesekkur", "tesekkurler",
            "sağol", "sağ ol", "saol", "saolasin", "sağolasın",
            "eyvallah", "eyv", "eyw", "Allah razı olsun", "çok sağol",
            "minnettarım", "eline sağlık", "emeğine sağlık"
        ],

        # Geçiş / Konu değiştirme sinyalleri
        "gecis": [
            "gel", "hadi", "şimdi", "peki", "neyse", "herneyse",
            "bir de", "bu arada", "ayrıca", "ha bir de", "ha bu arada",
            "başka", "başka bir şey", "farklı bir konu",
            "konuyu değiştirelim", "başka şeyden konuşalım"
        ],

        # Vedalaşma
        "veda": [
            "görüşürüz", "görüşmek üzere", "hoşçakal", "hoşça kal",
            "hadi", "hadi bakalım", "hadi eyvallah", "hadi bye",
            "bye", "bb", "bay bay", "bye bye",
            "iyi geceler", "kendine iyi bak", "dikkat et kendine",
            "haydi", "kalın sağlıcakla", "Allah'a emanet",
            # Karşılıklı veda (kullanıcı AI'ın vedasına cevap veriyor)
            "sende", "sen de", "sana da", "size de", "sende de",
            "iyi günler sana da", "sana da iyi geceler"
        ],

        # Samimi hitap (arkadaşça)
        "samimi_hitap": [
            "kanka", "kanki", "dostum", "kardeşim", "bro", "moruk",
            "hacı", "abi", "abicim", "reis", "şef", "patron",
            "canım", "tatlım", "güzelim", "birtanem"
        ],

        # Merak / Soru
        "merak": [
            "neden", "niye", "niçin", "nasıl", "ne", "kim", "nerde", "nerede",
            "ne zaman", "hangi", "kaç", "ne kadar",
            "nası yani", "nasıl yani", "yani nasıl", "ne demek",
            "açıklar mısın", "anlatır mısın", "söyler misin"
        ],

        # Devam isteği
        "devam_istek": [
            "anlat", "anlatır mısın", "anlatsana", "devam et", "devam",
            "sonra", "peki sonra", "sonra ne oldu", "e sonra",
            "daha fazla", "biraz daha", "detay ver", "detaylı anlat",
            "açıkla", "açıklasana", "örnek ver"
        ],

        # Duraklama / Düşünme
        "duraklama": [
            "hmm", "hmmm", "şey", "yani", "hani", "bak", "şöyle",
            "nasıl desem", "şimdi", "bir dakika", "dur", "bekle"
        ],

        # Olumsuz / Ret
        "olumsuz": [
            "hayır", "yok", "olmaz", "istemem", "istemiyorum",
            "değil", "hiç", "asla", "kesinlikle hayır"
        ],

        # Şikayet / Sızlanma
        "sikayet": [
            "yoruldum", "yorgunum", "sıkıldım", "bıktım", "canım sıkkın",
            "kötüyüm", "moralim bozuk", "üzgünüm", "mutsuzum"
        ],

        # Empati isteme
        "empati_iste": [
            "biliyo musun", "biliyon mu", "var ya", "yaa", "düşünsene",
            "inanamıyorum", "çok kötü", "çok zor"
        ],

        # Onaylama beklentisi
        "onay_bekle": [
            "değil mi", "di mi", "dimi", "öyle değil mi",
            "katılıyor musun", "sen ne düşünüyorsun", "sence"
        ],

        # Şaşırma (nötr)
        "sasirma": [
            "cidden mi", "ciddi misin", "gerçekten mi", "harbiden mi",
            "şaka yapıyorsun", "yok artık", "inanmıyorum"
        ],

        # İlgi gösterme
        "ilgi": [
            "ilginç", "enteresan", "vay", "öyle mi", "aa öyle mi",
            "hiç bilmiyordum", "ilk defa duydum"
        ],

        # Şüphe / Sorgulama (kullanıcı emin değil)
        "suphe": [
            "mı ki", "mi ki", "acaba", "sanmıyorum", "sanmam",
            "emin misin", "emin değilim", "bilmiyorum ki",
            "olabilir mi", "mümkün mü", "gerçekten öyle mi",
            "yanlış mı", "doğru mu", "şüpheliyim"
        ],

        # Kabullenme / Pasif kabul (durumu kabulleniyor)
        "kabullenme": [
            "yapacak bir şey yok", "neyse ne", "sağlık olsun",
            "olsun", "boşver", "takma", "geç", "unut gitsin",
            "ne yapalım", "kaderimiz böyle", "kabul ettim",
            "mecbur", "başka çare yok", "idare eder"
        ],

        # Vurgulama / Dikkat çekme
        "vurgulama": [
            "ya", "işte", "bak", "şimdi bak", "dinle",
            "önemli", "dikkat", "şunu söyleyeyim", "bi dakka",
            "dur dur", "ama", "fakat", "lakin", "ancak"
        ],

        # Sabırsızlık
        "sabirsizlik": [
            "hadi", "hadi ama", "çabuk", "acele et", "ne oldu",
            "bekliyorum", "yeter", "tamam tamam", "geç artık"
        ],

        # Takdir / Övgü
        "takdir": [
            "helal", "bravo", "aferin", "tebrikler", "alkış",
            "iyi iş", "süpersin", "harikasın", "efsanesin"
        ],

        # Bilgi Testi (kullanıcı AI'ı test ediyor, belirsiz referans olabilir, önce netleştir)
        "bilgi_testi": [
            "bildin mi", "bildinmi", "biliyor musun", "biliyo musun", "biliyomusun",
            "biliyon mu", "biliyonmu", "bilir misin", "bilirmisin",
            "duydun mu", "duydunmu", "duymuş muydun",
            "haberin var mı", "haberin varmı",
            "farkında mısın", "gördün mü", "gordunmu",
            "tanıyor musun", "tanıyon mu",
            "hatırlıyor musun", "hatırlıyon mu"
        ],

        # Kinaye / İroni (alaycı, ters anlam - tekrarlı ifadeler genelde ironi)
        "kinaye": [
            "tabi tabi", "tabii tabii", "tabi cnm", "tabi canım",
            "kesin öyledir", "kesin yaşanmıştır",
            "yav he he", "yav he hee", "yav he heee", "he he", "he hee", "he heee",
            "çok beklersin", "oldu canım", "hadi ordan", "tabiki tabiki", "nasıl olsa"
        ]
    }

    # ═══════════════════════════════════════════════════════════════
    # EMOJİ DUYGU ANALİZİ
    # ═══════════════════════════════════════════════════════════════

    EMOJI_DUYGULAR = {
        "kahkaha": ["😂", "🤣", "😆", "😹", "😁", "😄"],
        "sevgi": ["❤️", "😍", "🥰", "💕", "😘", "💖", "💗", "♥️", "🫶"],
        "uzuntu": ["😢", "😭", "🥺", "😞", "😔", "💔", "😿"],
        "ofke": ["😡", "🤬", "😤", "💢"],
        "onay": ["👍", "✅", "👌", "🙌", "💪", "🤝", "👏"],
        "heyecan": ["🔥", "💯", "🚀", "⭐", "🎉", "✨", "🥳", "🎊"],
        "dusunme": ["🤔", "🧐", "❓", "❔", "🤷"],
        "sasirma": ["😱", "😮", "🤯", "😲", "😳", "🙀"],
        "uzgun_gulumse": ["🥲", "😅", "😬"],
        "memnun": ["😊", "🙂", "😌", "☺️", "🤗"],
        "cool": ["😎", "🤙", "✌️"],
    }

    # ═══════════════════════════════════════════════════════════════
    # DUYGU + NİYET KOMBİNASYONLARI
    # ═══════════════════════════════════════════════════════════════

    KOMBINASYONLAR = {
        # (durum1, durum2): sonuç_aksiyon
        ("heyecan", "onay"): "memnun_kapanış",           # "Wooow tamam harika"
        ("tesekkur", "gecis"): "yeni_konu_açma",         # "Sağol, şimdi şunu..."
        ("tesekkur", "veda"): "vedalaşma",               # "Sağol görüşürüz"
        ("onay", "gecis"): "konu_değiştirme",            # "Tamam, gel şunu..."
        ("onay", "veda"): "vedalaşma",                   # "Tamam hadi görüşürüz"
        ("duraklama", "merak"): "düşünerek_sorma",       # "Hmm peki nasıl..."
        ("heyecan", "merak"): "heyecanlı_soru",          # "Vay be nasıl yaptın?"
        ("heyecan", "devam_istek"): "heyecanlı_devam",   # "Harika! Devam et!"
        ("sikayet", "empati_iste"): "destek_bekliyor",   # "Yorgunum ya biliyo musun"
        ("tesekkur", "onay"): "memnun_kapanış",          # "Sağol tamam anladım"
        ("ilgi", "merak"): "meraklı_soru",               # "İlginç, nasıl oluyor?"
        ("sasirma", "merak"): "şaşkın_soru",             # "Cidden mi? Nasıl?"
        ("onay", "devam_istek"): "anlat_devam",          # "Tamam, anlat bakalım"
        ("olumsuz", "gecis"): "konu_reddet_değiştir",    # "Yok istemem, başka şey..."
        ("selamlasma", "merak"): "selamlı_soru",         # "Selam, nasılsın?"

        # YENİ: Şüphe kombinasyonları
        ("suphe", "merak"): "aciklama_bekliyor",         # "Acaba nasıl oluyor?"
        ("suphe", "onay_bekle"): "teyit_istiyor",        # "Emin misin, doğru mu?"
        ("vurgulama", "suphe"): "itiraz_ediyor",         # "Ama sanmıyorum ki"

        # YENİ: Kabullenme kombinasyonları
        ("kabullenme", "onay"): "pasif_kabul",           # "Neyse ne, tamam"
        ("sikayet", "kabullenme"): "uzgun_kabul",        # "Yoruldum, yapacak bir şey yok"
        ("kabullenme", "gecis"): "konu_birak",           # "Boşver, başka şey konuşalım"

        # YENİ: Takdir kombinasyonları
        ("takdir", "heyecan"): "coskulu_ovgu",           # "Bravo, harika!"
        ("takdir", "tesekkur"): "karsilikli_takdir",     # "Helal, sağol"

        # YENİ: Sabırsızlık kombinasyonları
        ("sabirsizlik", "merak"): "aceleci_soru",        # "Hadi, ne oldu?"
        ("sabirsizlik", "devam_istek"): "aceleci_devam", # "Çabuk, anlat"

        # YENİ: Samimi hitap kombinasyonları
        ("veda", "samimi_hitap"): "samimi_veda",         # "Görüşürüz kanka", "Sende dostum"
        ("tesekkur", "samimi_hitap"): "samimi_tesekkur", # "Sağol kanka"
        ("selamlasma", "samimi_hitap"): "samimi_selam",  # "Naber kanka"
    }

    # ═══════════════════════════════════════════════════════════════
    # ÖRTÜK (IMPLICIT) İSTEK KALIPLARI
    # ═══════════════════════════════════════════════════════════════

    ORTUK_ISTEKLER = [
        # (regex_pattern, istek_tipi, grup_adi)
        # Esnek "Gel X konuşalım/bahsedelim/anlat" kalıpları
        (r'gel\s+(.*?)\s+(konuşalım|bahsedelim|anlat|konuşak)', 'anlat', 'konu'),
        (r'gel\s+biraz\s+(.*?)\s+(konuşalım|bahsedelim)', 'anlat', 'konu'),
        (r'hadi\s+(.*?)\s+(konuşalım|bahsedelim|anlat)', 'anlat', 'konu'),

        # "X hakkında" kalıpları
        (r'(.*?)\s+hakkında\s+(konuşalım|anlat|bilgi)', 'anlat', 'konu'),
        (r'(.*?)\s+hakkında\s+ne\s+düşünüyorsun', 'görüş_sor', 'konu'),

        # "X anlatır mısın / anlatsana" kalıpları
        (r'(.*?)\s+anlatır\s*mısın', 'anlat', 'konu'),
        (r'(.*?)\s+anlatsana', 'anlat', 'konu'),
        (r'(.*?)\s+anlat\s+bana', 'anlat', 'konu'),
        (r'biraz\s+(.*?)\s+konuşalım', 'anlat', 'konu'),

        # Açıklama istekleri
        (r'nasıl\s+yani', 'açıkla', None),
        (r'nası\s+yani', 'açıkla', None),
        (r'yani\s+nasıl', 'açıkla', None),
        (r'ne\s+demek\s+(bu|o|şu)', 'açıkla', None),
        (r'tam\s+anlamadım', 'açıkla', None),
        (r'biraz\s+açar\s*mısın', 'açıkla', None),

        # Dikkat çekme
        (r'biliyo\s*musun', 'dikkat_çek', None),
        (r'biliyon\s*mu', 'dikkat_çek', None),
        (r'var\s+ya', 'dikkat_çek', None),
        (r'düşünsene', 'hayal_et', None),

        # Yardım istekleri
        (r'bir\s+bakar\s*mısın', 'yardım', None),
        (r'yardım\s+eder\s*misin', 'yardım', None),
        (r'yardımcı\s+olur\s*musun', 'yardım', None),
        (r'el\s+atar\s*mısın', 'yardım', None),

        # Görüş sorma
        (r'ne\s+dersin', 'görüş_sor', None),
        (r'sence\s+(nasıl|ne)', 'görüş_sor', None),
        (r'fikrin\s+ne', 'görüş_sor', None),
        (r'sen\s+ne\s+düşünüyorsun', 'görüş_sor', None),
        (r'senin\s+fikrin', 'görüş_sor', None),

        # Öneri isteme
        (r'ne\s+yapmalıyım', 'öneri_iste', None),
        (r'ne\s+önerirsin', 'öneri_iste', None),
        (r'tavsiye\s+eder\s*misin', 'öneri_iste', None),

        # Karşılaştırma
        (r'hangisi\s+daha\s+(iyi|güzel|mantıklı)', 'karşılaştır', None),
        (r'(.*?)\s+mı\s+(.*?)\s+mı', 'karşılaştır', None),
    ]

    # ═══════════════════════════════════════════════════════════════
    # KONU DEĞİŞİM SİNYALLERİ
    # ═══════════════════════════════════════════════════════════════

    KONU_DEGISIM_SINYALLERI = [
        "şimdi", "peki şimdi", "neyse", "başka", "farklı",
        "konuyu değiştir", "gel şunu", "bir de şu", "ha bir de"
    ]

    KONU_KAPAMA_SINYALLERI = [
        "tamam", "anladım", "ok", "sağol", "teşekkürler",
        "güzel", "harika", "süper", "yeter", "bu kadar"
    ]

    def __init__(self):
        """Türkçe sohbet zekası başlat"""
        self._compile_patterns()
        print("[OK] Turkce Sohbet Zekasi aktif!")

    def _compile_patterns(self):
        """Regex patternlarını önceden derle (performans için)"""
        self._ortuk_patterns = []
        for pattern, istek_tipi, grup in self.ORTUK_ISTEKLER:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._ortuk_patterns.append((compiled, istek_tipi, grup))
            except re.error:
                pass

    def analiz_et(
        self,
        mesaj: str,
        onceki_mesajlar: List[Dict[str, Any]] = None
    ) -> SohbetAnalizi:
        """
        Ana analiz fonksiyonu.

        Args:
            mesaj: Kullanıcının son mesajı
            onceki_mesajlar: Önceki sohbet geçmişi

        Returns:
            SohbetAnalizi: Detaylı analiz sonucu
        """
        mesaj_lower = mesaj.lower().strip()
        mesaj_clean = self._temizle(mesaj_lower)

        # 1. Durumları tespit et
        durumlar = self._durumlari_tespit_et(mesaj_clean)

        # 2. Emoji analizi - emoji'lerden duygu çıkar ve durumlara ekle
        emoji_duygular = self._emoji_analiz(mesaj)
        emoji_sayisi = self._emoji_sayisi(mesaj)

        # Emoji duyguları -> durum mapping
        emoji_durum_map = {
            "kahkaha": "heyecan",
            "sevgi": "heyecan",
            "heyecan": "heyecan",
            "memnun": "onay",
            "onay": "onay",
            "uzuntu": "sikayet",
            "ofke": "sikayet",
            "sasirma": "sasirma",
            "dusunme": "merak",
        }
        for emoji_duygu in emoji_duygular:
            mapped = emoji_durum_map.get(emoji_duygu)
            if mapped and mapped not in durumlar:
                durumlar.append(mapped)

        # 3. Ardarda soru kontrolü (öğrenme modu)
        ogrenme_modu = self._ardarda_soru_kontrolu(onceki_mesajlar)

        # 4. Kombinasyon bul
        kombinasyon = self._kombinasyon_bul(durumlar)

        # 5. Duygu analizi (emoji duyguları da dahil)
        duygu = self._duygu_analiz(durumlar, mesaj)
        # Emoji'den gelen duyguyu önceliklendir
        if emoji_duygular:
            duygu = emoji_duygular[0]  # İlk emoji duygusu

        # 6. Enerji seviyesi (emoji sayısı da etkiler)
        enerji = self._enerji_hesapla(durumlar, mesaj)
        # Çok emoji varsa enerjiyi yükselt
        if emoji_sayisi >= 3 and enerji == SohbetEnerjisi.NORMAL:
            enerji = SohbetEnerjisi.YUKSEK

        # 7. Beklenen cevap tipi
        beklenen = self._beklenen_cevap_belirle(durumlar, kombinasyon, duygu)
        # Öğrenme modundaysa detaylı cevap bekle
        if ogrenme_modu and beklenen == BeklenenCevap.NORMAL:
            beklenen = BeklenenCevap.ACIKLAMA

        # 8. Örtük istek
        ortuk_istek = self._ortuk_istek_bul(mesaj_clean)

        # 9. Konu değişimi kontrolü
        konu_degisimi = self._konu_degisimi_var_mi(mesaj_clean, durumlar)
        onceki_konuyu_kapat = self._onceki_konu_kapaniyor_mu(durumlar, kombinasyon)

        # 10. Devam beklentisi
        devam_beklentisi = "gecis" in durumlar or "merak" in durumlar or "devam_istek" in durumlar or ogrenme_modu

        # 11. Güven skoru (emoji varsa güveni artır)
        guven = self._guven_skoru_hesapla(durumlar, kombinasyon)
        if emoji_duygular:
            guven = min(guven + 0.1, 1.0)

        return SohbetAnalizi(
            durumlar=durumlar,
            kombinasyon=kombinasyon,
            beklenen_cevap=beklenen,
            ortuk_istek=ortuk_istek,
            sohbet_enerjisi=enerji,
            devam_beklentisi=devam_beklentisi,
            duygu=duygu,
            konu_degisimi=konu_degisimi,
            onceki_konuyu_kapat=onceki_konuyu_kapat,
            guven_skoru=guven,
            emoji_duygular=emoji_duygular,
            ogrenme_modu=ogrenme_modu
        )

    def _temizle(self, mesaj: str) -> str:
        """Mesajı temizle (noktalama, fazla boşluk)"""
        # Tekrarlı harfleri normalize et (wooooow -> wooow)
        mesaj = re.sub(r'(.)\1{3,}', r'\1\1\1', mesaj)
        # Fazla boşlukları temizle
        mesaj = re.sub(r'\s+', ' ', mesaj)
        return mesaj.strip()

    def _durumlari_tespit_et(self, mesaj: str) -> List[str]:
        """Mesajdaki tüm sohbet durumlarını tespit et"""
        tespit_edilenler = []

        for durum, kaliplar in self.SOHBET_DURUMLARI.items():
            for kalip in kaliplar:
                # Tam kelime eşleşmesi için word boundary
                pattern = r'\b' + re.escape(kalip) + r'\b'
                if re.search(pattern, mesaj, re.IGNORECASE):
                    if durum not in tespit_edilenler:
                        tespit_edilenler.append(durum)
                    break

        return tespit_edilenler

    def _emoji_analiz(self, mesaj: str) -> List[str]:
        """Mesajdaki emoji'lerden duygu durumları çıkar"""
        tespit_edilen_duygular = []

        for duygu, emojiler in self.EMOJI_DUYGULAR.items():
            for emoji in emojiler:
                if emoji in mesaj:
                    if duygu not in tespit_edilen_duygular:
                        tespit_edilen_duygular.append(duygu)
                    break

        return tespit_edilen_duygular

    def _emoji_sayisi(self, mesaj: str) -> int:
        """Mesajdaki toplam emoji sayısını hesapla"""
        sayac = 0
        for emojiler in self.EMOJI_DUYGULAR.values():
            for emoji in emojiler:
                sayac += mesaj.count(emoji)
        return sayac

    def _ardarda_soru_kontrolu(self, onceki_mesajlar: List[Dict[str, Any]]) -> bool:
        """Kullanıcı ardarda soru mu soruyor? (öğrenme/araştırma modu)"""
        if not onceki_mesajlar or len(onceki_mesajlar) < 4:
            return False

        # Son 4 mesaja bak (2 user, 2 assistant)
        soru_isareti_sayisi = 0
        merak_kelimeleri = ["ne", "nasıl", "neden", "niye", "kim", "nerede", "kaç", "hangi"]

        for msg in onceki_mesajlar[-4:]:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                if "?" in content or any(k in content for k in merak_kelimeleri):
                    soru_isareti_sayisi += 1

        # 2 veya daha fazla ardarda soru = öğrenme modu
        return soru_isareti_sayisi >= 2

    def _kombinasyon_bul(self, durumlar: List[str]) -> Optional[str]:
        """Durum kombinasyonunu bul"""
        if len(durumlar) < 2:
            return None

        # Tüm ikili kombinasyonları dene
        for i, d1 in enumerate(durumlar):
            for d2 in durumlar[i+1:]:
                # Her iki sırayı da dene
                if (d1, d2) in self.KOMBINASYONLAR:
                    return self.KOMBINASYONLAR[(d1, d2)]
                if (d2, d1) in self.KOMBINASYONLAR:
                    return self.KOMBINASYONLAR[(d2, d1)]

        return None

    def _duygu_analiz(self, durumlar: List[str], mesaj: str) -> Optional[str]:
        """Mesajdaki ana duyguyu belirle"""
        if "kinaye" in durumlar:
            return "kinaye"
        elif "heyecan" in durumlar:
            return "heyecan"
        elif "sikayet" in durumlar:
            return "şikayetçi/üzüntü"
        elif "sasirma" in durumlar:
            return "şaşkınlık"
        elif "tesekkur" in durumlar:
            return "memnun"
        elif "olumsuz" in durumlar:
            return "nötr"
        elif "merak" in durumlar:
            return "merak"
        elif "selamlasma" in durumlar:
            return "samimi"
        elif "veda" in durumlar:
            return "vedalaşma"
        elif "onay" in durumlar:
            return "kabul"
        return "samimi"

    def _enerji_hesapla(self, durumlar: List[str], mesaj: str) -> SohbetEnerjisi:
        """Sohbetin enerji seviyesini hesapla"""
        # Ünlem sayısı
        unlem_sayisi = mesaj.count('!')

        # Büyük harf oranı
        if len(mesaj) > 0:
            buyuk_harf_orani = sum(1 for c in mesaj if c.isupper()) / len(mesaj)
        else:
            buyuk_harf_orani = 0

        # Heyecan kelimeleri
        if "heyecan" in durumlar and (unlem_sayisi >= 2 or buyuk_harf_orani > 0.5):
            return SohbetEnerjisi.COK_YUKSEK
        elif "heyecan" in durumlar:
            return SohbetEnerjisi.YUKSEK
        elif "veda" in durumlar:
            return SohbetEnerjisi.KAPANIYOR
        elif "onay" in durumlar and len(durumlar) == 1:
            return SohbetEnerjisi.DUSUK
        elif "sikayet" in durumlar:
            return SohbetEnerjisi.DUSUK
        else:
            return SohbetEnerjisi.NORMAL

    def _beklenen_cevap_belirle(
        self,
        durumlar: List[str],
        kombinasyon: Optional[str],
        duygu: Optional[str]
    ) -> BeklenenCevap:
        """AI'dan beklenen cevap tipini belirle"""

        # Kombinasyona göre
        if kombinasyon == "vedalaşma":
            return BeklenenCevap.KISA_VEDA
        elif kombinasyon == "memnun_kapanış":
            return BeklenenCevap.KISA_POZITIF
        elif kombinasyon == "destek_bekliyor":
            return BeklenenCevap.EMPATI
        elif kombinasyon in ["heyecanlı_soru", "meraklı_soru", "şaşkın_soru"]:
            return BeklenenCevap.ACIKLAMA
        elif kombinasyon == "heyecanlı_devam":
            return BeklenenCevap.DEVAM_ANLAT
        elif kombinasyon == "yeni_konu_açma":
            return BeklenenCevap.ACIKLAMA  # Kullanıcı yeni bilgi istiyor
        elif kombinasyon == "konu_değiştirme":
            return BeklenenCevap.NORMAL
        elif kombinasyon == "samimi_veda":
            return BeklenenCevap.KISA_VEDA
        elif kombinasyon == "samimi_tesekkur":
            return BeklenenCevap.KISA_POZITIF
        elif kombinasyon == "samimi_selam":
            return BeklenenCevap.NORMAL

        # Tek duruma göre
        if "veda" in durumlar:
            return BeklenenCevap.KISA_VEDA
        elif "devam_istek" in durumlar:
            return BeklenenCevap.DEVAM_ANLAT
        elif "merak" in durumlar:
            return BeklenenCevap.ACIKLAMA
        elif "sikayet" in durumlar or "empati_iste" in durumlar:
            return BeklenenCevap.EMPATI
        elif "onay" in durumlar and len(durumlar) == 1:
            return BeklenenCevap.KISA_NOTR
        elif "heyecan" in durumlar and len(durumlar) == 1:
            return BeklenenCevap.KISA_POZITIF
        elif "selamlasma" in durumlar:
            return BeklenenCevap.NORMAL
        elif "onay_bekle" in durumlar:
            return BeklenenCevap.SORU_CEVAP

        return BeklenenCevap.NORMAL

    def _ortuk_istek_bul(self, mesaj: str) -> Optional[str]:
        """Örtük (implicit) istekleri tespit et"""
        for pattern, istek_tipi, grup in self._ortuk_patterns:
            match = pattern.search(mesaj)
            if match:
                if grup == 'konu' and match.groups():
                    konu = match.group(1).strip()
                    return f"{istek_tipi}:{konu}"
                else:
                    return istek_tipi

        return None

    def _konu_degisimi_var_mi(self, mesaj: str, durumlar: List[str]) -> bool:
        """Konu değişimi sinyali var mı?"""
        # Geçiş durumu varsa
        if "gecis" in durumlar:
            return True

        # Konu değişim sinyalleri
        for sinyal in self.KONU_DEGISIM_SINYALLERI:
            if sinyal in mesaj:
                return True

        return False

    def _onceki_konu_kapaniyor_mu(
        self,
        durumlar: List[str],
        kombinasyon: Optional[str]
    ) -> bool:
        """Önceki konu kapanıyor mu?"""
        if kombinasyon in ["memnun_kapanış", "vedalaşma", "yeni_konu_açma", "konu_değiştirme"]:
            return True

        # Sadece onay/teşekkür varsa konu kapanıyor olabilir
        kapama_durumlari = {"onay", "tesekkur"}
        if durumlar and all(d in kapama_durumlari for d in durumlar):
            return True

        return False

    def _guven_skoru_hesapla(
        self,
        durumlar: List[str],
        kombinasyon: Optional[str]
    ) -> float:
        """Analiz güven skorunu hesapla"""
        if not durumlar:
            return 0.3  # Hiçbir şey tespit edilemedi

        skor = 0.5  # Baz skor

        # Durum sayısına göre
        skor += min(len(durumlar) * 0.1, 0.3)

        # Kombinasyon varsa bonus
        if kombinasyon:
            skor += 0.2

        return min(skor, 1.0)

    # ═══════════════════════════════════════════════════════════════
    # YARDIMCI METODLAR
    # ═══════════════════════════════════════════════════════════════

    def hizli_niyet(self, mesaj: str) -> Dict[str, Any]:
        """
        Hızlı niyet tespiti - LLM'e sormadan karar ver.

        Returns:
            dict: {
                "bypass_llm": bool,  # LLM'i atla mı?
                "question_type": str,
                "response_style": str,
                "confidence": float
            }
        """
        analiz = self.analiz_et(mesaj)

        result = {
            "bypass_llm": False,
            "question_type": "general",
            "response_style": "normal",
            "confidence": analiz.guven_skoru,
            "analiz": analiz
        }

        # Yüksek güvenle LLM bypass edilebilecek durumlar
        if analiz.guven_skoru >= 0.7:

            if analiz.kombinasyon == "vedalaşma" or "veda" in analiz.durumlar:
                result["bypass_llm"] = True
                result["question_type"] = "farewell"
                result["response_style"] = "kısa_veda"

            elif analiz.kombinasyon == "memnun_kapanış":
                result["bypass_llm"] = True
                result["question_type"] = "acknowledgment"
                result["response_style"] = "kısa_pozitif"

            elif analiz.durumlar == ["selamlasma"]:
                result["bypass_llm"] = True
                result["question_type"] = "greeting"
                result["response_style"] = "normal"

            elif analiz.durumlar == ["onay"]:
                result["bypass_llm"] = True
                result["question_type"] = "acknowledgment"
                result["response_style"] = "kısa_nötr"

        return result

    def cevap_uzunlugu_onerisi(self, analiz: SohbetAnalizi) -> Tuple[int, int]:
        """
        Beklenen cevap uzunluğu önerisi (min, max karakter)
        """
        beklenen = analiz.beklenen_cevap

        uzunluklar = {
            BeklenenCevap.KISA_POZITIF: (20, 100),
            BeklenenCevap.KISA_NOTR: (10, 60),
            BeklenenCevap.KISA_VEDA: (30, 120),
            BeklenenCevap.ACIKLAMA: (150, 500),
            BeklenenCevap.SORU_CEVAP: (50, 200),
            BeklenenCevap.DEVAM_ANLAT: (200, 600),
            BeklenenCevap.EMPATI: (80, 200),
            BeklenenCevap.MERAK_GOSTER: (30, 100),
            BeklenenCevap.NORMAL: (100, 400),
        }

        return uzunluklar.get(beklenen, (100, 400))

    def debug_analiz(self, mesaj: str, onceki_mesajlar: List[Dict[str, Any]] = None) -> str:
        """Debug için detaylı analiz çıktısı"""
        analiz = self.analiz_et(mesaj, onceki_mesajlar)

        output = []
        output.append(f"Mesaj: \"{mesaj}\"")
        output.append(f"Durumlar: {analiz.durumlar}")
        output.append(f"Kombinasyon: {analiz.kombinasyon}")
        output.append(f"Duygu: {analiz.duygu}")
        output.append(f"Emoji Duygular: {analiz.emoji_duygular}")
        output.append(f"Enerji: {analiz.sohbet_enerjisi.value}")
        output.append(f"Beklenen Cevap: {analiz.beklenen_cevap.value}")
        output.append(f"Ogrenme Modu: {analiz.ogrenme_modu}")
        output.append(f"Ortuk Istek: {analiz.ortuk_istek}")
        output.append(f"Konu Degisimi: {analiz.konu_degisimi}")
        output.append(f"Onceki Konu Kapaniyor: {analiz.onceki_konuyu_kapat}")
        output.append(f"Guven Skoru: {analiz.guven_skoru:.2f}")

        return "\n".join(output)


# Test icin
if __name__ == "__main__":
    tci = TurkishConversationIntelligence()

    test_mesajlar = [
        "Wooow mukemel bayildim tamam",
        "Gel biraz uyumadan once kara delikleri konusalim senle",
        "Tamam saol guzel yolculuktu bana simdi allahin kudret sifatlarini anlat",
        "Cok guzel saol hadi gorusuruz",
        "Naber nasilsin",
        "Hmm peki bu nasil oluyor?",
        "Yoruldum ya biliyo musun",
        "Aynen oyle kesinlikle",
        # Emoji testleri
        "😂😂😂 çok komik",
        "harika olmuş ❤️🔥",
        "anladım 👍",
        "😭😭 çok üzüldüm",
        "🤔 emin değilim",
    ]

    print("=" * 60)
    print("TURKCE SOHBET ZEKASI TEST")
    print("=" * 60)

    for mesaj in test_mesajlar:
        print("\n" + "-" * 40)
        print(tci.debug_analiz(mesaj))

        hizli = tci.hizli_niyet(mesaj)
        if hizli["bypass_llm"]:
            print(f"LLM BYPASS: {hizli['question_type']} ({hizli['response_style']})")
