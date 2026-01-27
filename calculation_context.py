"""
Hesaplama Değişkenleri Yöneticisi
Konuşmada geçen sayısal değerleri otomatik yakalar ve saklar.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime


def turkish_word_to_number(text: str) -> str:
    """
    Türkçe sayı kelimelerini rakama çevirir.
    Örn: "yüz metre kare" -> "100 metre kare"
    """
    # Temel sayılar
    ones = {
        'bir': 1, 'iki': 2, 'üç': 3, 'uc': 3, 'dört': 4, 'dort': 4,
        'beş': 5, 'bes': 5, 'altı': 6, 'alti': 6, 'yedi': 7,
        'sekiz': 8, 'dokuz': 9
    }
    tens = {
        'on': 10, 'yirmi': 20, 'otuz': 30, 'kırk': 40, 'kirk': 40,
        'elli': 50, 'altmış': 60, 'altmis': 60, 'yetmiş': 70, 'yetmis': 70,
        'seksen': 80, 'doksan': 90
    }
    hundreds = {'yüz': 100, 'yuz': 100}
    thousands = {'bin': 1000}

    result = text.lower()

    # Önce bileşik sayıları çevir (örn: "iki yüz elli" -> 250)
    # Basit yaklaşım: Tek kelimelik sayıları çevir

    # "yüz" tek başına 100
    result = re.sub(r'\byüz\b(?!\s*(?:de|da|den|dan|ü|u))', '100', result)
    result = re.sub(r'\byuz\b(?!\s*(?:de|da|den|dan|u))', '100', result)

    # "bin" tek başına 1000
    result = re.sub(r'\bbin\b(?!\s*(?:de|da|den|dan|i|e))', '1000', result)

    # Onlar (on, yirmi, otuz...)
    for word, num in tens.items():
        result = re.sub(rf'\b{word}\b', str(num), result)

    # Birler (bir, iki, üç...) - sadece birim öncesinde
    for word, num in ones.items():
        # "iki metre" -> "2 metre" ama "birisi" değişmemeli
        result = re.sub(rf'\b{word}\s+(metre|m²|m³|kilo|kg|ton|kat)', rf'{num} \1', result)

    return result


class CalculationContext:
    """
    Konuşmada belirlenen hesaplama değişkenlerini tutar.
    Her prompt'a bu değişkenler eklenir, LLM tutarlı hesaplama yapar.
    """

    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.history: List[Dict] = []  # Değişken geçmişi

        # Birim dönüşümleri
        self.unit_aliases = {
            'm²': ['metrekare', 'metre kare', 'm2', 'metrekare'],
            'm³': ['metreküp', 'metre küp', 'm3', 'metrekup', 'küp'],
            'm': ['metre', 'mt'],
            'kg': ['kilogram', 'kilo'],
            'ton': ['ton'],
            '%': ['yüzde', 'yuzde', 'oran'],
            'kat': ['katlı', 'katli', 'kat'],
            'adet': ['adet', 'tane'],
        }

        # Değişken pattern'leri - (regex, değişken_adı, birim)
        self.patterns = [
            # Alan
            (r'(\d+(?:[.,]\d+)?)\s*(?:metre\s*kare|metrekare|m2|m²)', 'alan', 'm²'),
            # Hacim
            (r'(\d+(?:[.,]\d+)?)\s*(?:metre\s*küp|metreküp|m3|m³|küp)', 'hacim', 'm³'),
            # Yükseklik (hem "yükseklik" hem "yukseklik" destekli)
            (r'y[uü]ksekli[kğg][ıi]?\s*(?::|=|,)?\s*(\d+(?:[.,]\d+)?)\s*(?:metre|m)?', 'yukseklik', 'm'),
            (r'(\d+(?:[.,]\d+)?)\s*(?:metre|m)\s*y[uü]ksekli', 'yukseklik', 'm'),
            # Raf sayısı
            (r'(\d+)\s*(?:katlı|katli|kat)\s*(?:raf|sistem)', 'raf_sayisi', 'kat'),
            # Verim oranı
            (r'[%yüzde]\s*(\d+(?:[.,]\d+)?)', 'verim_orani', '%'),
            (r'(\d+(?:[.,]\d+)?)\s*[%]', 'verim_orani', '%'),
            # Ağırlık (kg)
            (r'(\d+(?:[.,]\d+)?)\s*(?:kg|kilogram|kilo)', 'agirlik', 'kg'),
            # Ağırlık (ton)
            (r'(\d+(?:[.,]\d+)?)\s*ton', 'agirlik_ton', 'ton'),
            # Yoğunluk
            (r'(\d+(?:[.,]\d+)?)\s*kg\s*/\s*(?:m³|metreküp|m3)', 'yogunluk', 'kg/m³'),
        ]

    def extract_from_text(self, text: str, is_user: bool = True) -> Dict[str, Any]:
        """
        Metinden sayısal değerleri çıkar.

        Args:
            text: Mesaj metni
            is_user: Kullanıcı mesajı mı (True) yoksa AI cevabı mı (False)

        Returns:
            Bulunan değişkenler
        """
        found = {}
        # Önce Türkçe sayı kelimelerini rakama çevir
        text_converted = turkish_word_to_number(text)
        text_lower = text_converted.lower()

        for pattern, var_name, unit in self.patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Son eşleşmeyi al (genellikle en güncel değer)
                value = matches[-1]
                # Virgülü noktaya çevir
                if isinstance(value, str):
                    value = value.replace(',', '.')
                try:
                    value = float(value)
                    # Tam sayıysa int'e çevir
                    if value == int(value):
                        value = int(value)
                    found[var_name] = {
                        'value': value,
                        'unit': unit,
                        'source': 'user' if is_user else 'ai',
                        'timestamp': datetime.now().isoformat()
                    }
                except ValueError:
                    pass

        # Özel hesaplama tespiti: "X x Y = Z" formatı
        calc_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:x|×|\*)\s*(\d+(?:[.,]\d+)?)\s*(?:=|eder)\s*(\d+(?:[.,]\d+)?)'
        calc_matches = re.findall(calc_pattern, text_lower)
        if calc_matches:
            for match in calc_matches:
                try:
                    a, b, result = [float(x.replace(',', '.')) for x in match]
                    # Hacim hesaplaması mı kontrol et (alan x yükseklik)
                    if 'alan' in self.variables and abs(a - self.variables['alan']['value']) < 1:
                        found['hacim'] = {
                            'value': int(result) if result == int(result) else result,
                            'unit': 'm³',
                            'source': 'calculated',
                            'formula': f"{int(a)} x {b} = {int(result)}",
                            'timestamp': datetime.now().isoformat()
                        }
                except ValueError:
                    pass

        return found

    def update(self, text: str, is_user: bool = True):
        """
        Mesajdan değişkenleri çıkar ve güncelle.
        """
        found = self.extract_from_text(text, is_user)

        for var_name, var_data in found.items():
            old_value = self.variables.get(var_name)
            self.variables[var_name] = var_data

            # Geçmişe ekle
            self.history.append({
                'variable': var_name,
                'old_value': old_value,
                'new_value': var_data,
                'timestamp': datetime.now().isoformat()
            })

            print(f"   [CALC] Degisken yakalandi: {var_name} = {var_data['value']} {var_data['unit']}")

    def get_prompt_section(self) -> str:
        """
        Prompt'a eklenecek değişkenler bölümünü oluştur.
        """
        if not self.variables:
            return ""

        lines = ["📊 HESAPLAMA DEĞİŞKENLERİ (Bu değerleri kullan!):"]

        # Değişkenleri sırala
        order = ['alan', 'yukseklik', 'hacim', 'raf_sayisi', 'verim_orani', 'yogunluk', 'agirlik', 'agirlik_ton']

        for var_name in order:
            if var_name in self.variables:
                var = self.variables[var_name]
                value = var['value']
                unit = var['unit']

                # İnsan-okunabilir isimler
                display_names = {
                    'alan': 'Alan',
                    'yukseklik': 'Yükseklik',
                    'hacim': 'Hacim',
                    'raf_sayisi': 'Raf Sayısı',
                    'verim_orani': 'Verim Oranı',
                    'yogunluk': 'Yoğunluk',
                    'agirlik': 'Ağırlık',
                    'agirlik_ton': 'Ağırlık',
                }

                display_name = display_names.get(var_name, var_name)

                if var_name == 'verim_orani':
                    lines.append(f"• {display_name}: %{value}")
                elif var_name == 'agirlik_ton':
                    lines.append(f"• {display_name}: {value} ton ({value * 1000} kg)")
                else:
                    lines.append(f"• {display_name}: {value} {unit}")

        # Kalan değişkenler
        for var_name, var in self.variables.items():
            if var_name not in order:
                lines.append(f"• {var_name}: {var['value']} {var['unit']}")

        if len(lines) > 1:
            lines.append("")
            lines.append("⚠️ Hesaplamalarda bu değerleri MUTLAKA kullan!")
            return "\n".join(lines)

        return ""

    def clear(self):
        """Tüm değişkenleri temizle."""
        self.variables = {}
        self.history = []

    def get_variable(self, name: str) -> Optional[Any]:
        """Belirli bir değişkeni getir."""
        if name in self.variables:
            return self.variables[name]['value']
        return None

    def set_variable(self, name: str, value: Any, unit: str = ''):
        """Manuel değişken ayarla."""
        self.variables[name] = {
            'value': value,
            'unit': unit,
            'source': 'manual',
            'timestamp': datetime.now().isoformat()
        }


# Test
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    ctx = CalculationContext()

    # Test mesajları
    test_messages = [
        ("Yüz metre kare kapalı alanda mantar üretimi", True),
        ("Yükseklik 2,5 metre zemin yüz metre kare", True),
        ("100 metrekare x 2.5 metre = 250 metreküp", False),
        ("3 katlı raf sistemi ile", True),
        ("%25 verim oranı ile hesaplayalım", False),
        ("10.5 ton kompost", False),
    ]

    print("=" * 50)
    print("CALCULATION CONTEXT TEST")
    print("=" * 50)

    for msg, is_user in test_messages:
        role = "USER" if is_user else "AI"
        print(f"\n[{role}]: {msg}")
        ctx.update(msg, is_user)

    print("\n" + "=" * 50)
    print("PROMPT SECTION:")
    print("=" * 50)
    print(ctx.get_prompt_section())
