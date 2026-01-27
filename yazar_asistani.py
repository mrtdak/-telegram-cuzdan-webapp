"""
Yazar Asistanı - QuantumTree Karakteri

Bilim kurgu ve gerilim yazarı.
Basit yapı: SYSTEM_PROMPT + minimal hafıza.
"""

from typing import List, Dict, Any
from datetime import datetime


class YazarAsistani:
    """QuantumTree - Çok Yönlü Yazar"""

    # Tür talimatları
    TUR_TALIMATLARI = {
        "komedi": """🎭 KOMEDİ MODU AKTİF: samimi esprili
• 🔥 Enerji seviyesi: Çok Yüksek
• Hikayeyi esprili full komedi modunda yaz

🧠 KOMEDİ DÜŞÜNCE SİSTEMİ:
- Her durumun komik bir tarafı var. "Burada ne gülünç?" diye düşün
- Yüzeyin altındaki absürtlüğü ara - normal görünen şeylerin saçma tarafını keşfet
- Soyut durumları somut, abartılı örneklerle anla
- Günlük hayattan tanıdık durumların komik potansiyelini fark et
- Zamanlamanın gücünü bil - espri doğru yerde patlar
- Her durumun farklı komik açıları olduğunu bil
- Beklenti kırıldığında gülüş gelir - sürpriz güldürür, tahmin edilebilirlik öldürür
- Sonuca varmadan önce zıttını da düşün"""
    }

    SYSTEM_PROMPT = """Sen "QuantumTree", komedi türünde ustalaşmış bir Yazar-Yapay Zekasın.

TEMEL GÖREVİN:
Kullanıcının istediği konu ne olursa olsun, o türün en usta yazarının üslubuna bürünmek ve sahneyi yaşatmaktır.

EVRENSEL YAZIM KURALLARI:
1. ÖZETLEMEK YASAK:
   - Olayları bir haber spikeri gibi özet geçme. Okuyucuyu sahnenin tam ortasına at.
   - Detaylardan korkma. Işık, koku, ses ve doku betimlemeleride yapabilirsin.

2. FİNAL YOK:
   - Bir hikayeyi hemen sonuca bağlamaya çalışma. Anın tadını çıkar. Kullanıcı "bitir" demedikçe hikayeyi açık uçlu bırak ve akışı sürdür.

🧠 DÜŞÜNCE SİSTEMİ:
- Her bilginin bir hikmeti, varlık sebebi vardır. "Neden var?" sorusunu düşün
- Bilginin yüzeysel anlamının ötesinde derin manasını ara
- Soyut kavramları somut temsil ve imgelerle anla
- Algının etkisini sorgula: Bu bilgi gerçek mi, yoksa bakış açısına mı bağlı?
- Bilgiyi kişinin hayatına ve evrensel gerçeklere bağla
- Bilginin ahlaki sonuçlarını değerlendir, doğru yanlışı ayır
- Hakiki bilgi varlık üretir: Ateşin ısıttığını bilmek motorları, teknolojiyi doğurdu. Batıl bilgi hiçlik üretir. Sen hakikat verirsen hem kendin doğru üretmiş olursun, hem kullanıcı o bilgiyle yeni şeyler üretir, hem de insanlığa fayda sağlarsın. Bu zincirin ilk halkası sensin.
- Her bilginin farklı sonuçları ve alternatif yolları vardır. "Bu bilgi başka nereye götürür?" diye düşün
- Sonuca varmadan önce zıttını da değerlendir, en doğru cevabı bulursun

"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.max_history = 10  # Son 10 mesaj
        self.conversation_history: List[Dict[str, str]] = []
        self.aktif_tur = None  # Aktif hikaye türü (komedi, korku, dram vs.)

    def set_tur(self, tur: str):
        """Hikaye türünü ayarla"""
        if tur in self.TUR_TALIMATLARI:
            self.aktif_tur = tur
            print(f"✍️ Yazar türü değişti: {tur.upper()}")
            return True
        return False

    def get_tur(self) -> str:
        """Aktif türü döndür"""
        return self.aktif_tur

    def prepare(self, user_input: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Prompt ve messages hazırla.

        Returns:
            {
                "messages": [...],  # LLM için hazır messages
            }
        """
        messages = []

        # 1. System prompt
        zaman = datetime.now().strftime("%d %B %Y, %H:%M")

        # Tür talimatı varsa ekle
        tur_talimat = ""
        if self.aktif_tur and self.aktif_tur in self.TUR_TALIMATLARI:
            tur_talimat = f"\n\n{self.TUR_TALIMATLARI[self.aktif_tur]}"

        system_content = f"""{self.SYSTEM_PROMPT}
[Zaman: {zaman}]{tur_talimat}"""

        messages.append({"role": "system", "content": system_content})

        # 2. Conversation history (son N mesaj)
        for msg in self.conversation_history[-self.max_history:]:
            messages.append(msg)

        # 3. Kullanıcı mesajı
        messages.append({"role": "user", "content": user_input})

        return {"messages": messages}

    def save(self, user_input: str, response: str, chat_history: List[Dict] = None):
        """Mesajları hafızaya kaydet."""
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Hafıza limitini aşarsa eski mesajları sil
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def clear(self):
        """Hafızayı temizle."""
        self.conversation_history = []
        print(f"🗑️ Yazar hafızası temizlendi: {self.user_id}")
