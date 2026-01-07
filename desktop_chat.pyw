"""
QuantumTree AI - NextGen Desktop Chat
CustomTkinter ile modern arayüz
"""
import sys
import os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

# ==================== AI IMPORT ====================
AI_AVAILABLE = False
try:
    from personal_ai import LocalLLM
    from hafiza_asistani import HafizaAsistani
    AI_AVAILABLE = True
    print("AI modülleri yüklendi (HafizaAsistani + LocalLLM)")
except Exception as e:
    print(f"AI yüklenemedi: {e}")
    LocalLLM = None
    HafizaAsistani = None

# ==================== GUI ====================
import tkinter as tk
import customtkinter as ctk
import threading
import asyncio
from datetime import datetime
from typing import Optional

# Görünüm Ayarları
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PROCESS_TIMEOUT = 120

class AIWrapper:
    """AI Wrapper - HafizaAsistani + LocalLLM"""
    def __init__(self, user_id="desktop_user"):
        self.user_id = user_id
        self.mode = "basit"  # varsayılan

        # HafizaAsistani + LLM
        self.llm = LocalLLM(user_id)
        self.hafiza = HafizaAsistani(
            saat_limiti=48,
            esik=0.50,
            max_mesaj=20,
            model_adi="BAAI/bge-m3",
            use_decision_llm=True,
            decision_model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        )
        self.hafiza.set_llm(self.llm)
        print(f"AI Wrapper başlatıldı (user: {user_id})")

    def set_mode(self, mode_name: str):
        """Yapay zeka modunu değiştir"""
        mode_map = {
            "Sohbet": "basit",
            "Derin Analiz": "derin"
        }
        self.mode = mode_map.get(mode_name, "basit")
        print(f"AI modu değişti: {self.mode}")

    async def process(self, user_input: str) -> str:
        """Kullanıcı girdisini işle ve yanıt döndür"""
        try:
            response = await asyncio.wait_for(
                self.hafiza.process(user_input, []),
                timeout=PROCESS_TIMEOUT
            )
            return response.strip() if response else "Yanıt alınamadı."
        except asyncio.TimeoutError:
            return "Yanıt süresi aşıldı, tekrar dener misin?"
        except Exception as e:
            print(f"Process hatası: {e}")
            return f"Bir hata oluştu: {str(e)[:100]}"

    def reset(self):
        """Konuşmayı sıfırla"""
        if hasattr(self.hafiza, 'hafiza'):
            self.hafiza.hafiza = []

    def close(self):
        """Kaynakları temizle"""
        pass


class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere Ayarları
        self.title("QuantumTree AI - NextGen")
        self.geometry("1100x750")
        self.minsize(800, 600)

        # İkon (varsa)
        try:
            self.iconbitmap("C:/Projects/quantumtree/tree_icon.ico")
        except: pass

        # Grid düzeni
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # AI Wrapper
        self.ai: Optional[AIWrapper] = None
        self.is_processing = False
        self.chat_history = []  # Mesajları tut (kopyalama için)

        self.setup_sidebar()
        self.setup_chat_area()
        self.init_ai()

        # Kapatma eventi
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_ai(self):
        """AI sistemini başlat"""
        if AI_AVAILABLE:
            try:
                self.ai = AIWrapper(user_id="desktop_user")
                self.status_label.configure(text="Durum: Hazır", text_color="green")
            except Exception as e:
                print(f"AI başlatma hatası: {e}")
                self.status_label.configure(text="Durum: AI Hatası", text_color="red")
        else:
            self.status_label.configure(text="Durum: AI Yok", text_color="red")

    def setup_sidebar(self):
        """Sol sidebar'ı oluştur"""
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        # Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🌳 QuantumTree",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Yeni Sohbet butonu
        self.new_chat_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="+ Yeni Sohbet",
            command=self.reset_chat,
            fg_color="#238636",
            hover_color="#2ea043"
        )
        self.new_chat_btn.grid(row=1, column=0, padx=20, pady=10)

        # Sohbeti Kopyala butonu
        self.copy_chat_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="Sohbeti Kopyala",
            command=self.copy_chat,
            fg_color="#1f6feb",
            hover_color="#388bfd"
        )
        self.copy_chat_btn.grid(row=2, column=0, padx=20, pady=5)

        # Mod seçici
        self.mode_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Yapay Zeka Modu:",
            anchor="w"
        )
        self.mode_label.grid(row=3, column=0, padx=20, pady=(20, 0))

        self.mode_option = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Sohbet", "Derin Analiz"],
            command=self.on_mode_change
        )
        self.mode_option.grid(row=4, column=0, padx=20, pady=10)

        # Tema değiştirici
        self.theme_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Tema:",
            anchor="w"
        )
        self.theme_label.grid(row=5, column=0, padx=20, pady=(20, 0))

        self.theme_option = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Dark", "Light", "System"],
            command=self.on_theme_change
        )
        self.theme_option.set("Dark")
        self.theme_option.grid(row=6, column=0, padx=20, pady=10, sticky="n")

        # Durum etiketi (altta)
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Durum: Başlatılıyor...",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.grid(row=7, column=0, pady=20)

    def setup_chat_area(self):
        """Sağ chat alanını oluştur"""
        self.main_chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_chat_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_chat_frame.grid_rowconfigure(0, weight=1)
        self.main_chat_frame.grid_columnconfigure(0, weight=1)

        # Mesaj alanı (Scrollable)
        self.chat_display = ctk.CTkScrollableFrame(
            self.main_chat_frame,
            fg_color="#1a1a1a",
            corner_radius=15
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 20))

        # Giriş alanı
        self.input_container = ctk.CTkFrame(self.main_chat_frame, fg_color="transparent")
        self.input_container.grid(row=1, column=0, sticky="ew")
        self.input_container.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Bir şeyler yazın...",
            height=50,
            font=ctk.CTkFont(size=14)
        )
        self.entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(
            self.input_container,
            text="Gönder",
            width=100,
            height=50,
            command=self.send_message,
            font=ctk.CTkFont(weight="bold")
        )
        self.send_button.grid(row=0, column=1)

        # Başlangıç mesajı
        self.add_bubble("Merhaba! Size bugün nasıl yardımcı olabilirim?", "ai")

    def add_bubble(self, text: str, sender: str, save_history: bool = True, animate: bool = None):
        """Mesaj baloncuğu ekle"""
        color = "#1f6feb" if sender == "user" else "#333333"
        txt_color = "white"

        # AI mesajları animasyonlu, kullanıcı mesajları direkt
        if animate is None:
            animate = (sender == "ai")

        bubble_frame = ctk.CTkFrame(self.chat_display, fg_color="transparent")
        bubble_frame.pack(fill="x", padx=10, pady=5)

        bubble = ctk.CTkLabel(
            bubble_frame,
            text="" if animate else text,
            fg_color=color,
            text_color=txt_color,
            corner_radius=15,
            padx=15,
            pady=10,
            wraplength=600,
            justify="left",
            font=ctk.CTkFont(size=14)
        )
        bubble.pack(side="right" if sender == "user" else "left")

        # Mesajı history'ye ekle
        if save_history:
            role = "Kullanıcı" if sender == "user" else "AI"
            self.chat_history.append(f"{role}: {text}")

        # AI mesajları için typing animasyonu
        if animate:
            self.animate_text(bubble, text, 0)
        else:
            # Otomatik scroll aşağı
            self.after(50, lambda: self.chat_display._parent_canvas.yview_moveto(1.0))

    def animate_text(self, label, full_text: str, index: int):
        """Yazı karakterleri tek tek göster (typewriter efekti)"""
        if index <= len(full_text):
            label.configure(text=full_text[:index])
            # Scroll aşağı
            self.chat_display._parent_canvas.yview_moveto(1.0)
            # Sonraki karakter (5ms aralık - çok hızlı)
            self.after(5, lambda: self.animate_text(label, full_text, index + 1))

    def send_message(self):
        """Mesaj gönder"""
        msg = self.entry.get().strip()
        if not msg or self.is_processing:
            return

        self.entry.delete(0, 'end')
        self.add_bubble(msg, "user")

        if not self.ai:
            self.add_bubble("AI sistemi başlatılamadı.", "ai")
            return

        self.is_processing = True
        self.send_button.configure(state="disabled")

        # Düşünüyor animasyonu başlat
        self.show_thinking_bubble()

        # AI yanıtını ayrı thread'de al
        threading.Thread(target=self.process_ai_response, args=(msg,), daemon=True).start()

    def show_thinking_bubble(self):
        """Düşünüyor baloncuğu göster"""
        self.thinking_frame = ctk.CTkFrame(self.chat_display, fg_color="transparent")
        self.thinking_frame.pack(fill="x", padx=10, pady=5)

        self.thinking_bubble = ctk.CTkLabel(
            self.thinking_frame,
            text="●",
            fg_color="#333333",
            text_color="#888888",
            corner_radius=15,
            padx=20,
            pady=10,
            font=ctk.CTkFont(size=16)
        )
        self.thinking_bubble.pack(side="left")
        self.thinking_dots = 0
        self.animate_thinking()

    def animate_thinking(self):
        """Düşünüyor animasyonu (● ●● ●●●)"""
        if not self.is_processing:
            return

        dots = ["●", "● ●", "● ● ●"]
        self.thinking_bubble.configure(text=dots[self.thinking_dots % 3])
        self.thinking_dots += 1
        self.chat_display._parent_canvas.yview_moveto(1.0)
        self.after(400, self.animate_thinking)

    def hide_thinking_bubble(self):
        """Düşünüyor baloncuğunu kaldır"""
        if hasattr(self, 'thinking_frame') and self.thinking_frame:
            self.thinking_frame.destroy()
            self.thinking_frame = None

    def process_ai_response(self, user_msg: str):
        """AI yanıtını işle (thread içinde çalışır)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.ai.process(user_msg))
            if not response or len(response) < 2:
                response = "Bir sorun oluştu, lütfen tekrar deneyin."
            self.after(0, lambda: self.finish_response(response))
        except Exception as e:
            self.after(0, lambda: self.finish_response(f"Hata: {str(e)[:100]}"))
        finally:
            loop.close()

    def finish_response(self, response: str):
        """AI yanıtını göster ve durumu güncelle"""
        self.hide_thinking_bubble()
        self.add_bubble(response, "ai")
        self.is_processing = False
        self.send_button.configure(state="normal")

    def on_mode_change(self, mode: str):
        """AI modunu değiştir"""
        if self.ai:
            self.ai.set_mode(mode)
            self.add_bubble(f"Mod değiştirildi: {mode}", "ai")

    def on_theme_change(self, theme: str):
        """Tema değiştir"""
        ctk.set_appearance_mode(theme)

    def copy_chat(self):
        """Tüm sohbeti panoya kopyala"""
        if not self.chat_history:
            self.status_label.configure(text="Kopyalanacak mesaj yok", text_color="orange")
            self.after(2000, lambda: self.status_label.configure(text="Durum: Hazır", text_color="green"))
            return

        chat_text = "\n\n".join(self.chat_history)
        self.clipboard_clear()
        self.clipboard_append(chat_text)
        self.status_label.configure(text="Sohbet kopyalandı!", text_color="#58a6ff")
        self.after(2000, lambda: self.status_label.configure(text="Durum: Hazır", text_color="green"))

    def reset_chat(self):
        """Sohbeti sıfırla"""
        for widget in self.chat_display.winfo_children():
            widget.destroy()

        self.chat_history = []  # Mesaj geçmişini temizle

        if self.ai:
            self.ai.reset()

        self.add_bubble("Sohbet sıfırlandı. Yeni bir başlangıç yapalım!", "ai")

    def on_close(self):
        """Uygulama kapatılırken"""
        if self.ai:
            try:
                self.ai.close()
            except: pass
        self.destroy()


if __name__ == "__main__":
    print("=" * 50)
    print("QuantumTree NextGen başlatılıyor...")
    print("=" * 50)
    app = ChatApp()
    app.mainloop()
