"""
QuantumTree AI - PyQt5 Desktop Chat
Renkli emoji destekli modern arayüz
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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame,
    QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import asyncio

PROCESS_TIMEOUT = 120


class AIWrapper:
    """AI Wrapper - HafizaAsistani + LocalLLM"""
    def __init__(self, user_id="desktop_user"):
        self.user_id = user_id
        self.mode = "basit"

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
        mode_map = {"Sohbet": "basit", "Derin Analiz": "derin"}
        self.mode = mode_map.get(mode_name, "basit")
        print(f"AI modu değişti: {self.mode}")

    async def process(self, user_input: str) -> str:
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
        if hasattr(self.hafiza, 'hafiza'):
            self.hafiza.hafiza = []


class AIWorker(QThread):
    """Arka plan AI işlemi"""
    finished = pyqtSignal(str)

    def __init__(self, ai, user_input):
        super().__init__()
        self.ai = ai
        self.user_input = user_input

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.ai.process(self.user_input))
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Hata: {str(e)[:100]}")
        finally:
            loop.close()


class MessageBubble(QFrame):
    """Mesaj baloncuğu"""
    def __init__(self, text, is_user=False, animate=False, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self.label = QLabel("" if animate else text)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Segoe UI Emoji", 11))
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.label.setMaximumWidth(600)

        if is_user:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: #1f6feb;
                    color: white;
                    border-radius: 15px;
                    padding: 12px 16px;
                }
            """)
            layout.addStretch()
            layout.addWidget(self.label)
        else:
            self.label.setStyleSheet("""
                QLabel {
                    background-color: #333333;
                    color: white;
                    border-radius: 15px;
                    padding: 12px 16px;
                }
            """)
            layout.addWidget(self.label)
            layout.addStretch()

        self.setStyleSheet("background: transparent;")

        # Typewriter animasyonu
        if animate and text:
            self.full_text = text
            self.current_index = 0
            self.timer = QTimer()
            self.timer.timeout.connect(self.animate_text)
            self.timer.start(5)  # 5ms aralık

    def animate_text(self):
        """Yazıyı karakter karakter göster"""
        if self.current_index <= len(self.full_text):
            self.label.setText(self.full_text[:self.current_index])
            self.current_index += 1
        else:
            self.timer.stop()


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantumTree AI - PyQt5")
        self.setGeometry(100, 100, 1100, 750)
        self.setMinimumSize(800, 600)

        # Dark tema
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
                color: white;
            }
            QPushButton {
                background-color: #1f6feb;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #388bfd;
            }
            QPushButton:disabled {
                background-color: #404040;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
                color: white;
            }
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1a1a1a;
            }
        """)

        # AI
        self.ai = None
        self.is_processing = False
        self.chat_history = []
        self.worker = None

        self.setup_ui()
        self.init_ai()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #2d2d2d;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)

        # Logo
        logo = QLabel("🌳 QuantumTree")
        logo.setFont(QFont("Segoe UI Emoji", 18, QFont.Bold))
        logo.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo)

        sidebar_layout.addSpacing(20)

        # Yeni Sohbet butonu
        new_chat_btn = QPushButton("+ Yeni Sohbet")
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #238636;
            }
            QPushButton:hover {
                background-color: #2ea043;
            }
        """)
        new_chat_btn.clicked.connect(self.reset_chat)
        sidebar_layout.addWidget(new_chat_btn)

        sidebar_layout.addSpacing(10)

        # Kopyala butonu
        copy_btn = QPushButton("Sohbeti Kopyala")
        copy_btn.clicked.connect(self.copy_chat)
        sidebar_layout.addWidget(copy_btn)

        sidebar_layout.addSpacing(20)

        # Mod seçici
        mode_label = QLabel("Yapay Zeka Modu:")
        sidebar_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Sohbet", "Derin Analiz"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        sidebar_layout.addWidget(self.mode_combo)

        sidebar_layout.addStretch()

        # Durum
        self.status_label = QLabel("Durum: Başlatılıyor...")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        sidebar_layout.addWidget(self.status_label)

        main_layout.addWidget(sidebar)

        # Chat alanı
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(20, 20, 20, 20)

        # Mesaj scroll alanı
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.messages_widget = QWidget()
        self.messages_widget.setStyleSheet("background-color: #1a1a1a;")
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(5)

        self.scroll_area.setWidget(self.messages_widget)
        self.scroll_area.viewport().setStyleSheet("background-color: #1a1a1a;")
        chat_layout.addWidget(self.scroll_area)

        # Giriş alanı
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Bir şeyler yazın...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Gönder")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        chat_layout.addLayout(input_layout)

        main_layout.addWidget(chat_container)

        # Başlangıç mesajı
        self.add_message("Merhaba! Size bugün nasıl yardımcı olabilirim? 😊", is_user=False)

    def init_ai(self):
        if AI_AVAILABLE:
            try:
                self.ai = AIWrapper(user_id="desktop_user")
                self.status_label.setText("Durum: Hazır ✅")
                self.status_label.setStyleSheet("color: #2ea043; font-size: 11px;")
            except Exception as e:
                print(f"AI başlatma hatası: {e}")
                self.status_label.setText("Durum: AI Hatası ❌")
                self.status_label.setStyleSheet("color: #f85149; font-size: 11px;")
        else:
            self.status_label.setText("Durum: AI Yok ❌")
            self.status_label.setStyleSheet("color: #f85149; font-size: 11px;")

    def add_message(self, text, is_user=False, animate=False):
        # AI mesajları animasyonlu, kullanıcı mesajları direkt
        if animate is False and not is_user:
            animate = True

        bubble = MessageBubble(text, is_user, animate)
        self.messages_layout.addWidget(bubble)

        # Otomatik scroll (animasyon için tekrarlı)
        def scroll_down():
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )

        QTimer.singleShot(50, scroll_down)
        if animate:
            # Animasyon süresince scroll'u güncelle
            for i in range(0, len(text) * 10, 100):
                QTimer.singleShot(i, scroll_down)

        role = "Kullanıcı" if is_user else "AI"
        self.chat_history.append(f"{role}: {text}")

    def send_message(self):
        msg = self.input_field.text().strip()
        if not msg or self.is_processing:
            return

        self.input_field.clear()
        self.add_message(msg, is_user=True, animate=False)

        if not self.ai:
            self.add_message("AI sistemi başlatılamadı.", is_user=False)
            return

        self.is_processing = True
        self.send_btn.setEnabled(False)

        # Düşünüyor baloncuğu
        self.show_thinking()

        # Arka plan işlemi
        self.worker = AIWorker(self.ai, msg)
        self.worker.finished.connect(self.on_ai_response)
        self.worker.start()

    def show_thinking(self):
        """Düşünüyor animasyonu göster"""
        self.thinking_bubble = MessageBubble("●", is_user=False, animate=False)
        self.messages_layout.addWidget(self.thinking_bubble)
        self.thinking_dots = 0

        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.animate_thinking)
        self.thinking_timer.start(400)

        # Scroll
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def animate_thinking(self):
        """Düşünüyor animasyonu (● ●● ●●●)"""
        if not self.is_processing:
            return

        dots = ["●", "● ●", "● ● ●"]
        self.thinking_bubble.label.setText(dots[self.thinking_dots % 3])
        self.thinking_dots += 1

    def hide_thinking(self):
        """Düşünüyor baloncuğunu kaldır"""
        if hasattr(self, 'thinking_timer'):
            self.thinking_timer.stop()
        if hasattr(self, 'thinking_bubble') and self.thinking_bubble:
            self.thinking_bubble.deleteLater()
            self.thinking_bubble = None

    def on_ai_response(self, response):
        self.hide_thinking()
        self.add_message(response, is_user=False, animate=True)
        self.is_processing = False
        self.send_btn.setEnabled(True)
        self.status_label.setText("Durum: Hazır ✅")
        self.status_label.setStyleSheet("color: #2ea043; font-size: 11px;")

    def on_mode_change(self, mode):
        if self.ai:
            self.ai.set_mode(mode)
            self.add_message(f"Mod değiştirildi: {mode}", is_user=False)

    def reset_chat(self):
        # Mesajları temizle
        while self.messages_layout.count():
            child = self.messages_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.chat_history = []

        if self.ai:
            self.ai.reset()

        self.add_message("Sohbet sıfırlandı. Yeni bir başlangıç yapalım! 🚀", is_user=False)

    def copy_chat(self):
        if not self.chat_history:
            self.status_label.setText("Kopyalanacak mesaj yok")
            return

        chat_text = "\n\n".join(self.chat_history)
        QApplication.clipboard().setText(chat_text)
        self.status_label.setText("Sohbet kopyalandı! 📋")
        QTimer.singleShot(2000, lambda: self.status_label.setText("Durum: Hazır ✅"))


if __name__ == "__main__":
    print("=" * 50)
    print("QuantumTree PyQt5 başlatılıyor...")
    print("=" * 50)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ChatWindow()
    window.show()

    sys.exit(app.exec_())
