"""
Web Arayüzü - FastAPI + WebSocket
Mevcut HafizaAsistani + PersonalAI sistemini web'e taşır
"""

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

# Mevcut modüller
from hafiza_asistani import HafizaAsistani
from personal_ai import PersonalAI

load_dotenv()

app = FastAPI(title="Asistan Web")

# Kullanıcı session'ları
sessions = {}

def get_or_create_session(session_id: str):
    """Session yoksa oluştur"""
    if session_id not in sessions:
        user_id = f"web_{session_id}"
        asistan = HafizaAsistani(user_id=user_id)
        ai = PersonalAI(user_id=user_id)
        sessions[session_id] = {
            "asistan": asistan,
            "ai": ai,
            "chat_history": [],
            "mode": "normal"
        }
        print(f"[WEB] Yeni session: {session_id}")
    return sessions[session_id]

# Ana sayfa - Chat arayüzü
HTML_PAGE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asistan</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        .header {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .header h1 {
            color: #fff;
            font-size: 1.3rem;
            font-weight: 600;
        }

        .header-buttons {
            display: flex;
            gap: 8px;
        }

        .header-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }

        .header-btn:hover {
            background: rgba(255,255,255,0.2);
        }

        .header-btn.active {
            background: #4CAF50;
            border-color: #4CAF50;
        }

        /* Chat container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        /* Mesaj baloncukları */
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            background: #4CAF50;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }

        .message.assistant {
            background: rgba(255,255,255,0.1);
            color: #e0e0e0;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }

        .message.system {
            background: rgba(255,193,7,0.2);
            color: #ffc107;
            align-self: center;
            font-size: 0.85rem;
            padding: 8px 16px;
        }

        /* Typing indicator */
        .typing {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 18px;
            align-self: flex-start;
            width: fit-content;
        }

        .typing span {
            width: 8px;
            height: 8px;
            background: #888;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-8px); opacity: 1; }
        }

        /* Input area */
        .input-area {
            background: rgba(255,255,255,0.05);
            padding: 15px 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            max-width: 900px;
            margin: 0 auto;
        }

        #messageInput {
            flex: 1;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 24px;
            padding: 12px 20px;
            color: #fff;
            font-size: 1rem;
            outline: none;
            transition: all 0.2s;
        }

        #messageInput:focus {
            border-color: #4CAF50;
            background: rgba(255,255,255,0.15);
        }

        #messageInput::placeholder {
            color: rgba(255,255,255,0.4);
        }

        .send-btn {
            background: #4CAF50;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .send-btn:hover {
            background: #43a047;
            transform: scale(1.05);
        }

        .send-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }

        .send-btn svg {
            width: 24px;
            height: 24px;
            fill: white;
        }

        /* Konum butonu */
        .location-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            font-size: 1.2rem;
        }

        .location-btn:hover {
            background: rgba(255,255,255,0.2);
        }

        /* Responsive */
        @media (max-width: 600px) {
            .message {
                max-width: 90%;
            }

            .header h1 {
                font-size: 1.1rem;
            }

            .header-btn {
                padding: 6px 10px;
                font-size: 0.8rem;
            }
        }

        /* Scrollbar */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: transparent;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Asistan</h1>
        <div class="header-buttons">
            <button class="header-btn active" data-mode="normal">Normal</button>
            <button class="header-btn" data-mode="yazar">Yazar</button>
            <button class="header-btn" data-mode="komedi">Komedi</button>
            <button class="header-btn" onclick="clearChat()">🗑️</button>
        </div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="message system">Merhaba! Seninle sohbet etmeye hazırım.</div>
    </div>

    <div class="input-area">
        <div class="input-wrapper">
            <button class="location-btn" onclick="shareLocation()" title="Konum Paylaş">📍</button>
            <input type="text" id="messageInput" placeholder="Mesajınızı yazın..." autocomplete="off">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
            </button>
        </div>
    </div>

    <script>
        let ws;
        let currentMode = 'normal';
        let isConnected = false;
        let reconnectAttempts = 0;

        // Session ID oluştur
        const sessionId = localStorage.getItem('sessionId') || Math.random().toString(36).substr(2, 9);
        localStorage.setItem('sessionId', sessionId);

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);

            ws.onopen = () => {
                console.log('WebSocket bağlandı');
                isConnected = true;
                reconnectAttempts = 0;
                document.getElementById('sendBtn').disabled = false;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                removeTyping();

                if (data.type === 'message') {
                    addMessage(data.content, 'assistant');
                } else if (data.type === 'error') {
                    addMessage('Hata: ' + data.content, 'system');
                } else if (data.type === 'location_received') {
                    addMessage('📍 Konum alındı: ' + data.address, 'system');
                }
            };

            ws.onclose = () => {
                console.log('WebSocket kapandı');
                isConnected = false;
                document.getElementById('sendBtn').disabled = true;

                // Yeniden bağlan
                if (reconnectAttempts < 5) {
                    reconnectAttempts++;
                    setTimeout(connect, 2000);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket hatası:', error);
            };
        }

        function addMessage(content, type) {
            const container = document.getElementById('chatContainer');
            const msg = document.createElement('div');
            msg.className = `message ${type}`;
            msg.textContent = content;
            container.appendChild(msg);
            container.scrollTop = container.scrollHeight;
        }

        function showTyping() {
            const container = document.getElementById('chatContainer');
            const typing = document.createElement('div');
            typing.className = 'typing';
            typing.id = 'typingIndicator';
            typing.innerHTML = '<span></span><span></span><span></span>';
            container.appendChild(typing);
            container.scrollTop = container.scrollHeight;
        }

        function removeTyping() {
            const typing = document.getElementById('typingIndicator');
            if (typing) typing.remove();
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || !isConnected) return;

            addMessage(message, 'user');
            showTyping();

            ws.send(JSON.stringify({
                type: 'message',
                content: message,
                mode: currentMode
            }));

            input.value = '';
        }

        function shareLocation() {
            if (!navigator.geolocation) {
                addMessage('Tarayıcınız konum özelliğini desteklemiyor.', 'system');
                return;
            }

            addMessage('📍 Konum alınıyor...', 'system');

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const { latitude, longitude } = position.coords;
                    ws.send(JSON.stringify({
                        type: 'location',
                        lat: latitude,
                        lon: longitude
                    }));
                },
                (error) => {
                    addMessage('Konum alınamadı: ' + error.message, 'system');
                },
                { enableHighAccuracy: true }
            );
        }

        function clearChat() {
            const container = document.getElementById('chatContainer');
            container.innerHTML = '<div class="message system">Sohbet temizlendi.</div>';

            ws.send(JSON.stringify({ type: 'clear' }));
        }

        // Mod butonları
        document.querySelectorAll('.header-btn[data-mode]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.header-btn[data-mode]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentMode = btn.dataset.mode;
                addMessage(`Mod değişti: ${currentMode}`, 'system');
            });
        });

        // Enter ile gönder
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Sayfa yüklenince bağlan
        connect();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTML_PAGE

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_or_create_session(session_id)

    print(f"[WEB] WebSocket bağlandı: {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "message":
                content = data.get("content", "")
                mode = data.get("mode", "normal")

                if not content:
                    continue

                try:
                    # Mod ayarla
                    session["asistan"].mode = mode

                    # Hafıza asistanından prompt hazırla
                    result = await session["asistan"].prepare(content, session["chat_history"])
                    messages = result.get("messages", [])
                    paket = result.get("paket", {})

                    # Direct response varsa (not sistemi vs.)
                    if paket.get("direct_response"):
                        response = paket["direct_response"]
                    else:
                        # LLM'den cevap al
                        response = await session["ai"].generate(messages)

                    # Hafızaya kaydet
                    session["asistan"].save(content, response, session["chat_history"])

                    # Chat history güncelle
                    session["chat_history"].append({"role": "user", "content": content})
                    session["chat_history"].append({"role": "assistant", "content": response})

                    # Son 20 mesajı tut
                    if len(session["chat_history"]) > 40:
                        session["chat_history"] = session["chat_history"][-40:]

                    await websocket.send_json({
                        "type": "message",
                        "content": response
                    })

                except Exception as e:
                    print(f"[WEB] Hata: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "content": str(e)
                    })

            elif msg_type == "location":
                lat = data.get("lat")
                lon = data.get("lon")

                if lat and lon:
                    # Reverse geocoding için basit API
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as http_session:
                            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=tr"
                            async with http_session.get(url, headers={"User-Agent": "AsistanWeb/1.0"}) as resp:
                                if resp.status == 200:
                                    geo_data = await resp.json()
                                    address = geo_data.get("display_name", f"{lat}, {lon}")
                                else:
                                    address = f"{lat}, {lon}"
                    except:
                        address = f"{lat}, {lon}"

                    # Asistana konumu kaydet
                    session["asistan"].set_location(lat, lon, address)

                    await websocket.send_json({
                        "type": "location_received",
                        "address": address[:100]
                    })

            elif msg_type == "clear":
                session["chat_history"] = []
                session["asistan"].clear_memory()
                print(f"[WEB] Sohbet temizlendi: {session_id}")

    except WebSocketDisconnect:
        print(f"[WEB] WebSocket kapandı: {session_id}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 Web Asistan Başlatılıyor...")
    print("="*50)
    print("\n📍 Adres: http://localhost:8000")
    print("📱 Mobil: http://<IP_ADRESIN>:8000")
    print("\nÇıkmak için Ctrl+C\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
