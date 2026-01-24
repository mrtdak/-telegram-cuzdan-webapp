# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal AI assistant system with Telegram bot interface. Turkish language focused conversational AI with multi-layer memory, web search, and creative writing capabilities.

## Architecture

```
telegram_bot.py          # Entry point - Telegram interface
    ↓
HafizaAsistani          # Prompt preparation + memory management
    ↓
PersonalAI              # LLM response generation (OpenRouter/Together/Ollama)
```

### Core Components

- **telegram_bot.py**: Telegram bot with user isolation, multiple modes (normal/yazar/komedi)
- **personal_ai.py**: LLM wrapper supporting Together.ai (Gemma 3 27B), OpenRouter, and Ollama
- **hafiza_asistani.py**: Memory assistant - prepares prompts with context from multiple sources
- **sohbet_zekasi.py**: Turkish conversation intelligence - pattern matching for Turkish chat flow
- **conversation_context.py**: LLM-based topic continuation detection and session management
- **topic_memory.py**: Long-term memory with semantic similarity grouping (FAISS + embeddings)
- **profile_manager.py**: Per-user profile storage (name, interests, facts)
- **web_search.py**: Tavily API integration with Turkish→English auto-translation
- **yazar_asistani.py**: Creative writing mode (QuantumTree character)

### Data Flow

1. User message → Telegram
2. `HafizaAsistani.prepare()` → Builds prompt with:
   - Turkish conversation analysis (sohbet_zekasi)
   - Profile context
   - Topic memory context
   - Web search results (if needed)
   - Chat history
3. `PersonalAI.generate()` → LLM call
4. `HafizaAsistani.save()` → Updates memories
5. Response → Telegram

### Memory Layers

1. **Short-term**: Chat history (in-memory, per-session)
2. **Conversation Context**: LLM-based topic tracking with auto-archiving
3. **Topic Memory**: Long-term semantic memory (FAISS index + category files)
4. **Profile**: Persistent user facts and preferences

## Running the Bot

```bash
# Set environment variables
export TELEGRAM_TOKEN=your_token
export TOGETHER_API_KEY=your_key      # PersonalAI (Gemma 3 27B)
export OPENROUTER_API_KEY=your_key    # DecisionLLM (Llama 405B)
export TAVILY_API_KEY=your_key        # Web search

# Run
python telegram_bot.py
```

## Key Configuration (personal_ai.py)

- `LLM_PROVIDER`: "openrouter" (default) | "together" | "ollama"
- `OPENROUTER_MODEL`: "google/gemma-3-27b-it" (Gemma 3 27B)
- `TEMPERATURE`: 0.70
- `MAX_TOKENS`: 4000

## DecisionLLM (hafiza_asistani.py)

- **Provider**: Together.ai
- **Model**: "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" (Llama 405B)
- **Purpose**: Smart decision making for tool selection and context gathering

## User Data Structure

```
user_data/user_{id}/
├── profile.json              # User profile
├── notes/
│   └── notlar.json           # User notes
├── conversation_context/     # Session tracking
│   ├── current_session.json
│   └── archive/
└── topic_memory/            # Long-term memory
    ├── topics_index.json
    └── categories/
```

## Note System

`NotManager` class in `hafiza_asistani.py` handles user notes with Turkish triggers.

**Triggers:**
```
# Save note
"not al: yarın toplantı var"
"not tut: market alışverişi"
"not ekle: doktora git"

# List notes
"notlarım"
"notlarıma bak"

# Delete note
"not sil #1"
"1 numaralı notu sil"
```

Notes are stored in `user_data/user_{id}/notes/notlar.json` with id, content, date, day, time.

## Location Services (Konum Hizmetleri)

GPS-based location services integrated with Telegram location sharing.

**Triggers:**
```
# Share location
"konum gönder" / "konum paylaş" → Location button appears

# After sharing location:
"hava nasıl?" → Weather (wttr.in API)
"namaz vakitleri" → Prayer times (Aladhan API)
"kıble nerede?" → Qibla direction + distance to Kaaba
"yakında eczane var mı?" → Nearby places (OpenStreetMap Overpass API)
"1 numaranın konumunu gönder" → Telegram location message
"1" veya "2" (sadece sayı) → Listeden konum gönder
```

**Nearby Places Categories:**
- benzinlik/akaryakıt (⛽)
- eczane (💊)
- restoran/lokanta (🍽️)
- kafe/kahve (☕)
- atm/bankamatik (🏧)
- hastane/acil (🏥)
- cami/mescit (🕌)
- market/süpermarket (🛒)

**Fuzzy Matching:** Yazım hataları otomatik düzeltilir (örn: "ezhane" → "eczane", "benznilik" → "benzinlik")

**Key Methods in `hafiza_asistani.py`:**
- `set_location(lat, lon, adres)` - Store user location
- `prepare_konum_alindi()` - LLM prompt when location received
- `_check_konum_sorgusu()` - Pattern matching for location queries
- `_get_yakin_yerler()` - OpenStreetMap nearby search
- `get_yakin_yer_konumu()` - Get coordinates for Telegram location message

**Data stored:** `self.user_location`, `self.user_location_adres`, `self.son_yakin_yerler`

## Turkish Language Handling

`sohbet_zekasi.py` provides pattern-based analysis for Turkish conversational patterns:
- Greeting detection (selamlasma)
- Emotion + intent combinations
- Expected response type inference
- Topic change signals

## Bot Commands

- `/start` - Initialize bot
- `/yeni` - Clear memory
- `/konum` - Location services menu
- `/normal` - Normal chat mode
- `/yazar` - Creative writing mode (QuantumTree)
- `/komedi` - Comedy writing mode
- `/firlama` - Toggle energetic mode
- `/kamera` - Kamera izlemeyi başlat (sadece admin)
- `/kamerakapat` - Kamera izlemeyi durdur (sadece admin)

## Ev Güvenlik Kamera Sistemi

**DVR Bilgileri:**
- **Model:** Dahua DH-XVR1A04
- **IP:** 192.168.1.4
- **Port:** 554 (RTSP)
- **Kullanıcı:** admin

**RTSP URL Formatı:**
```
rtsp://admin:SIFRE@192.168.1.4:554/cam/realmonitor?channel=1&subtype=0  # Cam1
rtsp://admin:SIFRE@192.168.1.4:554/cam/realmonitor?channel=2&subtype=0  # Cam2
rtsp://admin:SIFRE@192.168.1.4:554/cam/realmonitor?channel=3&subtype=0  # Cam3
```

**Kamera Dosyaları:**
- `kamera_telegram.py` - YOLO + LLM + Telegram entegrasyonu
- `guvenlik_kamera.py` - Ev/Kuyumcu modları (hareket algılama + AI analiz)
- `kamera_roi.json` - ROI koordinatları (cam1, cam2, cam3 bahçe/giriş alanları)
- `kamera_bildirim.json` - Son bildirim durumu
- `kamera_startup_durum.json` - Kamera başlangıç durumu

**Akış:**
1. YOLO ile insan tespiti
2. LLM (Gemini) ile doğrulama
3. ROI kontrolü (sadece belirlenen alanda)
4. Telegram bildirimi + fotoğraf
