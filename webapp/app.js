// Telegram Web App
const tg = window.Telegram.WebApp;

// Tema renkleri
tg.expand();
tg.ready();

// Kullanıcı bilgisi
const user = tg.initDataUnsafe?.user;
const userId = user?.id || 'demo';

// ==================== STATE ====================
let selectedKategori = null;
let silinecekIslemId = null;
let silinecekNotId = null;
let currentLocation = null;

// Cüzdan verileri
let cuzdan = {
    bakiye: 0,
    baslangic_bakiye: 0,
    aylik_gelir: 0,
    aylik_gider: 0,
    islemler: [],
    kategoriler: {}
};

// Not verileri
let notlar = [];

// ==================== SABITLER ====================
const kategoriRenkleri = {
    market: '#22c55e', yemek: '#f59e0b', fatura: '#3b82f6',
    yakit: '#ef4444', ulasim: '#8b5cf6', saglik: '#ec4899',
    giyim: '#06b6d4', eglence: '#f97316', kira: '#6366f1',
    diger_gider: '#94a3b8', maas: '#22c55e', ek_gelir: '#10b981',
    yatirim: '#3b82f6', hediye: '#f59e0b', iade: '#8b5cf6',
    diger_gelir: '#94a3b8'
};

const kategoriEmoji = {
    market: '🛒', yemek: '🍔', fatura: '📄', yakit: '⛽',
    ulasim: '🚌', saglik: '💊', giyim: '👕', eglence: '🎮',
    kira: '🏠', diger_gider: '💸', maas: '💼', ek_gelir: '💵',
    yatirim: '📈', hediye: '🎁', iade: '🔄', diger_gelir: '💰'
};

const kategoriIsim = {
    market: 'Market', yemek: 'Yemek', fatura: 'Fatura', yakit: 'Yakıt',
    ulasim: 'Ulaşım', saglik: 'Sağlık', giyim: 'Giyim', eglence: 'Eğlence',
    kira: 'Kira', diger_gider: 'Diğer', maas: 'Maaş', ek_gelir: 'Ek Gelir',
    yatirim: 'Yatırım', hediye: 'Hediye', iade: 'İade', diger_gelir: 'Diğer'
};

const nearbyTypes = {
    pharmacy: { query: 'amenity=pharmacy', emoji: '💊', name: 'Eczane' },
    fuel: { query: 'amenity=fuel', emoji: '⛽', name: 'Benzinlik' },
    hospital: { query: 'amenity=hospital', emoji: '🏥', name: 'Hastane' },
    mosque: { query: 'amenity=place_of_worship][religion=muslim', emoji: '🕌', name: 'Cami' },
    atm: { query: 'amenity=atm', emoji: '🏧', name: 'ATM' },
    supermarket: { query: 'shop=supermarket', emoji: '🛒', name: 'Market' },
    restaurant: { query: 'amenity=restaurant', emoji: '🍽️', name: 'Restoran' },
    cafe: { query: 'amenity=cafe', emoji: '☕', name: 'Kafe' },
    parking: { query: 'amenity=parking', emoji: '🅿️', name: 'Otopark' },
    police: { query: 'amenity=police', emoji: '👮', name: 'Polis' }
};

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', async () => {
    await loadAllData();
    setupEventListeners();
});

function setupEventListeners() {
    // Kategori butonları
    document.querySelectorAll('.kat-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('.kategori-grid');
            parent.querySelectorAll('.kat-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedKategori = btn.dataset.kat;
        });
    });
}

// ==================== TAB NAVIGATION ====================
function switchTab(tabName) {
    // Tab içeriklerini gizle
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Tab butonlarını güncelle
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Seçili tab'ı göster
    document.getElementById(`tab-${tabName}`).classList.add('active');
    event.currentTarget.classList.add('active');

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.selectionChanged();
    }
}

// ==================== DATA LOADING ====================
async function loadAllData() {
    await Promise.all([
        loadCuzdanData(),
        loadNotlarData(),
        loadKonumData()
    ]);
}

async function loadCuzdanData() {
    try {
        if (tg.CloudStorage) {
            tg.CloudStorage.getItem('cuzdan_veriler', (err, value) => {
                if (!err && value) {
                    cuzdan = JSON.parse(value);
                    hesaplaBakiye();
                }
                updateCuzdanUI();
            });
        } else {
            const saved = localStorage.getItem('cuzdan_veriler');
            if (saved) {
                cuzdan = JSON.parse(saved);
                hesaplaBakiye();
            }
            updateCuzdanUI();
        }
    } catch (e) {
        console.error('Cüzdan yükleme hatası:', e);
        updateCuzdanUI();
    }
}

async function loadNotlarData() {
    try {
        if (tg.CloudStorage) {
            tg.CloudStorage.getItem('notlar_veriler', (err, value) => {
                if (!err && value) {
                    notlar = JSON.parse(value);
                }
                updateNotlarUI();
            });
        } else {
            const saved = localStorage.getItem('notlar_veriler');
            if (saved) {
                notlar = JSON.parse(saved);
            }
            updateNotlarUI();
        }
    } catch (e) {
        console.error('Notlar yükleme hatası:', e);
        updateNotlarUI();
    }
}

async function loadKonumData() {
    try {
        if (tg.CloudStorage) {
            tg.CloudStorage.getItem('konum_veriler', (err, value) => {
                if (!err && value) {
                    currentLocation = JSON.parse(value);
                    if (currentLocation) {
                        showLocationInfo();
                    }
                }
            });
        } else {
            const saved = localStorage.getItem('konum_veriler');
            if (saved) {
                currentLocation = JSON.parse(saved);
                if (currentLocation) {
                    showLocationInfo();
                }
            }
        }
    } catch (e) {
        console.error('Konum yükleme hatası:', e);
    }
}

// ==================== DATA SAVING ====================
async function saveCuzdanData() {
    try {
        const dataStr = JSON.stringify(cuzdan);
        if (tg.CloudStorage) {
            tg.CloudStorage.setItem('cuzdan_veriler', dataStr, (err) => {
                if (err) console.error('CloudStorage kayıt hatası:', err);
            });
        } else {
            localStorage.setItem('cuzdan_veriler', dataStr);
        }
    } catch (e) {
        console.error('Cüzdan kaydetme hatası:', e);
    }
}

async function saveNotlarData() {
    try {
        const dataStr = JSON.stringify(notlar);
        if (tg.CloudStorage) {
            tg.CloudStorage.setItem('notlar_veriler', dataStr, (err) => {
                if (err) console.error('CloudStorage kayıt hatası:', err);
            });
        } else {
            localStorage.setItem('notlar_veriler', dataStr);
        }
    } catch (e) {
        console.error('Notlar kaydetme hatası:', e);
    }
}

async function saveKonumData() {
    try {
        const dataStr = JSON.stringify(currentLocation);
        if (tg.CloudStorage) {
            tg.CloudStorage.setItem('konum_veriler', dataStr, (err) => {
                if (err) console.error('CloudStorage kayıt hatası:', err);
            });
        } else {
            localStorage.setItem('konum_veriler', dataStr);
        }
    } catch (e) {
        console.error('Konum kaydetme hatası:', e);
    }
}

// ==================== CÜZDAN FONKSİYONLARI ====================
function hesaplaBakiye() {
    let bakiye = cuzdan.baslangic_bakiye || 0;

    cuzdan.islemler.forEach(islem => {
        if (islem.tip === 'gelir') {
            bakiye += islem.tutar;
        } else {
            bakiye -= islem.tutar;
        }
    });

    cuzdan.bakiye = bakiye;

    // Bu ayın gelir/gider hesapla
    const now = new Date();
    const ayBasi = new Date(now.getFullYear(), now.getMonth(), 1);

    cuzdan.aylik_gelir = 0;
    cuzdan.aylik_gider = 0;
    cuzdan.kategoriler = {};

    cuzdan.islemler.forEach(islem => {
        const islemTarih = new Date(islem.tarih);
        if (islemTarih >= ayBasi) {
            if (islem.tip === 'gelir') {
                cuzdan.aylik_gelir += islem.tutar;
            } else {
                cuzdan.aylik_gider += islem.tutar;
                cuzdan.kategoriler[islem.kategori] = (cuzdan.kategoriler[islem.kategori] || 0) + islem.tutar;
            }
        }
    });
}

function updateCuzdanUI() {
    document.getElementById('bakiye').textContent = formatMoney(cuzdan.bakiye);
    document.getElementById('aylik-gelir').textContent = '+' + formatMoney(cuzdan.aylik_gelir || 0);
    document.getElementById('aylik-gider').textContent = '-' + formatMoney(cuzdan.aylik_gider || 0);
    updateChart();
    updateTransactions();
}

function formatMoney(amount) {
    return new Intl.NumberFormat('tr-TR', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount) + ' ₺';
}

function updateChart() {
    const container = document.getElementById('chart');
    const kategoriler = cuzdan.kategoriler || {};

    if (Object.keys(kategoriler).length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div>Henüz harcama yok</div>
            </div>
        `;
        return;
    }

    const maxTutar = Math.max(...Object.values(kategoriler));
    let html = '';
    const sorted = Object.entries(kategoriler).sort((a, b) => b[1] - a[1]);

    for (const [kat, tutar] of sorted) {
        const yuzde = (tutar / maxTutar) * 100;
        const renk = kategoriRenkleri[kat] || '#667eea';
        const emoji = kategoriEmoji[kat] || '💸';
        const isim = kategoriIsim[kat] || kat;

        html += `
            <div class="chart-bar">
                <div class="chart-label">${emoji} ${isim}</div>
                <div class="chart-bar-bg">
                    <div class="chart-bar-fill" style="width: ${yuzde}%; background: ${renk};"></div>
                </div>
                <div class="chart-value">${formatMoney(tutar)}</div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function updateTransactions() {
    const container = document.getElementById('transactions');
    const islemler = (cuzdan.islemler || []).slice().reverse().slice(0, 10);

    if (islemler.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div>Henüz işlem yok</div>
            </div>
        `;
        return;
    }

    let html = '';

    for (const islem of islemler) {
        const tipClass = islem.tip === 'gelir' ? 'gelir' : 'gider';
        const amountClass = islem.tip === 'gelir' ? 'green' : 'red';
        const sign = islem.tip === 'gelir' ? '+' : '-';
        const emoji = kategoriEmoji[islem.kategori] || '💰';
        const isim = kategoriIsim[islem.kategori] || islem.kategori;
        const tarih = new Date(islem.tarih).toLocaleDateString('tr-TR');

        html += `
            <div class="transaction" data-id="${islem.id}">
                <div class="transaction-icon ${tipClass}">${emoji}</div>
                <div class="transaction-info">
                    <div class="transaction-title">${isim}${islem.aciklama ? ' - ' + islem.aciklama : ''}</div>
                    <div class="transaction-date">${tarih}</div>
                </div>
                <div class="transaction-amount ${amountClass}">${sign}${formatMoney(islem.tutar)}</div>
                <button class="delete-btn" onclick="islemSil(${islem.id}, event)">🗑️</button>
            </div>
        `;
    }

    container.innerHTML = html;
}

// Modal fonksiyonları
function showGelirModal() {
    selectedKategori = null;
    document.querySelectorAll('#gelir-kategoriler .kat-btn').forEach(b => b.classList.remove('selected'));
    document.getElementById('gelir-tutar').value = '';
    document.getElementById('gelir-aciklama').value = '';
    document.getElementById('gelir-modal').classList.add('active');
}

function showGiderModal() {
    selectedKategori = null;
    document.querySelectorAll('#gider-kategoriler .kat-btn').forEach(b => b.classList.remove('selected'));
    document.getElementById('gider-tutar').value = '';
    document.getElementById('gider-aciklama').value = '';
    document.getElementById('gider-modal').classList.add('active');
}

function showBakiyeModal() {
    document.getElementById('baslangic-tutar').value = cuzdan.baslangic_bakiye || '';
    document.getElementById('bakiye-modal').classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
    selectedKategori = null;
}

async function gelirEkle() {
    const tutar = parseFloat(document.getElementById('gelir-tutar').value);
    const aciklama = document.getElementById('gelir-aciklama').value;

    if (!tutar || tutar <= 0) {
        tg.showAlert('Geçerli bir tutar girin');
        return;
    }

    if (!selectedKategori) {
        tg.showAlert('Kategori seçin');
        return;
    }

    const islem = {
        id: Date.now(),
        tip: 'gelir',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    };

    cuzdan.islemler.push(islem);
    hesaplaBakiye();
    await saveCuzdanData();
    updateCuzdanUI();

    closeModal('gelir-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Gelir eklendi!');
}

async function giderEkle() {
    const tutar = parseFloat(document.getElementById('gider-tutar').value);
    const aciklama = document.getElementById('gider-aciklama').value;

    if (!tutar || tutar <= 0) {
        tg.showAlert('Geçerli bir tutar girin');
        return;
    }

    if (!selectedKategori) {
        tg.showAlert('Kategori seçin');
        return;
    }

    const islem = {
        id: Date.now(),
        tip: 'gider',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    };

    cuzdan.islemler.push(islem);
    hesaplaBakiye();
    await saveCuzdanData();
    updateCuzdanUI();

    closeModal('gider-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Gider eklendi!');
}

async function baslangicBakiyeAyarla() {
    const tutar = parseFloat(document.getElementById('baslangic-tutar').value) || 0;

    cuzdan.baslangic_bakiye = tutar;
    hesaplaBakiye();
    await saveCuzdanData();
    updateCuzdanUI();

    closeModal('bakiye-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Başlangıç bakiyesi ayarlandı!');
}

function islemSil(id, event) {
    event.stopPropagation();

    const islem = cuzdan.islemler.find(i => i.id === id);
    if (!islem) return;

    silinecekIslemId = id;

    const tipText = islem.tip === 'gelir' ? 'Gelir' : 'Gider';
    const emoji = kategoriEmoji[islem.kategori] || '💰';
    const isim = kategoriIsim[islem.kategori] || islem.kategori;

    document.getElementById('sil-detay').innerHTML = `
        <strong>${tipText}:</strong> ${emoji} ${isim}<br>
        <strong>Tutar:</strong> ${formatMoney(islem.tutar)}
    `;

    document.getElementById('sil-modal').classList.add('active');
}

async function islemSilOnayla() {
    if (!silinecekIslemId) return;

    cuzdan.islemler = cuzdan.islemler.filter(i => i.id !== silinecekIslemId);
    silinecekIslemId = null;

    hesaplaBakiye();
    await saveCuzdanData();
    updateCuzdanUI();

    closeModal('sil-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('🗑️ İşlem silindi!');
}

function sendToAI() {
    let ozet = `📊 CÜZDAN RAPORU\n\n`;
    ozet += `💰 Bakiye: ${formatMoney(cuzdan.bakiye)}\n`;
    ozet += `📈 Bu ay gelir: ${formatMoney(cuzdan.aylik_gelir || 0)}\n`;
    ozet += `📉 Bu ay gider: ${formatMoney(cuzdan.aylik_gider || 0)}\n`;

    const kategoriler = cuzdan.kategoriler || {};
    if (Object.keys(kategoriler).length > 0) {
        ozet += `\n📋 Harcamalar:\n`;
        const sorted = Object.entries(kategoriler).sort((a, b) => b[1] - a[1]);
        for (const [kat, tutar] of sorted) {
            const isim = kategoriIsim[kat] || kat;
            const emoji = kategoriEmoji[kat] || '💸';
            ozet += `${emoji} ${isim}: ${formatMoney(tutar)}\n`;
        }
    }

    ozet += `\n✨ Bu raporu analiz et`;

    tg.showPopup({
        title: '🤖 AI Analiz',
        message: 'Rapor panoya kopyalanacak. Bot\'a yapıştırıp gönder.',
        buttons: [
            {id: 'copy', type: 'default', text: '📋 Kopyala'},
            {id: 'cancel', type: 'cancel'}
        ]
    }, function(buttonId) {
        if (buttonId === 'copy') {
            const textArea = document.createElement('textarea');
            textArea.value = ozet;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            tg.showAlert('✅ Kopyalandı! Bot\'a yapıştır ve gönder.');
            tg.close();
        }
    });
}

// ==================== NOT DEFTERİ FONKSİYONLARI ====================
function updateNotlarUI() {
    const container = document.getElementById('notes-list');

    if (notlar.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📝</div>
                <div>Henüz not yok</div>
            </div>
        `;
        return;
    }

    let html = '';
    const sortedNotlar = [...notlar].reverse();

    for (const not of sortedNotlar) {
        const tarih = new Date(not.tarih).toLocaleDateString('tr-TR', {
            day: 'numeric',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit'
        });

        let hatirlatmaHtml = '';
        if (not.hatirlatma) {
            const hatirlatmaTarih = new Date(not.hatirlatma).toLocaleDateString('tr-TR', {
                day: 'numeric',
                month: 'short',
                hour: '2-digit',
                minute: '2-digit'
            });
            hatirlatmaHtml = `<div class="note-reminder">⏰ ${hatirlatmaTarih}</div>`;
        }

        html += `
            <div class="note-item" data-id="${not.id}">
                <div class="note-content">
                    <div class="note-text">${not.icerik}</div>
                    <div class="note-date">${tarih}</div>
                    ${hatirlatmaHtml}
                </div>
                <button class="delete-btn" onclick="notSil(${not.id}, event)">🗑️</button>
            </div>
        `;
    }

    container.innerHTML = html;
}

function showNotEkleModal() {
    document.getElementById('not-icerik').value = '';
    document.getElementById('hatirlatma-aktif').checked = false;
    document.getElementById('hatirlatma-zaman').style.display = 'none';
    document.getElementById('hatirlatma-zaman').value = '';
    document.getElementById('not-ekle-modal').classList.add('active');
}

function toggleHatirlatma() {
    const checkbox = document.getElementById('hatirlatma-aktif');
    const zamanInput = document.getElementById('hatirlatma-zaman');

    if (checkbox.checked) {
        zamanInput.style.display = 'block';
        // Varsayılan olarak 1 saat sonrası
        const now = new Date();
        now.setHours(now.getHours() + 1);
        zamanInput.value = now.toISOString().slice(0, 16);
    } else {
        zamanInput.style.display = 'none';
    }
}

async function notEkle() {
    const icerik = document.getElementById('not-icerik').value.trim();

    if (!icerik) {
        tg.showAlert('Not içeriği boş olamaz');
        return;
    }

    const not = {
        id: Date.now(),
        icerik: icerik,
        tarih: new Date().toISOString()
    };

    // Hatırlatma varsa ekle
    const hatirlatmaAktif = document.getElementById('hatirlatma-aktif').checked;
    if (hatirlatmaAktif) {
        const hatirlatmaZaman = document.getElementById('hatirlatma-zaman').value;
        if (hatirlatmaZaman) {
            not.hatirlatma = new Date(hatirlatmaZaman).toISOString();
        }
    }

    notlar.push(not);
    await saveNotlarData();
    updateNotlarUI();

    closeModal('not-ekle-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Not eklendi!');
}

function notSil(id, event) {
    event.stopPropagation();

    const not = notlar.find(n => n.id === id);
    if (!not) return;

    silinecekNotId = id;

    const kisaIcerik = not.icerik.length > 50 ? not.icerik.substring(0, 50) + '...' : not.icerik;
    document.getElementById('not-sil-detay').textContent = kisaIcerik;

    document.getElementById('not-sil-modal').classList.add('active');
}

async function notSilOnayla() {
    if (!silinecekNotId) return;

    notlar = notlar.filter(n => n.id !== silinecekNotId);
    silinecekNotId = null;

    await saveNotlarData();
    updateNotlarUI();

    closeModal('not-sil-modal');

    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('🗑️ Not silindi!');
}

// ==================== KONUM FONKSİYONLARI ====================
function getLocation() {
    const locationText = document.getElementById('location-text');
    locationText.textContent = 'Konum alınıyor...';

    if (!navigator.geolocation) {
        locationText.textContent = 'Tarayıcınız konum desteklemiyor';
        return;
    }

    navigator.geolocation.getCurrentPosition(
        async (position) => {
            currentLocation = {
                lat: position.coords.latitude,
                lon: position.coords.longitude,
                timestamp: new Date().toISOString()
            };

            await saveKonumData();
            showLocationInfo();

            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('success');
            }
        },
        (error) => {
            let msg = 'Konum alınamadı';
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    msg = 'Konum izni reddedildi';
                    break;
                case error.POSITION_UNAVAILABLE:
                    msg = 'Konum bilgisi mevcut değil';
                    break;
                case error.TIMEOUT:
                    msg = 'Konum isteği zaman aşımına uğradı';
                    break;
            }
            locationText.textContent = msg;

            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('error');
            }
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        }
    );
}

async function showLocationInfo() {
    if (!currentLocation) return;

    const locationText = document.getElementById('location-text');
    locationText.textContent = `📍 ${currentLocation.lat.toFixed(4)}, ${currentLocation.lon.toFixed(4)}`;

    document.getElementById('location-info').style.display = 'block';

    // Hava durumu ve namaz vakitlerini yükle
    await Promise.all([
        loadWeather(),
        loadPrayerTimes()
    ]);
}

async function loadWeather() {
    const container = document.getElementById('weather-content');

    try {
        const response = await fetch(
            `https://wttr.in/${currentLocation.lat},${currentLocation.lon}?format=j1&lang=tr`
        );
        const data = await response.json();

        const current = data.current_condition[0];
        const temp = current.temp_C;
        const desc = current.lang_tr?.[0]?.value || current.weatherDesc[0].value;
        const feelsLike = current.FeelsLikeC;
        const humidity = current.humidity;

        container.innerHTML = `
            <div class="weather-main">
                <span class="weather-temp">${temp}°C</span>
                <span class="weather-desc">${desc}</span>
            </div>
            <div class="weather-details">
                <span>Hissedilen: ${feelsLike}°C</span>
                <span>Nem: %${humidity}</span>
            </div>
        `;
    } catch (e) {
        container.textContent = 'Hava durumu alınamadı';
    }
}

async function loadPrayerTimes() {
    const container = document.getElementById('prayer-content');

    try {
        const today = new Date();
        const dateStr = `${today.getDate()}-${today.getMonth() + 1}-${today.getFullYear()}`;

        const response = await fetch(
            `https://api.aladhan.com/v1/timings/${dateStr}?latitude=${currentLocation.lat}&longitude=${currentLocation.lon}&method=13`
        );
        const data = await response.json();
        const timings = data.data.timings;

        container.innerHTML = `
            <div class="prayer-times">
                <div class="prayer-time"><span>İmsak</span><span>${timings.Fajr}</span></div>
                <div class="prayer-time"><span>Güneş</span><span>${timings.Sunrise}</span></div>
                <div class="prayer-time"><span>Öğle</span><span>${timings.Dhuhr}</span></div>
                <div class="prayer-time"><span>İkindi</span><span>${timings.Asr}</span></div>
                <div class="prayer-time"><span>Akşam</span><span>${timings.Maghrib}</span></div>
                <div class="prayer-time"><span>Yatsı</span><span>${timings.Isha}</span></div>
            </div>
        `;
    } catch (e) {
        container.textContent = 'Namaz vakitleri alınamadı';
    }
}

async function findNearby(type) {
    const container = document.getElementById('nearby-results');
    const typeInfo = nearbyTypes[type];

    if (!currentLocation) {
        tg.showAlert('Önce konumunuzu alın');
        return;
    }

    container.innerHTML = `<div class="loading">🔍 ${typeInfo.name} aranıyor...</div>`;

    // Aktif butonu işaretle
    document.querySelectorAll('.nearby-btn').forEach(btn => btn.classList.remove('active'));
    event.currentTarget.classList.add('active');

    try {
        const radius = 10000; // 10km
        const query = `
            [out:json][timeout:10];
            node[${typeInfo.query}](around:${radius},${currentLocation.lat},${currentLocation.lon});
            out body 10;
        `;

        const response = await fetch('https://overpass-api.de/api/interpreter', {
            method: 'POST',
            body: query
        });

        const data = await response.json();

        if (data.elements.length === 0) {
            container.innerHTML = `<div class="empty-state">Yakında ${typeInfo.name.toLowerCase()} bulunamadı</div>`;
            return;
        }

        let html = '';

        for (const place of data.elements.slice(0, 5)) {
            const name = place.tags?.name || `${typeInfo.name}`;
            const distance = calculateDistance(
                currentLocation.lat, currentLocation.lon,
                place.lat, place.lon
            );

            html += `
                <div class="nearby-item" onclick="openInMaps(${place.lat}, ${place.lon})">
                    <div class="nearby-icon">${typeInfo.emoji}</div>
                    <div class="nearby-info">
                        <div class="nearby-name">${name}</div>
                        <div class="nearby-distance">${distance}</div>
                    </div>
                    <div class="nearby-arrow">→</div>
                </div>
            `;
        }

        container.innerHTML = html;

    } catch (e) {
        container.innerHTML = `<div class="empty-state">Arama yapılamadı</div>`;
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000; // Dünya yarıçapı (metre)
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c;

    if (distance < 1000) {
        return Math.round(distance) + ' m';
    } else {
        return (distance / 1000).toFixed(1) + ' km';
    }
}

function openInMaps(lat, lon) {
    // iOS/macOS ise Apple Maps, değilse Google Maps
    const isApple = /iPad|iPhone|iPod|Macintosh/.test(navigator.userAgent);

    let url;
    if (isApple) {
        // Apple Maps (http versiyonu daha güvenli)
        url = `https://maps.apple.com/?daddr=${lat},${lon}&dirflg=d`;
    } else {
        // Google Maps
        url = `https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}`;
    }

    // Telegram içindeyse tg.openLink kullan
    if (tg.openLink) {
        tg.openLink(url);
    } else {
        window.open(url, '_blank');
    }
}
