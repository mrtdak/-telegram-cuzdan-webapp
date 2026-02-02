// Telegram Web App
const tg = window.Telegram.WebApp;

// Tema renkleri
tg.expand();
tg.ready();

// Kullanıcı bilgisi
const user = tg.initDataUnsafe?.user;
const userId = user?.id || 'demo';

// API URL (bot'unuzun backend'i)
const API_URL = 'https://YOUR_BACKEND_URL/api'; // Bunu değiştirin

// State
let selectedKategori = null;
let veriler = {
    bakiye: 0,
    aylik_gelir: 0,
    aylik_gider: 0,
    islemler: [],
    kategoriler: {}
};

// Kategori renkleri
const kategoriRenkleri = {
    market: '#22c55e',
    yemek: '#f59e0b',
    fatura: '#3b82f6',
    yakit: '#ef4444',
    ulasim: '#8b5cf6',
    saglik: '#ec4899',
    giyim: '#06b6d4',
    eglence: '#f97316',
    kira: '#6366f1',
    diger_gider: '#94a3b8',
    maas: '#22c55e',
    ek_gelir: '#10b981',
    yatirim: '#3b82f6',
    hediye: '#f59e0b',
    iade: '#8b5cf6',
    diger_gelir: '#94a3b8'
};

const kategoriEmoji = {
    market: '🛒',
    yemek: '🍔',
    fatura: '📄',
    yakit: '⛽',
    ulasim: '🚌',
    saglik: '💊',
    giyim: '👕',
    eglence: '🎮',
    kira: '🏠',
    diger_gider: '💸',
    maas: '💼',
    ek_gelir: '💵',
    yatirim: '📈',
    hediye: '🎁',
    iade: '🔄',
    diger_gelir: '💰'
};

const kategoriIsim = {
    market: 'Market',
    yemek: 'Yemek',
    fatura: 'Fatura',
    yakit: 'Yakıt',
    ulasim: 'Ulaşım',
    saglik: 'Sağlık',
    giyim: 'Giyim',
    eglence: 'Eğlence',
    kira: 'Kira',
    diger_gider: 'Diğer',
    maas: 'Maaş',
    ek_gelir: 'Ek Gelir',
    yatirim: 'Yatırım',
    hediye: 'Hediye',
    iade: 'İade',
    diger_gelir: 'Diğer'
};

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', () => {
    // Demo veri yükle (backend olmadan test için)
    loadDemoData();

    // Kategori butonlarına event listener ekle
    document.querySelectorAll('.kat-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('.kategori-grid');
            parent.querySelectorAll('.kat-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedKategori = btn.dataset.kat;
        });
    });
});

// Demo veri (backend olmadan test için)
function loadDemoData() {
    veriler = {
        bakiye: 24850,
        aylik_gelir: 25000,
        aylik_gider: 1570,
        islemler: [
            { tip: 'gelir', tutar: 25000, kategori: 'maas', aciklama: 'Şubat maaşı', tarih: '2026-02-01' },
            { tip: 'gider', tutar: 450, kategori: 'market', aciklama: 'Haftalık alışveriş', tarih: '2026-02-02' },
            { tip: 'gider', tutar: 320, kategori: 'yemek', aciklama: '', tarih: '2026-02-02' },
            { tip: 'gider', tutar: 800, kategori: 'yakit', aciklama: 'Benzin', tarih: '2026-02-01' }
        ],
        kategoriler: {
            market: 450,
            yemek: 320,
            yakit: 800
        }
    };

    updateUI();
}

// Backend'den veri yükle
async function loadData() {
    try {
        const response = await fetch(`${API_URL}/cuzdan/${userId}`);
        if (response.ok) {
            veriler = await response.json();
            updateUI();
        }
    } catch (e) {
        console.log('Backend bağlantısı yok, demo veri kullanılıyor');
        loadDemoData();
    }
}

// UI güncelle
function updateUI() {
    // Bakiye
    document.getElementById('bakiye').textContent = formatMoney(veriler.bakiye);
    document.getElementById('aylik-gelir').textContent = '+' + formatMoney(veriler.aylik_gelir);
    document.getElementById('aylik-gider').textContent = '-' + formatMoney(veriler.aylik_gider);

    // Grafik
    updateChart();

    // İşlemler
    updateTransactions();
}

// Para formatla
function formatMoney(amount) {
    return new Intl.NumberFormat('tr-TR', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount) + ' ₺';
}

// Grafik güncelle
function updateChart() {
    const container = document.getElementById('chart');
    const kategoriler = veriler.kategoriler;

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

// İşlemler güncelle
function updateTransactions() {
    const container = document.getElementById('transactions');
    const islemler = veriler.islemler.slice(0, 10);

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
            <div class="transaction">
                <div class="transaction-icon ${tipClass}">${emoji}</div>
                <div class="transaction-info">
                    <div class="transaction-title">${isim}${islem.aciklama ? ' - ' + islem.aciklama : ''}</div>
                    <div class="transaction-date">${tarih}</div>
                </div>
                <div class="transaction-amount ${amountClass}">${sign}${formatMoney(islem.tutar)}</div>
            </div>
        `;
    }

    container.innerHTML = html;
}

// Modal göster
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

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
    selectedKategori = null;
}

// Gelir ekle
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

    // Demo modda local güncelle
    veriler.bakiye += tutar;
    veriler.aylik_gelir += tutar;
    veriler.islemler.unshift({
        tip: 'gelir',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    });

    updateUI();
    closeModal('gelir-modal');
    tg.showAlert('✅ Gelir eklendi!');

    // Backend'e gönder
    sendToBot('gelir', tutar, selectedKategori, aciklama);
}

// Gider ekle
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

    // Demo modda local güncelle
    veriler.bakiye -= tutar;
    veriler.aylik_gider += tutar;
    veriler.kategoriler[selectedKategori] = (veriler.kategoriler[selectedKategori] || 0) + tutar;
    veriler.islemler.unshift({
        tip: 'gider',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    });

    updateUI();
    closeModal('gider-modal');
    tg.showAlert('✅ Gider eklendi!');

    // Backend'e gönder
    sendToBot('gider', tutar, selectedKategori, aciklama);
}

// Bot'a veri gönder
function sendToBot(tip, tutar, kategori, aciklama) {
    const data = {
        action: 'cuzdan_islem',
        tip: tip,
        tutar: tutar,
        kategori: kategori,
        aciklama: aciklama
    };

    // Telegram üzerinden gönder
    tg.sendData(JSON.stringify(data));
}
