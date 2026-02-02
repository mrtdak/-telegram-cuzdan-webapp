// Telegram Web App
const tg = window.Telegram.WebApp;

// Tema renkleri
tg.expand();
tg.ready();

// Kullanıcı bilgisi
const user = tg.initDataUnsafe?.user;
const userId = user?.id || 'demo';

// State
let selectedKategori = null;
let veriler = {
    bakiye: 0,
    baslangic_bakiye: 0,
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
document.addEventListener('DOMContentLoaded', async () => {
    // Telegram CloudStorage'dan veri yükle
    await loadData();

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

// Telegram CloudStorage'dan veri yükle
async function loadData() {
    try {
        // CloudStorage'dan oku
        if (tg.CloudStorage) {
            tg.CloudStorage.getItem('cuzdan_veriler', (err, value) => {
                if (!err && value) {
                    veriler = JSON.parse(value);
                    hesaplaBakiye();
                    updateUI();
                } else {
                    // İlk kullanım - boş veri
                    veriler = {
                        bakiye: 0,
                        baslangic_bakiye: 0,
                        islemler: []
                    };
                    updateUI();
                }
            });
        } else {
            // CloudStorage yoksa localStorage kullan
            const saved = localStorage.getItem('cuzdan_veriler');
            if (saved) {
                veriler = JSON.parse(saved);
                hesaplaBakiye();
            }
            updateUI();
        }
    } catch (e) {
        console.error('Veri yükleme hatası:', e);
        updateUI();
    }
}

// Veriyi kaydet
async function saveData() {
    try {
        const dataStr = JSON.stringify(veriler);

        if (tg.CloudStorage) {
            tg.CloudStorage.setItem('cuzdan_veriler', dataStr, (err) => {
                if (err) console.error('CloudStorage kayıt hatası:', err);
            });
        } else {
            localStorage.setItem('cuzdan_veriler', dataStr);
        }
    } catch (e) {
        console.error('Kaydetme hatası:', e);
    }
}

// Bakiye hesapla
function hesaplaBakiye() {
    let bakiye = veriler.baslangic_bakiye || 0;

    veriler.islemler.forEach(islem => {
        if (islem.tip === 'gelir') {
            bakiye += islem.tutar;
        } else {
            bakiye -= islem.tutar;
        }
    });

    veriler.bakiye = bakiye;

    // Bu ayın gelir/gider hesapla
    const now = new Date();
    const ayBasi = new Date(now.getFullYear(), now.getMonth(), 1);

    veriler.aylik_gelir = 0;
    veriler.aylik_gider = 0;
    veriler.kategoriler = {};

    veriler.islemler.forEach(islem => {
        const islemTarih = new Date(islem.tarih);
        if (islemTarih >= ayBasi) {
            if (islem.tip === 'gelir') {
                veriler.aylik_gelir += islem.tutar;
            } else {
                veriler.aylik_gider += islem.tutar;
                veriler.kategoriler[islem.kategori] = (veriler.kategoriler[islem.kategori] || 0) + islem.tutar;
            }
        }
    });
}

// UI güncelle
function updateUI() {
    // Bakiye
    document.getElementById('bakiye').textContent = formatMoney(veriler.bakiye);
    document.getElementById('aylik-gelir').textContent = '+' + formatMoney(veriler.aylik_gelir || 0);
    document.getElementById('aylik-gider').textContent = '-' + formatMoney(veriler.aylik_gider || 0);

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
    const kategoriler = veriler.kategoriler || {};

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
    const islemler = (veriler.islemler || []).slice().reverse().slice(0, 10);

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

    // İşlem ekle
    const islem = {
        id: Date.now(),
        tip: 'gelir',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    };

    veriler.islemler.push(islem);
    hesaplaBakiye();
    await saveData();
    updateUI();

    closeModal('gelir-modal');

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Gelir eklendi!');
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

    // İşlem ekle
    const islem = {
        id: Date.now(),
        tip: 'gider',
        tutar: tutar,
        kategori: selectedKategori,
        aciklama: aciklama,
        tarih: new Date().toISOString()
    };

    veriler.islemler.push(islem);
    hesaplaBakiye();
    await saveData();
    updateUI();

    closeModal('gider-modal');

    // Haptic feedback
    if (tg.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred('success');
    }

    tg.showAlert('✅ Gider eklendi!');
}
