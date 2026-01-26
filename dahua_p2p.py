"""
Dahua P2P Bağlantı Modülü
Kullanıcının yerel kamerasına P2P üzerinden bağlanır.
Port forwarding gerektirmez!

Kaynak: https://github.com/khoanguyen-3fc/dh-p2p
"""
import subprocess
import sys
import os
import time
import threading
from typing import Optional, Dict

# P2P process'lerini takip et
p2p_processes: Dict[str, subprocess.Popen] = {}


def start_p2p_tunnel(serial: str, username: str, password: str, port: int = 554) -> bool:
    """
    P2P tüneli başlat

    Args:
        serial: DVR/Kamera seri numarası
        username: Kullanıcı adı (genelde admin)
        password: Şifre
        port: Dinlenecek yerel port (varsayılan 554)

    Returns:
        bool: Başarılı mı
    """
    global p2p_processes

    # Zaten çalışıyorsa durdur
    if serial in p2p_processes:
        stop_p2p_tunnel(serial)

    # dh-p2p dizini
    dh_p2p_dir = os.path.join(os.path.dirname(__file__), "dh-p2p")
    main_py = os.path.join(dh_p2p_dir, "main.py")

    if not os.path.exists(main_py):
        print(f"❌ dh-p2p bulunamadı: {main_py}")
        return False

    try:
        # P2P process'i başlat
        # --type 1 = kimlik doğrulama gerekli (çoğu DVR için)
        cmd = [
            sys.executable,
            main_py,
            "--type", "1",
            "--username", username,
            "--password", password,
            serial
        ]

        # Subprocess başlat
        process = subprocess.Popen(
            cmd,
            cwd=dh_p2p_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        p2p_processes[serial] = process

        # Bağlantının kurulmasını bekle
        time.sleep(3)

        # Process hala çalışıyor mu kontrol et
        if process.poll() is not None:
            # Process bitti, hata var
            stdout, stderr = process.communicate()
            print(f"❌ P2P bağlantı hatası: {stderr or stdout}")
            return False

        print(f"✅ P2P tüneli başlatıldı: {serial} -> localhost:{port}")
        return True

    except Exception as e:
        print(f"❌ P2P başlatma hatası: {e}")
        return False


def stop_p2p_tunnel(serial: str) -> bool:
    """P2P tünelini durdur"""
    global p2p_processes

    if serial not in p2p_processes:
        return False

    try:
        process = p2p_processes[serial]
        process.terminate()
        process.wait(timeout=5)
        del p2p_processes[serial]
        print(f"⏹️ P2P tüneli durduruldu: {serial}")
        return True
    except Exception as e:
        print(f"⚠️ P2P durdurma hatası: {e}")
        # Zorla öldür
        try:
            process.kill()
            del p2p_processes[serial]
        except:
            pass
        return False


def is_p2p_running(serial: str) -> bool:
    """P2P tüneli çalışıyor mu"""
    if serial not in p2p_processes:
        return False

    return p2p_processes[serial].poll() is None


def get_rtsp_url(serial: str, username: str, password: str, channel: int = 1) -> str:
    """
    P2P üzerinden RTSP URL döndür
    P2P tüneli açıkken localhost üzerinden bağlanır
    """
    return f"rtsp://{username}:{password}@127.0.0.1/cam/realmonitor?channel={channel}&subtype=0"


async def test_p2p_connection(serial: str, username: str, password: str) -> Dict:
    """
    P2P bağlantısını test et

    Returns:
        {"success": bool, "message": str}
    """
    try:
        # P2P tüneli başlat
        if not start_p2p_tunnel(serial, username, password):
            return {"success": False, "message": "P2P tüneli başlatılamadı"}

        # Biraz bekle
        time.sleep(2)

        # Tünel çalışıyor mu
        if not is_p2p_running(serial):
            return {"success": False, "message": "P2P bağlantısı kurulamadı"}

        # OpenCV ile test et
        import cv2
        rtsp_url = get_rtsp_url(serial, username, password)
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            stop_p2p_tunnel(serial)
            return {"success": False, "message": "RTSP stream açılamadı"}

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            stop_p2p_tunnel(serial)
            return {"success": False, "message": "Görüntü alınamadı"}

        # Test başarılı, tüneli kapat
        stop_p2p_tunnel(serial)
        return {"success": True, "message": "Bağlantı başarılı!"}

    except Exception as e:
        stop_p2p_tunnel(serial)
        return {"success": False, "message": f"Hata: {str(e)}"}
