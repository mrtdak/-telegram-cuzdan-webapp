"""
Veritabani Yonetici Modulu
SQLite (lokal) / PostgreSQL (VPS) uyumlu

Kullanici yonetimi, plan takibi, rate limiting
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from enum import Enum


class PlanType(Enum):
    """Kullanici plan turleri - Beta: Sadece FREE"""
    FREE = "free"


# Plan limitleri - Beta donemi (tum ozellikler acik, gunluk limitli)
PLAN_LIMITS = {
    PlanType.FREE: {
        "daily_messages": 30,              # Gunluk mesaj limiti
        "daily_camera_notifications": 5,   # Gunluk kamera bildirimi limiti
        "daily_location_queries": 10,      # Gunluk konum sorgusu limiti
        "max_cameras": 1,                  # Maksimum kamera sayisi
        "web_search": True,
        "photo_analysis": True,
        "memory": True,
        "notes": True,
        "location": True
    }
}


class DatabaseManager:
    """
    Veritabani yonetici sinifi.
    SQLite ve PostgreSQL destekler.
    """

    def __init__(self, db_path: str = "bot_database.db", use_postgres: bool = False):
        """
        Args:
            db_path: SQLite dosya yolu veya PostgreSQL connection string
            use_postgres: True ise PostgreSQL kullan
        """
        self.db_path = db_path
        self.use_postgres = use_postgres

        if use_postgres:
            # PostgreSQL icin psycopg2 gerekli
            try:
                import psycopg2
                self.pg_module = psycopg2
            except ImportError:
                raise ImportError("PostgreSQL icin 'pip install psycopg2-binary' calistirin")

        self._init_database()

    @contextmanager
    def get_connection(self):
        """Veritabani baglantisi context manager"""
        if self.use_postgres:
            conn = self.pg_module.connect(self.db_path)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_database(self):
        """Veritabani tablolarini olustur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Kullanicilar tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    plan TEXT DEFAULT 'free',
                    plan_start_date TEXT,
                    plan_end_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_active TEXT,
                    is_blocked INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

            # Gunluk kullanim tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date TEXT,
                    message_count INTEGER DEFAULT 0,
                    web_search_count INTEGER DEFAULT 0,
                    photo_count INTEGER DEFAULT 0,
                    camera_notification_count INTEGER DEFAULT 0,
                    location_query_count INTEGER DEFAULT 0,
                    UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Yeni sutunlari ekle (eski veritabanlari icin)
            try:
                cursor.execute("ALTER TABLE daily_usage ADD COLUMN camera_notification_count INTEGER DEFAULT 0")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE daily_usage ADD COLUMN location_query_count INTEGER DEFAULT 0")
            except:
                pass

            # Odeme gecmisi
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    plan TEXT,
                    amount_tl REAL,
                    payment_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    payment_method TEXT,
                    transaction_id TEXT,
                    status TEXT DEFAULT 'completed',
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Admin loglari
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
                    action TEXT,
                    target_user_id INTEGER,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    # ==================== KULLANICI ISLEMLERI ====================

    def get_or_create_user(self, user_id: int, username: str = None,
                           first_name: str = None, last_name: str = None) -> Dict:
        """Kullaniciyi getir veya olustur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Kullanici var mi?
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                # Guncelle
                cursor.execute("""
                    UPDATE users
                    SET username = ?, first_name = ?, last_name = ?, last_active = ?
                    WHERE user_id = ?
                """, (username, first_name, last_name, datetime.now().isoformat(), user_id))
                conn.commit()

                # Tekrar oku
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
            else:
                # Yeni kullanici
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO users (user_id, username, first_name, last_name,
                                      plan, plan_start_date, created_at, last_active)
                    VALUES (?, ?, ?, ?, 'free', ?, ?, ?)
                """, (user_id, username, first_name, last_name, now, now, now))
                conn.commit()

                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()

            return dict(row)

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Kullanici bilgilerini getir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_plan(self, user_id: int) -> PlanType:
        """Kullanicinin aktif planini getir"""
        user = self.get_user(user_id)
        if not user:
            return PlanType.FREE

        plan_str = user.get("plan", "free")

        # Plan suresi dolmus mu?
        if plan_str != "free" and user.get("plan_end_date"):
            end_date = datetime.fromisoformat(user["plan_end_date"])
            if datetime.now() > end_date:
                # Plan dolmus, free'ye dusur
                self._downgrade_to_free(user_id)
                return PlanType.FREE

        try:
            return PlanType(plan_str)
        except ValueError:
            return PlanType.FREE

    def _downgrade_to_free(self, user_id: int):
        """Kullaniciyi free plana dusur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET plan = 'free', plan_end_date = NULL
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()

    def upgrade_plan(self, user_id: int, new_plan: PlanType, months: int = 1) -> bool:
        """Kullanici planini yukselt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now()
            end_date = now + timedelta(days=30 * months)

            cursor.execute("""
                UPDATE users
                SET plan = ?, plan_start_date = ?, plan_end_date = ?
                WHERE user_id = ?
            """, (new_plan.value, now.isoformat(), end_date.isoformat(), user_id))
            conn.commit()

            return cursor.rowcount > 0

    def block_user(self, user_id: int, reason: str = None) -> bool:
        """Kullaniciyi engelle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET is_blocked = 1, notes = ?
                WHERE user_id = ?
            """, (reason, user_id))
            conn.commit()
            return cursor.rowcount > 0

    def unblock_user(self, user_id: int) -> bool:
        """Kullanici engelini kaldir"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET is_blocked = 0
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    def is_user_blocked(self, user_id: int) -> bool:
        """Kullanici engelli mi?"""
        user = self.get_user(user_id)
        return user.get("is_blocked", 0) == 1 if user else False

    # ==================== KULLANIM TAKIBI ====================

    def get_daily_usage(self, user_id: int, date: str = None) -> Dict:
        """Gunluk kullanimi getir"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, date))
            row = cursor.fetchone()

            if row:
                return dict(row)
            else:
                return {
                    "user_id": user_id,
                    "date": date,
                    "message_count": 0,
                    "web_search_count": 0,
                    "photo_count": 0,
                    "camera_notification_count": 0,
                    "location_query_count": 0
                }

    def increment_usage(self, user_id: int, field: str = "message_count") -> int:
        """Kullanimi artir, yeni degeri dondur"""
        date = datetime.now().strftime("%Y-%m-%d")

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Kayit var mi?
            cursor.execute("""
                SELECT id FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, date))

            if cursor.fetchone():
                # Guncelle
                cursor.execute(f"""
                    UPDATE daily_usage
                    SET {field} = {field} + 1
                    WHERE user_id = ? AND date = ?
                """, (user_id, date))
            else:
                # Yeni kayit
                cursor.execute(f"""
                    INSERT INTO daily_usage (user_id, date, {field})
                    VALUES (?, ?, 1)
                """, (user_id, date))

            conn.commit()

            # Yeni degeri oku
            cursor.execute(f"""
                SELECT {field} FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, date))
            row = cursor.fetchone()
            return row[0] if row else 1

    # ==================== RATE LIMITING ====================

    def check_rate_limit(self, user_id: int) -> Dict:
        """
        Kullanici limitini kontrol et.

        Returns:
            {
                "allowed": True/False,
                "remaining": kalan mesaj sayisi (-1 = sinirsiz),
                "limit": toplam limit,
                "plan": plan adi,
                "message": kullaniciya gosterilecek mesaj (limit asildiysa)
            }
        """
        # Kullaniciyi al/olustur
        self.get_or_create_user(user_id)

        # Engelli mi?
        if self.is_user_blocked(user_id):
            return {
                "allowed": False,
                "remaining": 0,
                "limit": 0,
                "plan": "blocked",
                "message": "Hesabiniz engellenmi\u015f. Destek icin: @admin"
            }

        # Plan ve limitleri al
        plan = self.get_user_plan(user_id)
        limits = PLAN_LIMITS[plan]
        daily_limit = limits["daily_messages"]

        # Sinirsiz plan
        if daily_limit == -1:
            return {
                "allowed": True,
                "remaining": -1,
                "limit": -1,
                "plan": plan.value,
                "message": None
            }

        # Gunluk kullanimi kontrol et
        usage = self.get_daily_usage(user_id)
        current_count = usage["message_count"]

        if current_count >= daily_limit:
            return {
                "allowed": False,
                "remaining": 0,
                "limit": daily_limit,
                "plan": plan.value,
                "message": f"Bugunluk {daily_limit} mesaj hakkin doldu.\n"
                          f"Yarin sifirlanir, gorusuruz!\n\n"
                          f"Projeyi desteklemek istersen: /bagis"
            }

        return {
            "allowed": True,
            "remaining": daily_limit - current_count,
            "limit": daily_limit,
            "plan": plan.value,
            "message": None
        }

    def can_use_feature(self, user_id: int, feature: str) -> bool:
        """
        Kullanici belirli bir ozelligi kullanabilir mi?

        Args:
            feature: "web_search", "photo_analysis", "memory", "notes", "location"
        """
        plan = self.get_user_plan(user_id)
        limits = PLAN_LIMITS[plan]
        return limits.get(feature, False)

    def check_camera_limit(self, user_id: int) -> Dict:
        """Kamera bildirimi limitini kontrol et"""
        self.get_or_create_user(user_id)

        plan = self.get_user_plan(user_id)
        limits = PLAN_LIMITS[plan]
        daily_limit = limits["daily_camera_notifications"]

        usage = self.get_daily_usage(user_id)
        current_count = usage.get("camera_notification_count", 0)

        if current_count >= daily_limit:
            return {
                "allowed": False,
                "remaining": 0,
                "limit": daily_limit,
                "message": f"Bugunluk {daily_limit} kamera bildirimi limitin doldu.\n"
                          f"Yarin sifirlanir!\n\n"
                          f"Projeyi desteklemek istersen: /bagis"
            }

        return {
            "allowed": True,
            "remaining": daily_limit - current_count,
            "limit": daily_limit,
            "message": None
        }

    def check_location_limit(self, user_id: int) -> Dict:
        """Konum sorgusu limitini kontrol et"""
        self.get_or_create_user(user_id)

        plan = self.get_user_plan(user_id)
        limits = PLAN_LIMITS[plan]
        daily_limit = limits["daily_location_queries"]

        usage = self.get_daily_usage(user_id)
        current_count = usage.get("location_query_count", 0)

        if current_count >= daily_limit:
            return {
                "allowed": False,
                "remaining": 0,
                "limit": daily_limit,
                "message": f"Bugunluk {daily_limit} konum sorgusu limitin doldu.\n"
                          f"Yarin sifirlanir!\n\n"
                          f"Projeyi desteklemek istersen: /bagis"
            }

        return {
            "allowed": True,
            "remaining": daily_limit - current_count,
            "limit": daily_limit,
            "message": None
        }

    def get_max_cameras(self, user_id: int) -> int:
        """Kullanicinin ekleyebilecegi maksimum kamera sayisi"""
        plan = self.get_user_plan(user_id)
        limits = PLAN_LIMITS[plan]
        return limits.get("max_cameras", 1)

    # ==================== ODEME ISLEMLERI ====================

    def record_payment(self, user_id: int, plan: PlanType, amount_tl: float,
                       payment_method: str, transaction_id: str) -> int:
        """Odeme kaydi olustur"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO payments (user_id, plan, amount_tl, payment_method, transaction_id)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, plan.value, amount_tl, payment_method, transaction_id))
            conn.commit()
            return cursor.lastrowid

    def get_user_payments(self, user_id: int) -> List[Dict]:
        """Kullanicinin odeme gecmisi"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM payments
                WHERE user_id = ?
                ORDER BY payment_date DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== ADMIN ISLEMLERI ====================

    def get_stats(self) -> Dict:
        """Genel istatistikler"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Toplam kullanici
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]

            # Plan dagilimi
            cursor.execute("""
                SELECT plan, COUNT(*) as count
                FROM users
                GROUP BY plan
            """)
            plan_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            # Bugunun aktif kullanicilari
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) FROM daily_usage
                WHERE date = ?
            """, (today,))
            active_today = cursor.fetchone()[0]

            # Bugunun toplam mesaj sayisi
            cursor.execute("""
                SELECT SUM(message_count) FROM daily_usage
                WHERE date = ?
            """, (today,))
            messages_today = cursor.fetchone()[0] or 0

            # Son 7 gun
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) FROM daily_usage
                WHERE date >= ?
            """, (week_ago,))
            active_week = cursor.fetchone()[0]

            return {
                "total_users": total_users,
                "plan_distribution": plan_distribution,
                "active_today": active_today,
                "messages_today": messages_today,
                "active_week": active_week
            }

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Tum kullanicilari listele"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users
                ORDER BY last_active DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def log_admin_action(self, admin_id: int, action: str,
                         target_user_id: int = None, details: str = None):
        """Admin islemini logla"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO admin_logs (admin_id, action, target_user_id, details)
                VALUES (?, ?, ?, ?)
            """, (admin_id, action, target_user_id, details))
            conn.commit()


# Singleton instance
_db_instance = None


def get_db() -> DatabaseManager:
    """Veritabani instance'ini getir (singleton)"""
    global _db_instance
    if _db_instance is None:
        # Proje klasorunde veritabani olustur
        db_path = os.path.join(os.path.dirname(__file__), "bot_database.db")
        _db_instance = DatabaseManager(db_path)
    return _db_instance


# ==================== TEST ====================

if __name__ == "__main__":
    print("Veritabani Test")
    print("=" * 50)

    db = get_db()

    # Test kullanici
    test_user_id = 123456789

    # Kullanici olustur
    user = db.get_or_create_user(test_user_id, "test_user", "Test", "User")
    print(f"\nKullanici: {user}")

    # Plan kontrol
    plan = db.get_user_plan(test_user_id)
    print(f"Plan: {plan.value}")

    # Rate limit kontrol (30 mesaj limiti)
    for i in range(35):
        result = db.check_rate_limit(test_user_id)
        if result["allowed"]:
            db.increment_usage(test_user_id)
            print(f"Mesaj {i+1}: OK (Kalan: {result['remaining']})")
        else:
            print(f"Mesaj {i+1}: ENGELLENDI - {result['message']}")
            break

    # Kamera limiti test
    print("\n--- Kamera Limiti Test ---")
    for i in range(7):
        result = db.check_camera_limit(test_user_id)
        if result["allowed"]:
            db.increment_usage(test_user_id, "camera_notification_count")
            print(f"Kamera {i+1}: OK (Kalan: {result['remaining']})")
        else:
            print(f"Kamera {i+1}: ENGELLENDI")
            break

    # Istatistikler
    print("\n--- Istatistikler ---")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
