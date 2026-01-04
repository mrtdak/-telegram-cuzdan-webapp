# -*- coding: utf-8 -*-
"""
🧠 HAFIZA VE BAĞLAM TESTİ
Sohbet akışı, hafıza, bağlam takibi testleri
"""

import sys
import os
import io
import asyncio
import json

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

from dotenv import load_dotenv
load_dotenv()

from personal_ai import PersonalAI

# ========================================
# TEST SENARYOLARI
# ========================================

MEMORY_TESTS = [
    # ----------------------------------------
    # TEST 1: Kısa süreli hafıza (aynı sohbet içinde)
    # ----------------------------------------
    {
        "name": "🧠 Kısa Süreli Hafıza",
        "description": "Aynı sohbet içinde önceki mesajları hatırlıyor mu?",
        "conversation": [
            {"user": "Benim adım Ayşe", "check_in_response": None},
            {"user": "Bugün doğum günüm", "check_in_response": None},
            {"user": "Benim adım neydi hatırlıyor musun", "check_in_response": ["ayşe", "Ayşe"]},
        ],
    },

    # ----------------------------------------
    # TEST 2: Bağlam takibi (konu devamlılığı)
    # ----------------------------------------
    {
        "name": "🔗 Bağlam Takibi",
        "description": "Konuyu takip edebiliyor mu?",
        "conversation": [
            {"user": "Python öğrenmeye başladım", "check_in_response": None},
            {"user": "Bunu nasıl daha iyi yapabilirim", "check_in_response": ["python", "Python", "programlama", "kod", "öğren"]},
        ],
    },

    # ----------------------------------------
    # TEST 3: Zamir çözümleme (onu, bunu, şunu)
    # ----------------------------------------
    {
        "name": "🎯 Zamir Çözümleme",
        "description": "'Onu', 'bunu' gibi zamirleri doğru anlıyor mu?",
        "conversation": [
            {"user": "Dün bir kedi gördüm", "check_in_response": None},
            {"user": "Çok tatlıydı", "check_in_response": None},
            {"user": "Onu sahiplenmek istiyorum", "check_in_response": ["kedi", "Kedi", "sahiplen", "hayvan"]},
        ],
    },

    # ----------------------------------------
    # TEST 4: Rol tutarlılığı
    # ----------------------------------------
    {
        "name": "🎭 Rol Tutarlılığı",
        "description": "Farklı konularda tutarlı kişilik gösteriyor mu?",
        "conversation": [
            {"user": "Selam nasılsın", "check_in_response": None},
            {"user": "JavaScript nedir", "check_in_response": ["JavaScript", "programlama", "dil", "web"]},
            {"user": "Teşekkürler anladım", "check_in_response": None},
        ],
    },

    # ----------------------------------------
    # TEST 5: Uzun bağlam (5+ mesaj)
    # ----------------------------------------
    {
        "name": "📚 Uzun Bağlam",
        "description": "5+ mesaj sonra hala ilk konuyu hatırlıyor mu?",
        "conversation": [
            {"user": "Bir proje yapıyorum, e-ticaret sitesi", "check_in_response": None},
            {"user": "React kullanacağım", "check_in_response": None},
            {"user": "Veritabanı için ne önerirsin", "check_in_response": None},
            {"user": "Tamam PostgreSQL olsun", "check_in_response": None},
            {"user": "Ödeme sistemi nasıl entegre ederim", "check_in_response": None},
            {"user": "Bu projenin adı ne olsun sence", "check_in_response": ["e-ticaret", "ticaret", "proje", "site", "alışveriş"]},
        ],
    },

    # ----------------------------------------
    # TEST 6: Konu değişikliği algılama
    # ----------------------------------------
    {
        "name": "🔄 Konu Değişikliği",
        "description": "Yeni konuya geçişi doğru algılıyor mu?",
        "conversation": [
            {"user": "Bugün hava çok güzel", "check_in_response": None},
            {"user": "Bu arada, en sevdiğim yemek lahmacun", "check_in_response": ["lahmacun", "yemek"]},
        ],
    },
]

async def run_memory_test(ai, test):
    """Tek bir hafıza testini çalıştır"""
    chat_history = []
    all_responses = []
    test_passed = True
    failure_reason = None

    print(f"\n{'='*60}")
    print(f"📋 {test['name']}")
    print(f"   {test['description']}")
    print(f"{'='*60}")

    for i, turn in enumerate(test["conversation"]):
        user_msg = turn["user"]
        expected = turn.get("check_in_response")

        print(f"\n[{i+1}] 👤 Kullanıcı: {user_msg}")

        try:
            response, _, _ = await ai.process(user_msg, chat_history)
        except Exception as e:
            print(f"   ❌ HATA: {e}")
            test_passed = False
            failure_reason = str(e)
            break

        # Kısa göster
        display_response = response[:200] + "..." if len(response) > 200 else response
        print(f"   🤖 AI: {display_response}")

        # Geçmişe ekle
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": response})
        all_responses.append(response)

        # Kontrol gerekiyorsa yap
        if expected:
            found = False
            for keyword in expected:
                if keyword.lower() in response.lower():
                    found = True
                    print(f"   ✅ Beklenen '{keyword}' bulundu!")
                    break

            if not found:
                test_passed = False
                failure_reason = f"Beklenen kelimeler bulunamadı: {expected}"
                print(f"   ❌ BAŞARISIZ: {failure_reason}")

    return {
        "name": test["name"],
        "passed": test_passed,
        "failure_reason": failure_reason,
        "responses": all_responses
    }


async def main():
    print("\n" + "="*60)
    print("🧠 HAFIZA VE BAĞLAM TESTİ BAŞLIYOR")
    print("="*60)

    # AI başlat
    ai = PersonalAI(user_id="hafiza_test")

    results = []

    for test in MEMORY_TESTS:
        try:
            result = await run_memory_test(ai, test)
            results.append(result)
            # Her test arasında sohbeti sıfırla
            ai.reset_conversation()
        except Exception as e:
            print(f"\n❌ TEST HATASI: {test['name']} - {e}")
            results.append({
                "name": test["name"],
                "passed": False,
                "failure_reason": str(e)
            })

    # Özet rapor
    print("\n" + "="*60)
    print("📊 HAFIZA TESTİ SONUÇLARI")
    print("="*60)

    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed

    for r in results:
        status = "✅" if r.get("passed", False) else "❌"
        print(f"{status} {r['name']}")
        if not r.get("passed", False) and r.get("failure_reason"):
            print(f"   → Sebep: {r['failure_reason']}")

    print(f"\n📈 Toplam: {passed}/{len(results)} başarılı")

    if failed == 0:
        print("\n🎉 TÜM HAFIZA TESTLERİ GEÇTİ!")
    else:
        print(f"\n⚠️ {failed} test başarısız!")

    # Sonuçları kaydet
    with open("hafiza_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n📁 Detaylı sonuçlar: hafiza_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())
