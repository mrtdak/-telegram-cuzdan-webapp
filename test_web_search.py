# -*- coding: utf-8 -*-
"""
Web Search Test Script
Farkli kategorilerde sorgulari hizlica test et
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from web_search import WebSearch

def test_search():
    searcher = WebSearch()

    # Farkli kategorilerde test sorgulari
    test_queries = [
        # Tarim
        "mantar nasil uretilir",
        # Teknoloji
        "Python nedir",
        # Finans
        "bitcoin son fiyat",
        # Haber
        "Turkiye son depremler",
        # Genel bilgi
        "Istanbul nufusu kac",
    ]

    print("=" * 60)
    print("WEB SEARCH TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\n{'-' * 60}")
        print(f"SORGU: {query}")
        print("-" * 60)

        # quick_answer ile test
        answer = searcher.quick_answer(query)
        if answer:
            print(f"\nCEVAP:\n{answer[:500]}...")
        else:
            print("\nCEVAP: Sonuc yok")

        # Detayli sonuclar icin
        result = searcher.search(query, max_results=3)
        if result.get("results"):
            print(f"\nKAYNAKLAR:")
            for i, r in enumerate(result["results"], 1):
                title = r.get('title', '')[:50]
                url = r.get('url', '')
                print(f"   {i}. {title}")
                print(f"      {url}")

        print()

    # Interaktif mod
    print("\n" + "=" * 60)
    print("INTERAKTIF MOD (cikis icin 'q')")
    print("=" * 60)

    while True:
        try:
            query = input("\nSorgu: ").strip()
        except EOFError:
            break
        if query.lower() == 'q':
            print("Cikis...")
            break
        if query:
            print("\nAraniyor...")
            answer = searcher.quick_answer(query)
            print(f"\nCEVAP:\n{answer}")

            # Kaynaklari da goster
            result = searcher.search(query, max_results=3)
            if result.get("results"):
                print(f"\nKAYNAKLAR:")
                for i, r in enumerate(result["results"], 1):
                    title = r.get('title', '')[:50]
                    url = r.get('url', '')
                    print(f"   {i}. {title}")
                    print(f"      {url}")


if __name__ == "__main__":
    test_search()
