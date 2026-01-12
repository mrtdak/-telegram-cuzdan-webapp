"""
Tavily API ile Web Arama Modulu
Temiz, guvenilir sonuclar
"""

import requests
from typing import List, Dict, Optional
import os

# Tavily API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")


def fix_encoding(text: str) -> str:
    """Bozuk UTF-8 karakterleri düzelt"""
    if not text:
        return text
    try:
        # Latin-1 olarak encode edilmiş UTF-8'i düzelt
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Düzeltemiyorsa olduğu gibi döndür
        return text


class WebSearch:
    """Tavily tabanli internet arama sinifi."""

    def __init__(self):
        self.api_key = TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"
        self.timeout = 30

    def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> Dict:
        """
        Ana arama fonksiyonu.

        Args:
            query: Arama sorgusu
            max_results: Maksimum sonuc sayisi (1-10)
            search_depth: "basic" (hizli) veya "advanced" (detayli, 2 kredi)

        Returns:
            Arama sonuclari
        """
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": max_results,
                    "auto_parameters": True
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            return {
                "query": query,
                "answer": fix_encoding(data.get("answer", "")),
                "results": [
                    {
                        "title": fix_encoding(r.get("title", "")),
                        "url": r.get("url", ""),
                        "content": fix_encoding(r.get("content", "")),
                        "score": r.get("score", 0)
                    }
                    for r in data.get("results", [])
                ]
            }

        except requests.exceptions.RequestException as e:
            return {
                "query": query,
                "answer": "",
                "results": [],
                "error": str(e)
            }

    def quick_answer(self, query: str) -> str:
        """
        Hizli cevap - direkt kullanilabilir metin doner.

        Args:
            query: Soru/sorgu

        Returns:
            Cevap metni
        """
        result = self.search(query, max_results=3, search_depth="basic")

        if result.get("error"):
            return f"Arama hatasi: {result['error']}"

        # Öncelik 1: Tavily'nin AI-generated cevabı (temiz ve öz)
        if result.get("answer"):
            return result["answer"]

        # Öncelik 2: AI cevabı yoksa ilk sonucun içeriğini kullan (temizse)
        if result.get("results") and len(result["results"]) > 0:
            first = result["results"][0]
            content = first.get("content", "")
            if content and "�" not in content:
                return content[:500]  # Max 500 karakter

        return "Sonuc bulunamadi."

    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Haber arama.

        Args:
            query: Haber konusu
            max_results: Maksimum sonuc

        Returns:
            Haber listesi
        """
        result = self.search(f"{query} haber guncel", max_results=max_results)
        return result.get("results", [])

    def get_news_summary(self, topic: str) -> str:
        """
        Belirli bir konu hakkinda haber ozeti.

        Args:
            topic: Haber konusu

        Returns:
            Haber ozeti
        """
        result = self.search(f"{topic} son haberler", max_results=5, search_depth="advanced")

        if result.get("error"):
            return f"Haber alinamadi: {result['error']}"

        output = []
        output.append(f"'{topic.upper()}' HAKKINDA GUNCEL BILGILER")
        output.append("=" * 50)

        if result.get("answer"):
            output.append(f"\n{result['answer']}")
            output.append("")

        if result.get("results"):
            output.append("\nKaynaklar:")
            for i, r in enumerate(result["results"], 1):
                output.append(f"\n{i}. {r['title']}")
                if r.get("content"):
                    output.append(f"   {r['content'][:150]}...")
                output.append(f"   [{r['url']}]")

        return "\n".join(output)


def main():
    """Test fonksiyonu."""
    searcher = WebSearch()

    print("=" * 60)
    print("Tavily API Arama Testi")
    print("=" * 60)

    # Test sorgusu
    test_query = "Python 3.12 yeni ozellikler"
    print(f"\nSorgu: {test_query}")
    print("-" * 40)

    answer = searcher.quick_answer(test_query)
    print(answer)

    print("\n" + "=" * 60)
    print("Interaktif Mod (cikis icin 'q')")
    print("=" * 60)

    while True:
        query = input("\nSorgu: ").strip()
        if query.lower() == 'q':
            break
        if query:
            print("\nAraniyor...")
            answer = searcher.quick_answer(query)
            print(f"\n{answer}")


if __name__ == "__main__":
    main()
