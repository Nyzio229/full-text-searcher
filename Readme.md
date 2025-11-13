# Full-Text Searcher - README

## Projekt: Wyszukiwarka Full-Text dla Wikipedii

Projekt implementuje kompletny pipeline crawlowania, indeksowania i wyszukiwania stron internetowych (fokus na Wikipedię) z wykorzystaniem algorytmu TF-IDF.

---

## Architektura

### 1. **Crawler** (`crawler_indexer.py`)
- **Cel**: Pobieranie stron internetowych z określonych domen
- **Cechy**:
  - Przechodzenie przez strony do wybranej głębokości (0-5 poziomów)
  - Obsługa pliku `robots.txt` (respektowanie ograniczeń dla botów)
  - Rate limiting - unika zbyt częstych żądań do tego samego hosta
  - Filtrowanie domeny - opcjonalnie ograniczy crawlowanie do wybranych domen
  - Synchroniczny crawler (`SyncCrawler`) - alternatywie Apache Beam dla dużych zbiorów

### 2. **Indekser** (`crawler_indexer.py`)
- **Cel**: Budowanie indeksu TF-IDF z pobranych dokumentów
- **Funkcjonalność**:
  - Tokenizacja i normalizacja tekstu
  - Obliczanie TF-IDF dla każdego terminu w dokumencie
  - Generowanie trzech plików wyjściowych:
    - `docs.jsonl` - lista wszystkich pobranych dokumentów
    - `tfidf_index.jsonl` - pełny indeks TF-IDF (jedna linia per dokument)
    - `tfidf_index.json` - kompaktowy format (mapa ID → dane)

### 3. **Wyszukiwarka** (`searcher.py`)
- **Cel**: Wyszukiwanie w indeksie względem zapytania użytkownika
- **Funkcjonalność**:
  - Wczytanie indeksu z pliku JSON/JSONL
  - Tokenizacja zapytania użytkownika
  - Obliczanie podobieństwa cosinusowego między wektorem zapytania a dokumentami
  - Zwracanie top-k wyników posortowanych po relevancji

### 4. **Utilities** (`webutils.py`)
- Pobieranie stron HTML (`fetch`)
- Ekstrakcja tekstu i linków z HTML (`extract_text_and_links`)
- Zarządzanie `robots.txt` (`RobotsManager`)
- Rate limiting per-host (`HostRateLimiter`)

---

## Instalacja

### Wymagania
- Python 3.7+
- Biblioteki:
  ```bash
  pip install requests beautifulsoup4 tldextract robotexclusionrulesparser
  ```

### Opcjonalne
- Apache Beam (dla dużych zbiorów):
  ```bash
  pip install apache-beam
  ```

---

## Użycie

### 1. Crawlowanie i indeksowanie

```bash
python crawler_indexer.py --seeds https://en.wikipedia.org/wiki/Poland --out out/wiki_poland --depth 0  
```

**Parametry**:
- `--seeds`: URL strony początkowej (lub lista URLs)
- `--out`: Katalog wyjściowy na indeks
- `--depth`: Głębokość crawlowania (0-5, domyślnie 2)
- `--min-interval`: Minimalny interwał między żądaniami do tego samego hosta (sekundy)
- `--allowed-domains`: Opcjonalne - ogranicza crawlowanie do wybranych domen

**Wynik**:
- `docs.jsonl` - lista dokumentów
- `tfidf_index.jsonl` - indeks TF-IDF
- `tfidf_index.json` - kompaktowy format indeksu

### 2. Wyszukiwanie

```bash
python searcher.py --index out/wiki_poland/tfidf_index.jsonl --query "population history poland" --debug
```

**Parametry**:
- `--index`: Ścieżka do pliku indeksu
- `--query`: Zapytanie wyszukiwawcze
- `--top-k`: Liczba zwracanych wyników (domyślnie 10)
- `--debug`: Wyświetl informacje diagnostyczne

**Wynik**:
```
Rank | Score   | URL | Title
-----|---------|-----|------
1    | 0.8542  | https://... | Artykuł
2    | 0.7231  | https://... | Artykuł
...
```

---

## Struktury danych

### `docs.jsonl`
Jedna linia = jeden dokument (JSONL format):
```json
{"id": "uuid5-hash", "url": "https://...", "title": "...", "text": "..."}
```

### `tfidf_index.jsonl`
Jedna linia = TF-IDF wektor dokumentu:
```json
{"id": "uuid5-hash", "url": "https://...", "title": "...", "tfidf": {"word1": 0.123, "word2": 0.456, ...}}
```

### `tfidf_index.json`
Kompaktowy format (mapa ID → dane):
```json
{
  "uuid5-hash1": {"url": "https://...", "title": "...", "tfidf": {...}},
  "uuid5-hash2": {"url": "https://...", "title": "...", "tfidf": {...}}
}
```

---

## Algorytm TF-IDF

**TF (Term Frequency)**:
```
TF(t, d) = liczba wystąpień termu t w dokumencie d / całkowita liczba termów w d
```

**IDF (Inverse Document Frequency)**:
```
IDF(t) = log(liczba wszystkich dokumentów / liczba dokumentów zawierających t)
```

**TF-IDF**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Wyszukiwanie (Cosine Similarity)**:
```
similarity(query, doc) = cos(angle) = (query_vector · doc_vector) / (|query_vector| × |doc_vector|)
```

---

## Cechy bezpieczeństwa

1. **Respektowanie robots.txt**
   - Automatyczne pobieranie i parsowanie `robots.txt`
   - Fallback: domyślnie zezwala na dostęp jeśli `robots.txt` niedostępny

2. **Rate Limiting**
   - Per-host minimalna przerwa między żądaniami
   - Domyślnie 1 sekunda między żądaniami do tego samego hosta

3. **Timeout**
   - Żądania HTTP mają timeout (domyślnie 10 sekund)
   - Unika zawieszania się na wolnych serwerach

4. **User-Agent**
   - Identyfikuje się jako bot (edukacyjny): `BeamWikiBot/0.1`
   - Niektóre serwery mogą blokować boty

---

## Troubleshooting

### Problem: `docs.jsonl` jest pusty

**Przyczyny**:
1. Brak połączenia z internetem
2. Serwer blokuje User-Agent
3. Niepoprawny URL seed
4. Błąd parsowania HTML

**Rozwiązanie**:
- Uruchom z parametrem `--debug` (jeśli dostępny)
- Zwiększ `--min-interval` (np. 2.0)
- Zmień User-Agent w `webutils.py`
- Sprawdź czy serwer jest dostępny: `curl -I <URL>`

### Problem: Błąd importu `webutils`

**Rozwiązanie**:
- Upewnij się że plik `webutils.py` istnieje w tym samym katalogu
- Sprawdź nazwę pliku (możliwa literówka: `webutlis.py` → `webutils.py`)

### Problem: Powolne crawlowanie

**Rozwiązanie**:
- Zmniejsz `--depth` (np. z 2 na 1)
- Zmniejsz `--min-interval` (np. z 1.0 na 0.1) - z ostrożnością!
- Ograncz `--allowed-domains` do wybranych domen

---

## Ograniczenia

1. **Synchroniczny crawler** - wolniejszy niż Apache Beam dla dużych zbiorów
2. **Pełne pobieranie HTML** - bez inkrementalnego crawlowania
3. **Brak cache'owania** - każdorazowe pełne przetwarzanie
4. **Tokenizacja** - prosta, bez uwzględniania lingwistyki
5. **Brak obsługi języków nie-łacińskich** - fokus na angielskim


## Licencja

Projekt edukacyjny (Uniwersytet Mikołaja Kopernika, Toruń)