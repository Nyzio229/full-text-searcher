from __future__ import annotations

import re
import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import apache_beam as beam
from apache_beam import pvalue as beam_pvalue

import nltk
from nltk.data import find as nltk_find
from nltk.corpus import stopwords

try:
    STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP = set(stopwords.words("english"))

try:
    nltk_find("corpora/wordnet")
    from nltk.stem import WordNetLemmatizer
    _HAS_WORDNET = True
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    try:
        nltk_find("corpora/wordnet")
        from nltk.stem import WordNetLemmatizer
        _HAS_WORDNET = True
    except LookupError:
        _HAS_WORDNET = False

# Prostszy wzorzec: tylko litery eng, rozbijamy np. "can't" -> "can", "t"
_TOK = re.compile(r"[A-Za-z]+")


def lex(text: str) -> List[str]:
    """Ekstrakcja słów (lowercase, A-Z)."""
    return _TOK.findall((text or "").lower())


def normalize_en(tokens: Iterable[str], lemmatize: bool = True, min_len: int = 2) -> List[str]:
    """Filtr EN: min długość, stopwords, opcjonalnie lematyzacja (WordNet)."""
    toks = [t for t in tokens if len(t) >= min_len and t not in STOP]
    if lemmatize and _HAS_WORDNET:
        lem = WordNetLemmatizer()
        toks = [lem.lemmatize(t) for t in toks]
    return toks


def term_freq(tokens: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


# ---------- Beam DoFns ----------

class LoadText(beam.DoFn):
    """Wejście: (doc_id, path) → wyjście: {id, path, text}."""
    def process(self, pair: Tuple[str, str]):
        doc_id, path = pair
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            txt = ""
        yield {"id": doc_id, "path": path, "text": txt}


class TermStatsFn(beam.DoFn):
    """
    Z dokumentu robi:
    - wyjście 'tf': (doc_id, path, {token:count}, total_count)
    - wyjście 'df': (token, doc_id) do liczenia DF
    """
    def setup(self):
        # Przyspieszenie: trzymamy lematyzer w workerze
        self._lem = WordNetLemmatizer() if _HAS_WORDNET else None

    def process(self, doc: Dict):
        tokens = normalize_en(lex(doc["text"]), lemmatize=bool(self._lem))
        counts = Counter(tokens)
        total = sum(counts.values())

        # TF do dalszych obliczeń
        yield beam.pvalue.TaggedOutput("tf", (doc["id"], doc["path"], dict(counts), total))
        # (token, doc_id) do DF (później Distinct → Count)
        for tok in counts.keys():
            yield beam.pvalue.TaggedOutput("df", (tok, doc["id"]))


class MakeTfIdfFn(beam.DoFn):
    """Z TF + (DF, N) robi słownik TF-IDF (bez normalizacji do 1 — to po stronie searchera)."""
    def process(self, tf_row, df_map: Dict[str, int], N: int):
        doc_id, path, counts, total = tf_row
        if total <= 0:
            yield {"id": doc_id, "path": path, "tfidf": {}}
            return

        # TF (częstość względna)
        tf_rel = {t: c / float(total) for t, c in counts.items()}

        # Smoothed IDF: log(1 + N/df)
        tfidf = {}
        for t in counts.keys():
            df = df_map.get(t, 0) or 1
            idf = math.log(1.0 + (N / float(df)))
            tfidf[t] = tf_rel[t] * idf

        yield {"id": doc_id, "path": path, "tfidf": tfidf}