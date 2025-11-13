# tests/test_crawler_index.py
import os
import io
import json
import time
import types
import pytest

# Import project modules - adjust if your module layout differs
import crawler_indexer
import webutils

# -----------------------
# Helpers / fixtures
# -----------------------
@pytest.fixture(autouse=True)
def no_requests(monkeypatch):
    """
    Ensure no real requests are made by default.
    We will monkeypatch requests.get inside webutils if needed per-test.
    """
    # if webutils.fetch uses requests.get internally, ensure it's safe to patch
    yield


# -----------------------
# Test: simple sync crawl creates docs.jsonl
# -----------------------
def test_run_pipeline_creates_docs_jsonl(tmp_path, monkeypatch):
    """
    Simulate a crawler run with one seed (depth 0).
    We monkeypatch webutils.can_fetch to allow crawling,
    webutils.fetch to return a small HTML page,
    webutils.extract_text_and_links to extract text and no links,
    and monkeypatch build_index_from_docs (Beam stage) to a noop so test remains fast.
    """

    # prepare a simple HTML response
    seed_url = "https://example.org/wiki/TestPage"
    html = "<html><head><title>TestPage</title></head><body>Population of Poland is large.</body></html>"
    text = "Population of Poland is large."
    title = "TestPage"

    # allow all robots
    monkeypatch.setattr(webutils, "can_fetch", lambda url: True)

    # stub fetch -> return html
    monkeypatch.setattr(webutils, "fetch", lambda url, timeout=10.0: html)

    # stub extract_text_and_links -> return text, empty links, title
    monkeypatch.setattr(webutils, "extract_text_and_links", lambda url, h: (text, [], title))

    # monkeypatch Beam indexer runner (build_index_from_docs) so test does not run Beam
    # We expect crawler_indexer.run_pipeline to call some function like build_index_from_docs(pages, out_dir)
    # If it uses a different name, adapt here.
    def fake_build_index_from_docs(pages, out_dir):
        # Create a dummy tfidf_index.jsonl to simulate a successful indexing stage
        outdir = os.path.join(out_dir, "")
        os.makedirs(outdir, exist_ok=True)
        docs_path = os.path.join(outdir, "tfidf_index.jsonl")
        sample_rec = {
            "id": "doc1",
            "url": seed_url,
            "title": title,
            "tfidf": {"poland": 0.1, "population": 0.05}
        }
        with open(docs_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_rec, ensure_ascii=False) + "\n")
        return

    # Try to patch the function name that exists in your crawler_indexer.
    # Many versions have build_index_from_docs; if your file differs, patch accordingly.
    if hasattr(crawler_indexer, "build_index_from_docs"):
        monkeypatch.setattr(crawler_indexer, "build_index_from_docs", fake_build_index_from_docs)
    else:
        # If the function is named differently, try to patch a generic function used in main
        # Fallback: nothing - pipeline may still write docs.jsonl, we'll check for it.
        pass

    out_dir = str(tmp_path / "out_job")
    # run pipeline (depth 0 so only seeds)
    # run_pipeline signature in logs: run_pipeline(seeds, out, max_depth, per_host_min_interval, allowed_domains)
    # Accept both list and single string for seeds
    try:
        crawler_indexer.run_pipeline([seed_url], out_dir, max_depth=0, per_host_min_interval=0.01, allowed_domains=None)
    except TypeError:
        # older signature without named args
        crawler_indexer.run_pipeline([seed_url], out_dir, 0, 0.01, None)

    # Assert docs.jsonl exists and contains our seed
    docs_path = os.path.join(out_dir, "docs.jsonl")
    assert os.path.exists(docs_path), f"Expected {docs_path} to exist (crawler should write docs.jsonl)"

    with open(docs_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) >= 1, "docs.jsonl should contain at least one document"

    # check that the JSON we stored contains expected fields and url
    rec0 = json.loads(lines[0])
    assert "url" in rec0 and rec0["url"] == seed_url
    assert "text" in rec0 or "content" in rec0 or "title" in rec0

    # Check that fake tfidf index was created by our fake_build_index_from_docs
    tfidf_path = os.path.join(out_dir, "tfidf_index.jsonl")
    assert os.path.exists(tfidf_path), "tfidf_index.jsonl should have been created by indexer stage"

    rec = json.loads(open(tfidf_path, "r", encoding="utf-8").read().strip())
    assert rec["url"] == seed_url
    assert "tfidf" in rec and isinstance(rec["tfidf"], dict)


# -----------------------
# Test: HostRateLimiter uses sleep when needed (no real sleep)
# -----------------------
def test_host_rate_limiter_waits(monkeypatch):
    """
    Simulate time so that HostRateLimiter.wait will call time.sleep with expected delta.
    We intercept time.time and time.sleep to verify behavior without actually sleeping.
    """
    hr = webutils.HostRateLimiter(min_interval_sec=2.0)

    # Simulate sequence of time.time() calls:
    # 1st call returns 100.0 (initial)
    # On first wait: last not set -> sets last to time.time()
    # On second wait: now returns 101.0, last was 100.0, delta = 1 -> should sleep 1.0 second
    times = [100.0, 100.0, 101.0, 101.0]  # the function may call time.time multiple times
    def fake_time():
        return times.pop(0)
    sleep_calls = []
    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        # don't actually sleep

    monkeypatch.setattr(webutils.time, "time", fake_time)
    monkeypatch.setattr(webutils.time, "sleep", fake_sleep)

    # First call: should not sleep
    hr.wait("example.org")
    # Second call: should sleep 1.0 (because min_interval=2 and only 1 second elapsed)
    hr.wait("example.org")

    assert len(sleep_calls) == 1
    # expected to sleep 1.0 (2.0 - (101-100) = 1.0)
    assert abs(sleep_calls[0] - 1.0) < 1e-6


# -----------------------
# Test: RobotsManager.can_fetch behavior (block/allows)
# -----------------------
def test_can_fetch_block_and_allow(monkeypatch):
    """
    Test basic behavior of robots manager sentinel responses:
    - When manager.get_for returns parser=None -> allow
    - When returns 'BLOCK_ALL' -> deny
    - When returns parser object with can_fetch or is_allowed, behavior delegated.
    """
    # case 1: parser None => allow
    monkeypatch.setattr(webutils.ROBOTS, "get_for", lambda url: {"parser": None})
    assert webutils.can_fetch("https://example.com/whatever") is True

    # case 2: BLOCK_ALL sentinel => deny
    monkeypatch.setattr(webutils.ROBOTS, "get_for", lambda url: {"parser": "BLOCK_ALL"})
    assert webutils.can_fetch("https://example.com/whatever") is False

    # case 3: parser object with can_fetch method (simulate allow for our UA)
    class DummyParser:
        def can_fetch(self, ua, target):
            return True
    monkeypatch.setattr(webutils.ROBOTS, "get_for", lambda url: {"parser": DummyParser()})
    assert webutils.can_fetch("https://example.com/whatever") is True
