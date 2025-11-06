from __future__ import annotations

import time
import re
import threading
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional
from urllib.parse import urljoin, urlparse
import urllib.robotparser as urobot

import requests
from bs4 import BeautifulSoup

USER_AGENT = "BeamWikiBot/0.1 (+https://example.local/edu)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

class HostRateLimiter:
    """Simple per-host token bucket limiter shared per DoFn instance."""
    def __init__(self, min_interval_sec: float = 1.0):
        self.min_interval = float(min_interval_sec)
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}

    def wait(self, host: str):
        now = time.time()
        with self._lock:
            last = self._last.get(host, 0.0)
            delta = now - last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last[host] = time.time()

_robot_cache_lock = threading.Lock()
_robot_cache: Dict[str, urobot.RobotFileParser] = {}

def can_fetch(url: str, user_agent: str = USER_AGENT) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    with _robot_cache_lock:
        rfp = _robot_cache.get(origin)
        if not rfp:
            rfp = urobot.RobotFileParser()
            rfp.set_url(origin + "/robots.txt")
            try:
                rfp.read()
            except Exception:
                _robot_cache[origin] = rfp
                return False
            _robot_cache[origin] = rfp
    try:
        return rfp.can_fetch(user_agent, url)
    except Exception:
        return False

WHITESPACE_RE = re.compile(r"\s+", re.U)

def extract_text_and_links(url: str, html: str) -> Tuple[str, List[str], str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = WHITESPACE_RE.sub(" ", soup.get_text(" ")).strip()
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    return text, links, title

def fetch(url: str, timeout: float = 10.0) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if 200 <= resp.status_code < 300 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        return None
    except requests.RequestException:
        return None