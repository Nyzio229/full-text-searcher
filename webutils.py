from __future__ import annotations

import time
import re
import threading
import logging
from typing import List, Tuple, Dict, Optional, Literal
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from robotexclusionrulesparser import RobotExclusionRulesParser  # type: ignore
except Exception:
    RobotExclusionRulesParser = None

from urllib.robotparser import RobotFileParser

USER_AGENT = "BeamWikiBot/0.1 (+https://example.local/edu)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

logger = logging.getLogger(__name__)

# ---------------- Rate limiter ----------------
class HostRateLimiter:
    """Per-host minimal interval rate limiter."""
    def __init__(self, min_interval_sec: float = 1.0):
        self.min_interval = float(min_interval_sec)
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}

    def wait(self, host: str):
        """Wait until min interval elapsed since last request to host."""
        now = time.time()
        with self._lock:
            last = self._last.get(host, 0.0)
            delta = now - last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last[host] = time.time()

# ---------------- RobotsManager ----------------
RobotsMode = Literal["allow-on-error", "block-on-error", "strict"]

class RobotsManager:
    def __init__(self, timeout: float = 5.0, mode: RobotsMode = "allow-on-error"):
        self.cache: Dict[str, Dict] = {}
        self.timeout = float(timeout)
        if mode not in ("allow-on-error", "block-on-error", "strict"):
            raise ValueError("Invalid robots mode")
        self.mode = mode

    def set_mode(self, mode: RobotsMode):
        if mode not in ("allow-on-error", "block-on-error", "strict"):
            raise ValueError("Invalid robots mode")
        self.mode = mode

    def _fetch_raw(self, robots_url: str) -> Optional[str]:
        try:
            resp = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout, allow_redirects=True)
            if resp.status_code == 200 and resp.text and resp.text.strip():
                return resp.text
            logger.debug("Robots: non-200 or empty for %s -> %s", robots_url, resp.status_code)
            return None
        except Exception as e:
            logger.debug("Robots fetch error for %s : %s", robots_url, e)
            return None

    def _parse_crawl_delay(self, text: str) -> Optional[float]:
        for line in text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if k.strip().lower() == "crawl-delay":
                try:
                    return float(v.strip().split()[0])
                except Exception:
                    continue
        return None

    def _build_parser(self, scheme: str, host: str):
        robots_url = f"{scheme}://{host}/robots.txt"
        raw = self._fetch_raw(robots_url)

        if RobotExclusionRulesParser is not None:
            parser = RobotExclusionRulesParser()
            if raw:
                try:
                    parser.parse(raw)
                    self.cache[host] = {"parser": parser, "crawl_delay": self._parse_crawl_delay(raw)}
                    logger.info("Robots parsed (external) for %s", host)
                    return self.cache[host]
                except Exception as e:
                    logger.warning("RobotExclusion parse error for %s: %s", host, e)
            # raw missing or parse failed
            if self.mode == "allow-on-error":
                self.cache[host] = {"parser": None, "crawl_delay": None}
                logger.info("Robots missing or invalid for %s -> allow-on-error", host)
                return self.cache[host]
            else:
                self.cache[host] = {"parser": "BLOCK_ALL", "crawl_delay": None}
                logger.info("Robots missing/invalid for %s -> treated as blocked (mode=%s)", host, self.mode)
                return self.cache[host]

        # fallback to stdlib RobotFileParser
        rfp = RobotFileParser()
        rfp.set_url(robots_url)
        if raw:
            try:
                # use read() to populate; acceptable here
                rfp.read()
                crawl_delay = self._parse_crawl_delay(raw)
                self.cache[host] = {"parser": rfp, "crawl_delay": crawl_delay}
                logger.info("Robots parsed (stdlib) for %s", host)
                return self.cache[host]
            except Exception as e:
                logger.warning("RobotFileParser read error for %s: %s", host, e)
                if self.mode == "allow-on-error":
                    self.cache[host] = {"parser": None, "crawl_delay": None}
                    return self.cache[host]
                else:
                    self.cache[host] = {"parser": "BLOCK_ALL", "crawl_delay": None}
                    return self.cache[host]
        else:
            if self.mode == "allow-on-error":
                self.cache[host] = {"parser": None, "crawl_delay": None}
                return self.cache[host]
            else:
                self.cache[host] = {"parser": "BLOCK_ALL", "crawl_delay": None}
                return self.cache[host]

    def get_for(self, url: str) -> Dict:
        info = urlparse(url)
        host = info.netloc
        if host in self.cache:
            return self.cache[host]
        return self._build_parser(info.scheme, host)

    def _path_from_url(self, url: str) -> str:
        info = urlparse(url)
        path = info.path or "/"
        if info.query:
            path = path + "?" + info.query
        return path

    def can_fetch(self, url: str) -> bool:
        meta = self.get_for(url)
        parser = meta.get("parser")
        if parser is None:
            logger.debug("can_fetch: no parser cached -> allow-all for %s", url)
            return True
        if parser == "BLOCK_ALL":
            logger.debug("can_fetch: BLOCK_ALL sentinel -> deny for %s", url)
            return False

        path = self._path_from_url(url)

        def try_is_allowed(p, ua, target):
            try:
                if hasattr(p, "is_allowed"):
                    return bool(p.is_allowed(ua, target))
                if hasattr(p, "can_fetch"):
                    return bool(p.can_fetch(ua, target))
            except Exception as e:
                logger.debug("robots parser raised for target=%s ua=%s : %s", target, ua, e)
            return None

        ualist = [USER_AGENT, "*", "Mozilla/5.0"]
        for ua in ualist:
            res = try_is_allowed(parser, ua, url)
            if res is True:
                logger.debug("can_fetch: parser allowed (ua=%s, full url) %s", ua, url)
                return True
            if res is False:
                logger.debug("can_fetch: parser denied (ua=%s, full url) %s", ua, url)
            res2 = try_is_allowed(parser, ua, path)
            if res2 is True:
                logger.debug("can_fetch: parser allowed (ua=%s, path=%s) %s", ua, path, url)
                return True
            if res2 is False:
                logger.debug("can_fetch: parser denied (ua=%s, path=%s) %s", ua, path, url)

        logger.info(
            "can_fetch: parser denied for all tried agents for %s; mode=%s -> %s",
            url, getattr(self, "mode", "allow-on-error"),
            "allowing (mode=allow-on-error)" if getattr(self, "mode", "allow-on-error") == "allow-on-error" else "blocking"
        )
        return True if getattr(self, "mode", "allow-on-error") == "allow-on-error" else False

    def get_crawl_delay(self, url: str) -> Optional[float]:
        meta = self.get_for(url)
        return meta.get("crawl_delay")

ROBOTS = RobotsManager()

# ---------------- HTML helpers ----------------
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
        ct = resp.headers.get("Content-Type", "")
        if 200 <= resp.status_code < 300 and "text/html" in ct:
            return resp.text
        return None
    except requests.RequestException:
        return None

def can_fetch(url: str) -> bool:
    return ROBOTS.can_fetch(url)
