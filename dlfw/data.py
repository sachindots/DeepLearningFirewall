"""Data loading and request canonicalization.

Two sources are supported:

1. The **CSIC 2010** HTTP dataset (`normalTrafficTraining.txt`,
   `anomalousTrafficTest.txt`, …) — raw HTTP requests, one after another.
2. A **synthetic** generator used for tests, smoke runs, and demos when the
   full dataset isn't downloaded.

Every request is flattened to one canonical text string so the character model
sees the method, path, query, and body as a single sequence.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from urllib.parse import unquote_plus

NORMAL, ANOMALOUS = 0, 1


def request_to_text(method: str, url: str, body: str = "", headers: dict | None = None) -> str:
    """Flatten an HTTP request into a single normalized string.

    URL-decoded so the model reads the real payload (e.g. `%27` -> `'`), which
    is where injection attacks live.
    """
    parts = [f"{method.upper()} {unquote_plus(url)}"]
    if headers:
        for k in ("content-type", "cookie", "user-agent"):
            if k in {h.lower() for h in headers}:
                v = next(val for hk, val in headers.items() if hk.lower() == k)
                parts.append(f"{k}: {v}")
    if body:
        parts.append(unquote_plus(body))
    return " \n ".join(parts).lower()


# -- CSIC 2010 parser ----------------------------------------------------

_REQ_LINE = re.compile(r"^(GET|POST|PUT|DELETE|HEAD)\s+(\S+)\s+HTTP", re.I)


def parse_csic_file(path: str | Path, label: int) -> list[tuple[str, int]]:
    """Parse a CSIC 2010 traffic file into (text, label) pairs."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    out: list[tuple[str, int]] = []
    i, n = 0, len(lines)
    while i < n:
        m = _REQ_LINE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        method, url = m.group(1), m.group(2)
        # body is the last non-empty line of the request block (CSIC puts POST
        # bodies after the headers, separated by a blank line)
        j, body = i + 1, ""
        while j < n and lines[j].strip() and not _REQ_LINE.match(lines[j].strip()):
            j += 1
        # skip blank line, capture a trailing body line if present
        k = j + 1
        if k < n and lines[k].strip() and not _REQ_LINE.match(lines[k].strip()):
            body = lines[k].strip()
        out.append((request_to_text(method, url, body), label))
        i = j
    return out


def load_csic(data_dir: str | Path) -> tuple[list[str], list[int]]:
    """Load CSIC 2010 from a directory. Returns (texts, labels)."""
    d = Path(data_dir)
    files = {
        "normalTrafficTraining.txt": NORMAL,
        "normalTrafficTest.txt": NORMAL,
        "anomalousTrafficTest.txt": ANOMALOUS,
    }
    texts, labels = [], []
    for name, label in files.items():
        p = d / name
        if p.exists():
            for t, y in parse_csic_file(p, label):
                texts.append(t)
                labels.append(y)
    if not texts:
        raise FileNotFoundError(
            f"No CSIC files found in {d}. Expected {list(files)}. "
            "Download the CSIC 2010 HTTP dataset, or use synthetic_dataset()."
        )
    return texts, labels


# -- synthetic generator (no dataset needed) -----------------------------

_NORMAL_PATHS = ["/", "/index.jsp", "/products", "/search", "/login", "/cart", "/account/profile"]
_PARAMS = ["id", "q", "name", "page", "category", "user", "sort"]
_WORDS = ["shoes", "laptop", "book", "phone", "23", "john", "electronics", "asc", "1"]

_ATTACKS = [
    "' OR '1'='1", "1; DROP TABLE users--", "' UNION SELECT password FROM users--",
    "<script>alert(1)</script>", "../../../../etc/passwd", "%00../../etc/shadow",
    "; cat /etc/passwd", "| ls -la", "${jndi:ldap://x/a}", "1' AND SLEEP(5)--",
    "<img src=x onerror=alert(1)>", "../../windows/win.ini",
]


def synthetic_dataset(n: int = 600, seed: int = 7) -> tuple[list[str], list[int]]:
    """Balanced synthetic dataset of normal and anomalous requests."""
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n // 2):
        path = rng.choice(_NORMAL_PATHS)
        params = "&".join(f"{rng.choice(_PARAMS)}={rng.choice(_WORDS)}" for _ in range(rng.randint(0, 3)))
        url = f"{path}?{params}" if params else path
        method = rng.choice(["GET", "GET", "POST"])
        body = f"{rng.choice(_PARAMS)}={rng.choice(_WORDS)}" if method == "POST" else ""
        texts.append(request_to_text(method, url, body))
        labels.append(NORMAL)
    for _ in range(n - n // 2):
        path = rng.choice(_NORMAL_PATHS)
        param = rng.choice(_PARAMS)
        payload = rng.choice(_ATTACKS)
        method = rng.choice(["GET", "POST"])
        if method == "GET":
            texts.append(request_to_text("GET", f"{path}?{param}={payload}"))
        else:
            texts.append(request_to_text("POST", path, f"{param}={payload}"))
        labels.append(ANOMALOUS)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]
