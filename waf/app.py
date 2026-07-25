"""Flask WAF gateway.

A thin web application firewall that scores every incoming request with the
trained model and blocks the malicious ones. Two ways to use it:

- `POST /inspect` — score an arbitrary request (JSON: method, url, body).
- `ANY /proxy/<path>` — a demo endpoint that a real reverse proxy would front:
  the request is scored before it would be forwarded; malicious ones get 403.

Blocked requests are logged with their source IP and shown on `/` (dashboard).

    python -m dlfw.train --synthetic        # produce a model first
    python waf/app.py                        # http://localhost:5000
"""

from __future__ import annotations

import datetime as dt
from collections import deque

from flask import Flask, jsonify, render_template_string, request

from dlfw.config import DEFAULT
from dlfw.data import request_to_text
from dlfw.detector import FirewallDetector

app = Flask(__name__)
BLOCKED: deque = deque(maxlen=100)  # recent blocked requests

try:
    DET = FirewallDetector.load(DEFAULT)
except FileNotFoundError:
    DET = None  # dashboard still loads; inspect returns a helpful error


def _inspect(method: str, url: str, body: str = "") -> tuple[bool, float]:
    text = request_to_text(method, url, body, dict(request.headers))
    score = DET.score(text)
    return score >= DET.threshold, score


@app.post("/inspect")
def inspect():
    if DET is None:
        return jsonify(error="no model — run `python -m dlfw.train` first"), 503
    data = request.get_json(silent=True) or {}
    malicious, score = _inspect(data.get("method", "GET"), data.get("url", "/"), data.get("body", ""))
    if malicious:
        _record(data.get("url", "/"), score)
    return jsonify(malicious=malicious, score=round(score, 4))


@app.route("/proxy/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    if DET is None:
        return jsonify(error="no model loaded"), 503
    body = request.get_data(as_text=True)
    malicious, score = _inspect(request.method, "/" + path + "?" + request.query_string.decode(), body)
    if malicious:
        _record("/" + path, score)
        return jsonify(blocked=True, reason="request classified as malicious", score=round(score, 4)), 403
    return jsonify(allowed=True, score=round(score, 4))


def _record(url: str, score: float) -> None:
    BLOCKED.appendleft({
        "time": dt.datetime.now().strftime("%H:%M:%S"),
        "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
        "url": url[:120],
        "score": round(score, 3),
    })


@app.get("/")
def dashboard():
    return render_template_string(_DASH, blocked=list(BLOCKED), ready=DET is not None)


_DASH = """
<!doctype html><meta charset=utf-8><title>DL Firewall</title>
<style>
 body{font-family:ui-monospace,Menlo,monospace;background:#0c0e13;color:#e9eaee;margin:0;padding:32px}
 h1{font-weight:600} .badge{padding:3px 10px;border-radius:6px;font-size:12px}
 .ok{background:#12351f;color:#7ee2a8}.no{background:#3a1620;color:#f4a3b4}
 table{width:100%;border-collapse:collapse;margin-top:18px}
 th,td{border-bottom:1px solid #222632;padding:8px 10px;text-align:left;font-size:13px}
 th{color:#8b93ff}.s{color:#f4a3b4}
</style>
<h1>Deep Learning Firewall
 <span class="badge {{ 'ok' if ready else 'no' }}">{{ 'model loaded' if ready else 'no model' }}</span></h1>
<p>Blocked requests: {{ blocked|length }}</p>
<table><tr><th>time</th><th>source ip</th><th>url</th><th>score</th></tr>
{% for b in blocked %}<tr><td>{{b.time}}</td><td>{{b.ip}}</td><td>{{b.url}}</td><td class=s>{{b.score}}</td></tr>{% endfor %}
{% if not blocked %}<tr><td colspan=4 style="color:#6a7180">no malicious requests blocked yet</td></tr>{% endif %}
</table>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
