"""Deep Learning Firewall — a character-level neural web application firewall.

Detects malicious HTTP requests (SQL injection, XSS, path traversal, command
injection, and other anomalies) by reading the raw request character-by-character
with a TextCNN, rather than matching fixed signatures.
"""

__version__ = "1.0.0"
