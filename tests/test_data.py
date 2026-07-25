from dlfw.data import (
    ANOMALOUS,
    NORMAL,
    parse_csic_file,
    request_to_text,
    synthetic_dataset,
)


def test_request_to_text_url_decodes():
    # %27 -> ' so the model reads the real injection payload
    t = request_to_text("GET", "/p?id=1%27%20OR%20%271%27=%271")
    assert "or '1'='1" in t
    assert t.startswith("get /p")


def test_request_to_text_includes_body():
    t = request_to_text("POST", "/login", "user=admin&pass=x")
    assert "user=admin" in t


def test_synthetic_balanced_and_labeled():
    texts, labels = synthetic_dataset(200)
    assert len(texts) == len(labels) == 200
    assert set(labels) == {NORMAL, ANOMALOUS}
    assert abs(sum(labels) - 100) <= 1  # roughly balanced


def test_synthetic_attacks_look_malicious():
    texts, labels = synthetic_dataset(400)
    anomalous = [t for t, y in zip(texts, labels) if y == ANOMALOUS]
    joined = " ".join(anomalous)
    assert any(sig in joined for sig in ["union select", "<script>", "../", "drop table"])


def test_parse_csic(tmp_path):
    sample = (
        "GET /shop/index.jsp?id=2 HTTP/1.1\n"
        "Host: localhost\n"
        "\n"
        "POST /shop/login.jsp HTTP/1.1\n"
        "Host: localhost\n"
        "Content-Length: 20\n"
        "\n"
        "user=admin&pass=1234\n"
    )
    p = tmp_path / "normalTrafficTraining.txt"
    p.write_text(sample, encoding="latin-1")
    rows = parse_csic_file(p, NORMAL)
    assert len(rows) == 2
    assert all(y == NORMAL for _, y in rows)
    assert "/shop/index.jsp" in rows[0][0]
