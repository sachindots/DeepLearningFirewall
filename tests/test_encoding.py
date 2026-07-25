from dlfw.encoding import PAD_ID, UNK_ID, CharVocab


def test_encode_shape_and_pad():
    v = CharVocab(max_len=32)
    ids = v.encode("GET /")
    assert ids.shape == (32,)
    assert ids[-1] == PAD_ID          # padded
    assert (ids[:5] != PAD_ID).all()  # real chars first


def test_truncation():
    v = CharVocab(max_len=8)
    assert v.encode("x" * 100).shape == (8,)


def test_unknown_char_maps_to_unk():
    v = CharVocab(max_len=8)
    ids = v.encode("☃")  # snowman — outside printable ASCII
    assert ids[0] == UNK_ID


def test_batch():
    v = CharVocab(max_len=16)
    X = v.encode_batch(["GET /", "POST /login"])
    assert X.shape == (2, 16)


def test_roundtrip_save_load(tmp_path):
    v = CharVocab(max_len=64)
    p = tmp_path / "vocab.json"
    v.save(p)
    v2 = CharVocab.load(p)
    assert v2.max_len == 64 and v2.size == v.size
