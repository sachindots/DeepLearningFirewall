"""The neural model — a character-level TextCNN.

An embedding turns each character into a dense vector; parallel 1-D convolutions
with different kernel widths act as learnable n-gram detectors (a width-5 filter
can learn the shape of `union`, `<scri`, `../..`, etc.); global max-pooling keeps
the strongest signal per filter; a small dense head produces a single
maliciousness probability.

This is a stronger, simpler design than a single-kernel CNN: the multi-width
convolutions capture attack tokens of varying length in one pass.
"""

from __future__ import annotations


def build_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 64,
    kernel_sizes: tuple[int, ...] = (3, 5, 7),
    filters: int = 128,
    dropout: float = 0.4,
    learning_rate: float = 1e-3,
):
    """Build and compile the TextCNN. Imports TensorFlow lazily."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(max_len,), dtype="int32", name="chars")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=False, name="embed")(inp)

    conv_blocks = []
    for k in kernel_sizes:
        c = layers.Conv1D(filters, k, activation="relu", padding="same", name=f"conv_{k}")(x)
        c = layers.GlobalMaxPooling1D(name=f"pool_{k}")(c)
        conv_blocks.append(c)

    x = layers.Concatenate(name="concat")(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="malicious")(x)

    model = models.Model(inp, out, name="deep_learning_firewall")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
