import pandas as pd
import numpy as np
import pickle
import re
import tensorflow as tf

tf.keras.backend.clear_session()
print(tf.__version__)
print(tf.config.list_physical_devices())

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping



# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data.csv")

print("Jumlah data:", len(df))

print(df.head())
print(df["label"].value_counts())
print(df["label"].unique())

# ==========================
# Preprocessing
# ==========================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["text"] = df["text"].apply(clean_text)

print("Jumlah text:", len(df))

print("\nContoh text:")
for i in range(5):
    print(df["text"].iloc[i])

print("\nJumlah kata unik:")
all_words = " ".join(df["text"]).split()
print(len(set(all_words)))

print("\n10 kata pertama:")
print(list(set(all_words))[:10])

# ==========================
# Feature & Target
# ==========================

X = df["text"]
y = df["label"].astype("float32")

# ==========================
# Tokenizer
# ==========================

MAX_WORDS = 5000
MAX_LEN = 50

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post"
)

print("X_pad shape:", X_pad.shape)
print("X_pad dtype:", X_pad.dtype)

print("Min token:", X_pad.min())
print("Max token:", X_pad.max())

print(X_pad[:3])

print("Jumlah kata dalam vocab:", len(tokenizer.word_index))

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

print("Unique labels:", np.unique(y_train))
print("Jumlah label 0:", np.sum(y_train == 0))
print("Jumlah label 1:", np.sum(y_train == 1))

# ==========================
# Build LSTM Model
# ==========================

# model = Sequential([
#     Embedding(
#         input_dim=MAX_WORDS,
#         output_dim=128
#     ),

#     LSTM(128),

#     Dense(64, activation="relu"),

#     Dropout(0.5),

#     Dense(1, activation="sigmoid")
# ])

model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_length=MAX_LEN
    ),

    LSTM(
        128,
        dropout=0.2,
        recurrent_dropout=0.2
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, MAX_LEN))

model.summary()

# ==========================
# Training
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

import time

start = time.time()
print("Training dimulai:", start)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


end = time.time()

print("Training selesai")
print("Durasi:", end - start)

# ==========================
# Evaluation
# ==========================

loss, acc = model.evaluate(X_test, y_test)

print(f"\nAccuracy: {acc*100:.2f}%")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer berhasil disimpan!")

# ==========================
# Save Model
# ==========================

model.save("sentiment_lstm.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nModel berhasil disimpan!")