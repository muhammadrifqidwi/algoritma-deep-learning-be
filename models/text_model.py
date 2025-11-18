from flask import Blueprint, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle

text_bp = Blueprint('text_bp', __name__)

tokenizer = None
model = None
max_sequence_len = 0

@text_bp.route('/train', methods=['POST'])
def train_model():
    global tokenizer, model, max_sequence_len

    data = request.get_json()
    text = data.get("text", "")
    model_type = data.get("model", "lstm")

    if not text.strip():
        return jsonify({"error": "Teks training tidak boleh kosong"}), 400

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))

    if model_type == "vanilla":
        model.add(SimpleRNN(128))
    elif model_type == "gru":
        model.add(GRU(128))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(128)))
    else:
        model.add(LSTM(128))

    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=30, verbose=0)

    model.save("text_model.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return jsonify({"message": f"Model {model_type.upper()} berhasil dilatih!"})


@text_bp.route('/generate', methods=['POST'])
def generate_text():
    global tokenizer, model, max_sequence_len

    data = request.get_json()
    seed_text = data.get("seed", "")
    model_type = data.get("model", "lstm")
    next_words = int(data.get("length", 20))

    if not os.path.exists("text_model.h5") or not os.path.exists("tokenizer.pkl"):
        return jsonify({"error": "Model belum dilatih"}), 400

    model = load_model("text_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return jsonify({"generated_text": seed_text})
