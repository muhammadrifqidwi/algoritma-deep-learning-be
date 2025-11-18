import os
import io
import base64
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam

MODEL_DIR = "models"
LSTM_PATH = os.path.join(MODEL_DIR, "bbca_lstm.keras")
GRU_PATH = os.path.join(MODEL_DIR, "bbca_gru.keras")
BIDIR_PATH = os.path.join(MODEL_DIR, "bbca_bidir.keras")
SCALER_MIN = os.path.join(MODEL_DIR, "scaler_min.npy")
SCALER_SCALE = os.path.join(MODEL_DIR, "scaler_scale.npy")

DATA_PATH = os.path.join("datasets", "BBCA_Train.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


class StockModels:
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.data = None
        self.scaler: MinMaxScaler | None = None

    def load_data(self):
        if self.df is not None and self.data is not None:
            return self.df, self.data
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        cols_lower = [c.lower() for c in df.columns]
        close_col = df.columns[cols_lower.index("close")] if "close" in cols_lower else df.columns[-1]
        self.df = df.reset_index(drop=True)
        self.data = self.df[close_col].values.reshape(-1, 1).astype("float32")
        return self.df, self.data

    def make_sequences(self, n_past: int = 60):
        _, data = self.load_data()
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(n_past, len(scaled)):
            X.append(scaled[i - n_past:i, 0])
            y.append(scaled[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y, scaled

    def build_lstm(self, input_shape):
        m = Sequential()
        m.add(Input(shape=input_shape))
        m.add(LSTM(128, return_sequences=True))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(LSTM(128))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(Dense(1))
        m.compile(optimizer=Adam(0.0005), loss="mse", metrics=["mean_squared_error"])
        return m

    def build_gru(self, input_shape):
        m = Sequential()
        m.add(Input(shape=input_shape))
        m.add(GRU(128, return_sequences=True))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(GRU(128))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(Dense(1))
        m.compile(optimizer=Adam(0.0005), loss="mse", metrics=["mean_squared_error"])
        return m

    def build_bidir(self, input_shape):
        m = Sequential()
        m.add(Input(shape=input_shape))
        m.add(Bidirectional(GRU(128, return_sequences=True)))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(Bidirectional(LSTM(128)))
        m.add(LayerNormalization())
        m.add(Dropout(0.25))
        m.add(Dense(1))
        m.compile(optimizer=Adam(0.0005), loss="mse", metrics=["mean_squared_error"])
        return m

    # ---------------- training ----------------
    def train_all(self, epochs: int = 50, batch_size: int = 32, n_past: int = 60, test_size: float = 0.2, verbose: int = 1):

        X, y, scaled = self.make_sequences(n_past=n_past)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        input_shape = (X.shape[1], 1)

        # LSTM
        lstm = self.build_lstm(input_shape)
        lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
        lstm.save(LSTM_PATH)

        # GRU
        gru = self.build_gru(input_shape)
        gru.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
        gru.save(GRU_PATH)

        # Bidirectional
        bidir = self.build_bidir(input_shape)
        bidir.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)
        bidir.save(BIDIR_PATH)

        np.save(SCALER_MIN, self.scaler.min_)
        np.save(SCALER_SCALE, self.scaler.scale_)

        np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

        return {"lstm": LSTM_PATH, "gru": GRU_PATH, "bidir": BIDIR_PATH}

    def _safe_load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        m = load_model(path, compile=False)
        # recompile safely
        m.compile(optimizer=Adam(0.0005), loss="mse", metrics=["mean_squared_error"])
        return m

    def load_model_by_name(self, name: str):
        name = name.lower()
        if name == "lstm":
            return self._safe_load(LSTM_PATH)
        elif name == "gru":
            return self._safe_load(GRU_PATH)
        elif name in ("bidir", "bidirectional"):
            return self._safe_load(BIDIR_PATH)
        else:
            raise ValueError("Unknown model")

    def load_scaler(self) -> MinMaxScaler:
        if self.scaler is not None:
            return self.scaler
        if os.path.exists(SCALER_MIN) and os.path.exists(SCALER_SCALE):
            mn = np.load(SCALER_MIN, allow_pickle=True)
            sc = np.load(SCALER_SCALE, allow_pickle=True)
            s = MinMaxScaler()
            s.min_ = mn
            s.scale_ = sc
            self.scaler = s
            return s
        else:
            # fallback: fit on data
            _, data = self.load_data()
            s = MinMaxScaler()
            s.fit(data)
            self.scaler = s
            return s

    def predict_multistep(self, model_name: str, days: int = 1, n_past: int = 60) -> List[Dict[str, Any]]:

        _, data = self.load_data()
        scaler = self.load_scaler()
        model = self.load_model_by_name(model_name)

        scaled_all = scaler.transform(data)
        cur_window = scaled_all[-n_past:].flatten().copy()

        preds_scaled = []
        for _ in range(days):
            p = model.predict(cur_window.reshape(1, n_past, 1), verbose=0).flatten()[0]
            preds_scaled.append(p)
            cur_window = np.append(cur_window[1:], p) 

        preds_real = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        last_actual = data.flatten()[-1]

        results = []
        prev_price = float(last_actual)
        for i, price in enumerate(preds_real, start=1):
            change = float(price - prev_price)
            percent = float((change / prev_price) * 100) if prev_price != 0 else 0.0
            direction = "up" if change > 0 else ("down" if change < 0 else "flat")
            results.append({
                "day": i,
                "price": float(price),
                "change": round(change, 4),
                "percent": round(percent, 4),
                "direction": direction
            })
            prev_price = float(price)

        return results
