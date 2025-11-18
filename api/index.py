from flask import Flask, jsonify
from flask_cors import CORS
import os

from models.logic_model import logic_bp
from models.text_model import text_bp
from routes.stock_routes import stock_bp
from routes.object_routes import object_bp
from models.object_cnn_model import ObjectCNN

from models.stock_models import (
    StockModels,
    LSTM_PATH,
    GRU_PATH,
    BIDIR_PATH
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})


MODEL_PATH = "models/object_cnn_model.h5"
object_model = ObjectCNN()

if not os.path.exists(MODEL_PATH):
    print("⚠ Model object_cnn_model.h5 tidak ditemukan.")
    print("🔥 Melakukan training otomatis menggunakan CIFAR-10 (Cat vs Dog)...")
    object_model.train_cifar10(epochs=6)
    print("✅ Training selesai dan model sudah disimpan!")
else:
    print("✅ Model object detection sudah ada.")


sm = StockModels()

if not (os.path.exists(LSTM_PATH) and os.path.exists(GRU_PATH) and os.path.exists(BIDIR_PATH)):
    print("Models missing — training (this may take a while). Adjust epochs if needed.")
    sm.train_all(epochs=30, batch_size=32) 
else:
    print("Stock models present, skipping training.")

app.register_blueprint(logic_bp, url_prefix="/logic")
app.register_blueprint(text_bp, url_prefix="/text")
app.register_blueprint(stock_bp, url_prefix="/stock")
app.register_blueprint(object_bp, url_prefix="/object")

@app.route('/')
def home():
    return jsonify({
        "message": "API Aktif: Logic, Text, Stock Prediction (BBCA), Object Detection"
    })


if __name__ == "__main__":
    app.run(debug=True)
