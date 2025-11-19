from flask import Flask, jsonify, request
from flask_cors import CORS
import os

from models.logic_model import logic_bp
from models.text_model import text_bp
from routes.stock_routes import stock_bp
from routes.object_routes import object_bp

from models.object_cnn_model import ObjectCNN
from models.stock_models import StockModels, LSTM_PATH, GRU_PATH, BIDIR_PATH

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://algoritma-deep-learning-fe.vercel.app"}})

object_model = None
stock_models = None

@app.route("/")
def home():
    return jsonify({
        "message": "API Aktif: Logic, Text, Stock Prediction (BBCA), Object Detection"
    })

app.register_blueprint(logic_bp, url_prefix="/logic")
app.register_blueprint(text_bp, url_prefix="/text")
app.register_blueprint(stock_bp, url_prefix="/stock")
app.register_blueprint(object_bp, url_prefix="/object")

@app.before_request
def load_models():
    global object_model, stock_models

    if request.path.startswith("/object") and object_model is None:
        object_model = ObjectCNN()
        MODEL_PATH = "models/object_cnn_model.h5"
        if os.path.exists(MODEL_PATH):
            object_model.load_model(MODEL_PATH)
            print("✅ Object detection model loaded")
        else:
            print("⚠ Object detection model not found. Skipping load.")

    # Stock prediction models
    if request.path.startswith("/stock") and stock_models is None:
        stock_models = StockModels()
        if all(os.path.exists(p) for p in [LSTM_PATH, GRU_PATH, BIDIR_PATH]):
            stock_models.load_all_models()
            print("✅ Stock models loaded")
        else:
            print("⚠ Stock models missing. Skipping load.")

