from flask import Blueprint, request, jsonify
from models.stock_models import StockModels
import os

stock_bp = Blueprint("stock_bp", __name__)
sm = StockModels()

MAX_RECOMMENDED = 30
MAX_ALLOWED = 365

@stock_bp.route("/predict", methods=["POST"])
def predict_route():
    """
    Form fields:
      - model: 'lstm' | 'gru' | 'bidir' (default 'lstm')
      - days: int (default 1)
      - n_past: optional window size (default 60)
    Returns list of predictions per day (price, direction, percent)
    """
    try:
        model_name = request.form.get("model", "lstm").lower()
        days = int(request.form.get("days", 1))
        n_past = int(request.form.get("n_past", 60))

        if days <= 0:
            return jsonify({"error": "days must be >= 1"}), 400
        if days > MAX_ALLOWED:
            return jsonify({"error": f"days too large (max {MAX_ALLOWED})"}), 400

        predictions = sm.predict_multistep(model_name=model_name, days=days, n_past=n_past)

        note = None
        if days > MAX_RECOMMENDED:
            note = f"Requested {days} days (> {MAX_RECOMMENDED}). Predictions beyond {MAX_RECOMMENDED} may be less reliable."

        return jsonify({
            "model_used": model_name,
            "days_requested": days,
            "note": note,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
