import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from models.object_cnn_model import ObjectCNN

object_bp = Blueprint("object_bp", __name__)
model = ObjectCNN()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@object_bp.route("/train", methods=["POST"])
def train_model():
    result = model.train()
    return jsonify({"message": result})

@object_bp.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    label, confidence = model.predict(filepath)

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })
