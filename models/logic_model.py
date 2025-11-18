from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

logic_bp = Blueprint('logic_bp', __name__)

models = {}

def train_ann(operator):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])

    def logic_func(a, b):
        if operator == "AND": return a & b
        if operator == "OR": return a | b
        if operator == "XOR": return a ^ b
        if operator == "NAND": return int(not (a & b))
        if operator == "NOR": return int(not (a | b))
        return 0

    Y = np.array([logic_func(a,b) for a,b in X])

    model = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=300, verbose=0)
    return model

@logic_bp.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        inputs = data.get('input', [])
        operator = data.get('operator', 'AND')

        if len(inputs) != 2:
            return jsonify({"error": "Input harus 2 nilai"}), 400

        key = f"{operator}_2"
        if key not in models:
            models[key] = train_ann(operator)

        model = models[key]
        pred = model.predict(np.array([inputs]), verbose=0)[0][0]
        result = int(pred > 0.5)
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
