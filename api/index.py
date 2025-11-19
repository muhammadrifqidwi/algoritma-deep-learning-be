from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/logic/calculate", methods=["POST"])
def logic_calculate():
    data = request.get_json()
    input_vars = data.get("input", [0, 0])
    operator = data.get("operator", "AND")

    a, b = input_vars
    result = 0
    if operator == "AND":
        result = a & b
    elif operator == "OR":
        result = a | b
    elif operator == "XOR":
        result = a ^ b
    elif operator == "NAND":
        result = 1 - (a & b)
    elif operator == "NOR":
        result = 1 - (a | b)
    elif operator == "NOT_A":
        result = 1 - a
    elif operator == "NOT_B":
        result = 1 - b

    return jsonify({"result": result})

@app.route("/stock/predict", methods=["POST"])
def stock_predict():
    data = request.form
    model = data.get("model", "lstm")
    days = int(data.get("days", 1))
    # TODO: load model lazy
    predictions = [{"day": i + 1, "price": 100000 + i*1000, "direction": "up", "percent": 1.0} for i in range(days)]
    return jsonify({"model_used": model, "predictions": predictions})

@app.route("/text/train", methods=["POST"])
def text_train():
    data = request.get_json()
    model = data.get("model", "lstm")
    text = data.get("text", "")
    # TODO: lakukan training model ML disini
    return jsonify({"message": f"Model {model} berhasil di-train dengan {len(text)} karakter"})

@app.route("/text/generate", methods=["POST"])
def text_generate():
    data = request.get_json()
    model = data.get("model", "lstm")
    seed = data.get("seed", "")
    length = int(data.get("length", 20))
    # TODO: lakukan generate text
    generated_text = seed + " " + " ".join([f"kata{i}" for i in range(length)])
    return jsonify({"generated_text": generated_text, "model_used": model})
