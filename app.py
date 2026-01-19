import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

def custom_input_layer(*args, **kwargs):
    if "batch_shape" in kwargs:
        kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
    return tf.keras.layers.InputLayer(*args, **kwargs)

with tf.keras.utils.custom_object_scope({"InputLayer": custom_input_layer}):
    model = tf.keras.models.load_model("./acne_model_ready_for_deployment.keras")

CLASS_NAMES = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def apply_prediction_rules(probs, class_names):
    idx = np.argmax(probs)
    predicted_class = class_names[idx]
    confidence = probs[idx] * 100

    severe_idx = class_names.index("severe")
    moderate_idx = class_names.index("moderate")
    mild_idx = class_names.index("mild")

    severe_conf = probs[severe_idx] * 100
    moderate_conf = probs[moderate_idx] * 100

    if predicted_class == "severe" and severe_conf < 50:
        if moderate_conf < 30:
            return "mild", probs[mild_idx] * 100, True
        return "moderate", probs[moderate_idx] * 100, True

    if predicted_class == "moderate" and moderate_conf < 30:
        return "mild", probs[mild_idx] * 100, True

    return predicted_class, confidence, False

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Added Index route for basic API check
@app.route("/", methods=["GET"])
def index():
    return "Acne Severity Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    try:
        image = Image.open(request.files["image"])
        img = preprocess_image(image)
        probs = model.predict(img)[0]

        prediction, confidence, rule_applied = apply_prediction_rules(
            probs, CLASS_NAMES
        )

        return jsonify({
    "prediction": str(prediction),
    "confidence": float(round(confidence, 2)),
    "rule_applied": bool(rule_applied),
    "probabilities": {
        CLASS_NAMES[i]: float(round(float(probs[i]) * 100, 2))
        for i in range(len(CLASS_NAMES))
        }
    })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

