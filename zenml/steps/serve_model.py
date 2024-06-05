from zenml import step
from zenml.logger import get_logger
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from PIL import Image
import mlflow
import numpy as np

logger = get_logger(__name__)

@step(enable_cache=False)
def serve_model(model_uri: str):
    app = Flask(__name__)

    model = mlflow.fastai.load_model(model_uri)

    @app.route('/predict', methods=['POST'])
    @cross_origin(origins="*")
    def predict():
        file = request.files['image']
        img = Image.open(file.stream)
        img = np.asarray(img)
        prediction = model.predict(img)
        predicted_class = prediction[0]
        confidence = prediction[2][prediction[1]].item()
        return jsonify({'msg': 'success', 'class': predicted_class, 'confidence':  confidence})

    app.run()