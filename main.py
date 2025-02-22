import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import logging
from model import DigitRecognitionModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize and train the model
model = DigitRecognitionModel()
model.train()  # Using default parameters

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Resize to 8x8 (scikit-learn digits dataset size)
        image = image.resize((8, 8), Image.Resampling.LANCZOS)

        # Convert to numpy array and scale to match digits dataset range (0-16)
        image_array = np.array(image).astype('float32')
        image_array = image_array * (16.0 / 255.0)  # Scale from [0,255] to [0,16]

        logging.info(f"Preprocessed image shape: {image_array.shape}")
        logging.info(f"Preprocessed value range: [{image_array.min()}, {image_array.max()}]")

        # Make prediction
        predicted_digit, confidence = model.predict(image_array)

        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to process the image'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)