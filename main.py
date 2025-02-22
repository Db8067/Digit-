import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import logging
import threading
from model import DigitRecognitionModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Global model variable
model = None
model_ready = threading.Event()

def train_model_async():
    """Train the model in a separate thread"""
    global model
    try:
        logger.info("Starting model training in background...")
        success = model.train(epochs=1, batch_size=8, timeout=300)
        if success:
            logger.info("Model training completed successfully")
        else:
            logger.warning("Model training incomplete but usable")
        model_ready.set()
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        model_ready.set()  # Set the event even on failure to unblock predictions

def initialize_model(train_model=True):
    """Initialize and optionally start training the model"""
    global model
    try:
        logger.info("Starting model initialization...")
        model = DigitRecognitionModel()
        logger.info("Model instance created successfully")

        if train_model:
            # Start training in a separate thread
            training_thread = threading.Thread(target=train_model_async)
            training_thread.daemon = True  # Thread will be terminated when main program exits
            training_thread.start()
        else:
            model_ready.set()  # If not training, mark as ready immediately

        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}", exc_info=True)
        return False

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model_ready.is_set()
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is ready
        if not model_ready.is_set():
            return jsonify({
                'success': False,
                'error': 'Model is still training, please try again in a moment'
            }), 503

        # Get the image data from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        logger.debug("Successfully loaded and converted image")

        # Resize to 28x28 (MNIST dataset size)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        logger.debug("Successfully resized image")

        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        image_array = image_array.reshape(28, 28, 1)  # Add channel dimension

        # Make prediction
        predicted_digit, confidence = model.predict(image_array)
        logger.info(f"Prediction successful: digit={predicted_digit}, confidence={confidence}")

        return jsonify({
            'success': True,
            'digit': int(predicted_digit),
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to process the image'
        }), 400

if __name__ == '__main__':
    try:
        train_model_flag = os.environ.get('TRAIN_MODEL', 'true').lower() == 'true'

        logger.info("Starting Flask application...")
        if not initialize_model(train_model=train_model_flag):
            logger.error("Model initialization failed. Exiting.")
            exit(1)
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}", exc_info=True)
        raise