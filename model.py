import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import logging

class DigitRecognitionModel:
    def __init__(self):
        """Initialize the digit recognition model"""
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Increased network capacity
            max_iter=100,  # Increased max iterations
            activation='relu',
            solver='adam',
            random_state=1,
            learning_rate_init=0.001,  # Added explicit learning rate
            early_stopping=True,  # Added early stopping
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()
        self._is_trained = False

    def train(self, epochs=1, batch_size=32):
        """Train the model using MNIST dataset from scikit-learn"""
        try:
            from sklearn.datasets import load_digits

            logging.info("Loading digits dataset...")
            digits = load_digits()

            # Preprocess the data
            logging.info("Preprocessing training data...")
            X = self.scaler.fit_transform(digits.data)
            y = digits.target

            logging.info("Training model...")
            self.model.fit(X, y)
            self._is_trained = True

            # Get training score
            score = self.model.score(X, y)
            logging.info(f"Training accuracy: {score:.4f}")

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def predict(self, image):
        """
        Make prediction on input image

        Args:
            image: numpy array of shape (8, 8)

        Returns:
            tuple: (predicted_digit, confidence)
        """
        if not self._is_trained:
            raise RuntimeError("Model needs to be trained before making predictions")

        try:
            logging.info(f"Input image shape: {image.shape}")
            logging.info(f"Input image value range: [{image.min()}, {image.max()}]")

            # Reshape and scale the image using the same scaler as training
            image_reshaped = image.reshape(1, -1)
            image_scaled = self.scaler.transform(image_reshaped)

            logging.info(f"Preprocessed image shape: {image_scaled.shape}")
            logging.info(f"Preprocessed value range: [{image_scaled.min()}, {image_scaled.max()}]")

            # Get prediction and probability
            prediction = self.model.predict(image_scaled)
            probabilities = self.model.predict_proba(image_scaled)

            predicted_digit = prediction[0]
            confidence = float(probabilities[0][predicted_digit] * 100)

            logging.info(f"Predicted digit: {predicted_digit}, confidence: {confidence:.2f}%")
            return predicted_digit, confidence

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise