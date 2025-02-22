import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DigitRecognitionModel:
    def __init__(self):
        """Initialize the digit recognition model"""
        try:
            logging.info("Initializing DigitRecognitionModel...")
            self.device = torch.device("cpu")
            self.model = Net().to(self.device)
            self._is_trained = False

            # Create data directory if it doesn't exist
            os.makedirs('./data', exist_ok=True)
            logging.info("Model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def train(self, epochs=3, batch_size=64):
        """Train the model using MNIST dataset"""
        try:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # Load MNIST dataset with verbose logging
            logging.info("Loading MNIST dataset...")
            try:
                train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                logging.info("MNIST dataset loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load MNIST dataset: {str(e)}")
                raise

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)

            # Training settings
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

            logging.info("Starting model training...")
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                logging.info(f"Epoch {epoch+1}/{epochs}")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    if batch_idx % 100 == 0:
                        logging.info(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

                # Update learning rate based on loss
                scheduler.step(total_loss)

            self._is_trained = True
            logging.info("Training completed successfully")
            return True

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def predict(self, image):
        """
        Make prediction on input image

        Args:
            image: numpy array of shape (28, 28, 1)

        Returns:
            tuple: (predicted_digit, confidence)
        """
        if not self._is_trained:
            logging.warning("Model is not trained. Training now...")
            self.train()

        try:
            logging.debug(f"Input image shape: {image.shape}")

            # Convert numpy array to tensor
            image_tensor = torch.from_numpy(image).float()
            logging.debug(f"Image tensor shape before reshape: {image_tensor.shape}")

            # Add batch dimension and ensure channel dimension is correct
            image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            logging.debug(f"Image tensor shape after reshape: {image_tensor.shape}")

            # Normalize
            image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                pred_probabilities = torch.exp(output)
                predicted_digit = pred_probabilities.argmax(dim=1).item()
                confidence = pred_probabilities[0][predicted_digit].item() * 100

            logging.info(f"Predicted digit: {predicted_digit}, confidence: {confidence:.2f}%")
            return predicted_digit, confidence

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise