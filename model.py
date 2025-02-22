import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Simplified architecture
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DigitRecognitionModel:
    def __init__(self):
        try:
            logging.info("Initializing DigitRecognitionModel...")
            self.device = torch.device("cpu")
            self.model = Net().to(self.device)
            self._is_trained = False
            os.makedirs('./data', exist_ok=True)
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def train(self, epochs=1, batch_size=8, timeout=300):  # 5 minutes timeout
        try:
            start_time = time.time()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            logging.info("Loading MNIST dataset...")
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

            # Use a smaller subset of data for faster training
            subset_size = min(10000, len(train_dataset))
            indices = torch.randperm(len(train_dataset))[:subset_size]
            subset_dataset = torch.utils.data.Subset(train_dataset, indices)

            train_loader = torch.utils.data.DataLoader(
                subset_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1
            )

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)  # Increased learning rate

            logging.info("Starting model training...")
            self.model.train()

            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Check timeout
                    if time.time() - start_time > timeout:
                        logging.warning("Training timeout reached")
                        self._is_trained = True
                        return True

                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()

                    if batch_idx % 100 == 0:
                        logging.info(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            self._is_trained = True
            logging.info("Training completed successfully")
            return True

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def predict(self, image):
        if not self._is_trained:
            logging.warning("Model is not trained. Training now...")
            self.train()

        try:
            # Invert the image since MNIST has white digits on black background
            image = 1 - image

            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            image_tensor = transforms.Normalize((0.1307,), (0.3081,))(image_tensor)

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