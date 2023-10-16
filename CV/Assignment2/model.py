import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image

class ImageClassifier:
    def __init__(self, model_name='ConvNexT', lr=0.001):
        # Define the chosen model (ConvNexT or ViT)
        self.model_name = model_name
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def _load_model(self):
        if self.model_name == 'ConvNexT':
            return models.convnext_tiny(weights='DEFAULT')
        elif self.model_name == 'vit':
            # You can use a ViT model here.
            return models.vit_l_16(weights='DEFAULT')
        else:
            raise ValueError("Invalid model name.")
    
    def train_model(self, train_loader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                self.optimizer.zero_grad()
    
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
    
                running_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    
    def classify_images(self, image_files):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        results = []
        for image_file in image_files:
            image = Image.open(image_file)
            image = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(image)
            _, predicted = torch.max(output, 1)
            results.append(predicted.item())
    
        return results
    
    def classify_video(self, video_path):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        cap = cv2.VideoCapture(video_path)
        results = []
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(self.device)
    
            with torch.no_grad():
                output = self.model(image)
            _, predicted = torch.max(output, 1)
            results.append(predicted.item())
    
        cap.release()
        cv2.destroyAllWindows()
    
        return results
    
    def get_average_prediction(self, predictions):
        return np.mean(predictions)

# Example usage:
# classifier = ImageClassifier(model_name='ConvNexT', lr=0.001)
# Define your training data and data loaders
# classifier.train_model(train_loader, num_epochs)

# Inference on multiple images (video frames)
# image_files = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
# predicted_classes = classifier.classify_images(image_files)
# Calculate the average prediction for images
# average_image_prediction = classifier.get_average_prediction(predicted_classes)
# print(f"Average Image Prediction: {average_image_prediction}")

# Video file path
# video_path = 'video.mp4'
# Inference on the video
# predicted_video_classes = classifier.classify_video(video_path)
# Calculate the average prediction for the video
# average_video_prediction = classifier.get_average_prediction(predicted_video_classes)
# print(f"Average Video Prediction: {average_video_prediction}")
