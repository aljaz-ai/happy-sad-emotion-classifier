import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model class exactly as before
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 1152),
            nn.ReLU(),
            nn.Linear(1152, 576),
            nn.ReLU(),
            nn.Linear(576, 192),
            nn.ReLU(),
            nn.Linear(192, 2)
        )
    def forward(self, X):
        return self.model(X)

# Load model
model = NeuralNetwork()
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform for incoming image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Prediction function
def predict_emotion(image):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        prediction = torch.argmax(outputs, dim=1).item()
        return "Happy 😄" if prediction == 0 else "Sad 😢"

# Gradio UI
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Upload 48x48 face image"),
    outputs=gr.Label(label="Predicted Emotion"),
    title="Facial Emotion Classifier",
    description="Model is meant to be used with a 48x48 grayscale face image. In other cases, it works fine but is more unpredictable. This model was trained from scratch using PyTorch to classify Happy or Sad emotions.",
    examples=[]
)

demo.launch()