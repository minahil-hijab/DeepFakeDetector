import streamlit as st
import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "real_fake_detector.pth")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, 1)
    label = "Real" if pred_class.item() == 0 else "Fake"
    return label, conf.item() * 100

# Streamlit UI
st.title("🕵️ Real vs Fake Image Detector")
st.write("Upload an image and the AI will tell you if it's Real or AI-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        label, confidence = predict_image(img)
        color = "green" if label == "Real" else "red"
        st.markdown(f"<h3 style='color:{color}'>Prediction: {label} ({confidence:.2f}%)</h3>", unsafe_allow_html=True)
