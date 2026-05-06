from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


MODEL_PATH = Path("cnn_improved_brain_tumor.pth")
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


class CNNImproved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = CNNImproved(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


def predict_image(image):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_label = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score, probabilities.cpu().numpy()[0]


def render_probability_bars(class_names, probabilities):
    for class_name, probability in zip(class_names, probabilities):
        st.write(f"**{DISPLAY_NAMES.get(class_name, class_name.title())}**: {probability * 100:.2f}%")
        st.progress(float(probability))


def apply_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: #050A12;
            color: #F8FAFC;
        }

        [data-testid="stHeader"] {
            background: rgba(5, 10, 18, 0.88);
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: #0B1220;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #111827;
            border-color: #243044;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #1F2937;
            border-color: #334155;
            color: #F8FAFC;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="auto",
)

model, class_names = load_model()
apply_dark_theme()

st.title("Brain Tumor MRI Classification")
st.caption("Educational AI prototype for classifying uploaded brain MRI images.")

with st.sidebar:
    st.header("Model")
    st.write("CNN Improved")
    st.write(f"Input size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    st.write(f"Device: {DEVICE}")
    st.write("Classes:")
    for class_name in class_names:
        st.write(f"- {DISPLAY_NAMES.get(class_name, class_name.title())}")
    st.divider()
    st.warning("Not intended for clinical diagnosis.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a JPG or PNG brain MRI image to get a prediction.")
else:
    image = Image.open(uploaded_file)
    predicted_label, confidence_score, probabilities = predict_image(image)

    image_column, result_column = st.columns([1, 1], gap="large")

    with image_column:
        st.subheader("Uploaded Image")
        st.image(image, caption="MRI image", width="stretch")

    with result_column:
        st.subheader("Prediction")
        st.metric("Predicted class", DISPLAY_NAMES.get(predicted_label, predicted_label.title()))
        st.metric("Confidence", f"{confidence_score * 100:.2f}%")

        st.subheader("Class Probabilities")
        render_probability_bars(class_names, probabilities)

    st.info("Review predictions with qualified medical professionals before making decisions.")
