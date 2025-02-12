import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import io

def classify_image(uploaded_file, model):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))  # Resize the image to match model input size
        
        # Convert image to numpy array and preprocess
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1, 224, 224, 3)
        
        # Predict the label
        label = model.predict(img)
        
        # Determine the predicted class
        class_names = ['Dermatofibroma', 'Melanocyticnevus', 'Melanoma']
        predicted_class_index = np.argmax(label)
        predicted_class = class_names[predicted_class_index]
        
        # Display results
        st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
        st.markdown(f"## 🎯 Prediction: **{predicted_class}**")

# Streamlit UI
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 5px;
            font-size: 16px;
        }
        .black-text {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main'><h1 class='black-text'>🔬 Skin Cancer Classifier</h1></div>", unsafe_allow_html=True)

# Load the trained model
model = load_model('my_model.keras')

uploaded_file = st.file_uploader("📂 Upload an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    classify_image(uploaded_file, model)
