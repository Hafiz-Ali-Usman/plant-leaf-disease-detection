# -*- coding: utf-8 -*-
"""
Created on Tue April 28 15:43:56 2025
@author: HP
"""

# Run using:
# streamlit run "C:/Users/DELL/Downloads/files/PDD_MobileNet.py"

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load your trained model
MODEL_PATH = r"H:\Plant-leaf-disease-detection-main\Plant-leaf-disease-detection-main\plant_leaf_disease_detection_MobileNet.h5"
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}
)

# List of all possible categories
CATEGORIES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Image prediction function
def model_predict(img):
    try:
        img_resized = cv2.resize(img, (224, 224))  # Resize to model input size
        img_normalized = img_resized / 255.0       # Normalize pixel values
        img_reshaped = img_normalized.reshape(-1, 224, 224, 3)  # Add batch dimension
        preds = model.predict(img_reshaped)
        return preds
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Main Streamlit app
def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
    st.title("🌿 Plant Leaf Disease Classifier")

    # Project and submission info
    st.markdown("""
        ### 👩‍💻 ANN Project by:
        - Ali Usman (21-SE-19)
        - Syed Moazam Ali (21-SE-48)
        

        ### 🧑‍🏫 Submitted to:
        - Sir Hassan Dawood
    """)

    # File uploader
    uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert uploaded file to OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None or len(image.shape) != 3:
                st.error("⚠️ Invalid image format. Please upload a clear color image.")
                return

            # Display uploaded image (in RGB)
            display_img = cv2.resize(image, (200, 200))
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            st.write("🔍 Classifying... Please wait.")
            preds = model_predict(image)

            if preds is not None:
                pred_class = np.argmax(preds)
                percentage = preds[0][pred_class] * 100
                st.success(f"✅ Prediction: **{CATEGORIES[pred_class]}**")
                st.info(f"🔢 Confidence: **{percentage:.2f}%**")

        except Exception as e:
            st.error(f"🚨 Error processing image: {e}")

# Run the app
if __name__ == "__main__":
    main()
