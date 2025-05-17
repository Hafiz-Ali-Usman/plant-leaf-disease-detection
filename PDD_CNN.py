import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
MODEL_PATH = r"H:\Plant-leaf-disease-detection-main\Plant-leaf-disease-detection-main\my_model.h5"
model = load_model(MODEL_PATH)

# The image size your model expects
IMG_SIZE = 224

# Replace with your actual classes (based on dataset folder names)
CATEGORIES = ['Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
              'Blueberry_healthy', 'Cherry_healthy', 'Cherry_Powdery_mildew',
              'Corn_Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_healthy',
              'Corn_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca',
              'Grape_healthy', 'Grape_Leaf_blight', 'Orange_Haunglongbing',
              'Peach_Bacterial_spot', 'Peach_healthy', 'Pepper_Bacterial_spot',
              'Pepper_healthy', 'Potato_Early_blight', 'Potato_healthy',
              'Potato_Late_blight', 'Raspberry_healthy', 'Soybean_healthy',
              'Squash_Powdery_mildew', 'Strawberry_healthy', 'Strawberry_Leaf_scorch',
              'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
              'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
              'Tomato_Spider_mites', 'Tomato_Target_Spot',
              'Tomato_Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']

def model_predict(img):
    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img / 255.0
    reshaped_img = np.reshape(normalized_img, (1, IMG_SIZE, IMG_SIZE, 3))
    preds = model.predict(reshaped_img)
    return preds

def main():
    st.title("🌿 Plant Disease Classifier")

    uploaded_file = st.file_uploader("Upload a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        preds = model_predict(image)
        pred_class = np.argmax(preds)
        confidence = preds[0][pred_class] * 100

        st.success(f"Prediction: **{CATEGORIES[pred_class]}** with **{confidence:.2f}%** confidence.")

if __name__ == "__main__":
    main()
