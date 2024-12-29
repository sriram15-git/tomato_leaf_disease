import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Setting up working directory paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/tomato_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Disease solutions dictionary
disease_solutions = {
    "Bacterial_spot": "Use copper-based fungicides and avoid overhead irrigation.",
    "Early_blight": "Apply fungicides and remove infected leaves to reduce spread.",
    "Late_blight": "Use resistant varieties and apply fungicides.",
    "Leaf_Mold": "Ensure good air circulation and avoid excess humidity.",
    "Septoria_leaf_spot": "Use fungicides and practice crop rotation.",
    "Spider_mites Two-spotted_spider_mite": "Use miticides and encourage natural predators.",
    "Target_Spot": "Apply fungicides and remove affected plant debris.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Use virus-resistant varieties and control whiteflies.",
    "Tomato_mosaic_virus": "Remove infected plants and avoid tobacco products near plants.",
    "healthy": "✨No issues detected. Keep up good practices!",
    "powdery_mildew": "Apply sulfur-based fungicides and improve airflow around plants."
}

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, disease_solutions[predicted_class_name]


# Streamlit page layout and custom CSS
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://tse2.mm.bing.net/th?id=OIG4.51V.YHqIDxretrtVd79N&pid=ImgGn");
    background-size: 100% 100%; 
    background-attachment: scroll; 
    background-repeat: no-repeat; 
    background-position: center; 
}
h1 {
    color: #03031a;
    text-align: center;
    font-family: 'Arial', sans-serif;
}
.upload-area {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.result-box {
    background-color: #b787e8;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
}
.button:hover {
    background-color: #45a049;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar with additional information
st.sidebar.title("About")
st.sidebar.info(
    "This app allows you to classify Tomato plant diseases from uploaded images using a pre-trained machine learning model."
)

# Main Title
st.title('🌿 Plant-tom🍅 Disease Classifier')

# Image upload
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)

    # Display original and resized images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Resized Image (150x150)")
        resized_img = image.resize((150, 150))
        st.image(resized_img, use_column_width=True)

    # Button to classify the uploaded image
    if st.button('Classify🔎'):
        # Predict the class of the image and get solution
        prediction, solution = predict_image_class(model, image, class_indices)

        # Display prediction and solution
        st.markdown(f"<div class='result-box'><h3>Prediction: {prediction}</h3><p>Solution: {solution}</p></div>",
                    unsafe_allow_html=True)
else:
    st.markdown("<div class='upload-area'><h4>Please upload an image of the plant leaf to classify.</h4></div>",
                unsafe_allow_html=True)
