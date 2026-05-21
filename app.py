import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# ১. গুগল ড্রাইভ থেকে মডেল ডাউনলোড করার ফাংশন
@st.cache_resource
def load_model_from_drive():
    model_path = 'makeup_style_resnet50.h5'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... Please wait, this takes a few minutes but happens only once."):
            # তোর শেয়ার করা গুগল ড্রাইভ লিঙ্ক
            drive_url = "https://drive.google.com/file/d/1vACDcidGM2r41GAMqbKRySnIbJ6y7JAg/view?usp=sharing"
            
            # লিঙ্ক থেকে ফাইল আইডি বের করার নিয়ম
            if "id=" in drive_url:
                file_id = drive_url.split("id=")[1].split("&")[0]
            else:
                file_id = drive_url.split("/d/")[1].split("/")[0]
                
            download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
            
            # ডাউনলোড শুরু
            response = requests.get(download_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
    return tf.keras.models.load_model(model_path)

try:
    model = load_model_from_drive()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# ২. ক্লাসের নামগুলো
class_names = [
    'smokey_eyes_makeup', 'evening_makeup', 'evening_glamour_makeup', 
    'bold_makeup', 'fantasy_makeup', 'casual_makeup', 'no_makeup', 'vintage_makeup'
]

# ৩. ইন্টারফেস
st.set_page_config(page_title="Makeup Classifier", page_icon="💄")
st.title("💄 Makeup Style Classification System")
st.write("Upload an image, and our ResNet50 model will detect the makeup style!")

if model_loaded:
    uploaded_file = st.file_uploader("Choose a makeup image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("🔄 Classifying...")
        
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"🎯 **Prediction:** {predicted_class.replace('_', ' ').title()}")
        st.info(f"📊 **Confidence Level:** {confidence:.2f}%")