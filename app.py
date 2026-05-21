import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Makeup Classifier", page_icon="💄")

# Function to load the model with custom object handling
@st.cache_resource
def load_model():
    # 'GetItem' error usually occurs due to specific Keras versioning
    # We define a custom Lambda layer to handle this specific issue
    model = tf.keras.models.load_model(
        'model.h5', 
        custom_objects={'GetItem': tf.keras.layers.Lambda(lambda x: x)}
    )
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI Layout
st.title("💄 Makeup Style Classifier")
st.write("Upload an image to identify the makeup style.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Loading model and analyzing...'):
        try:
            model = load_model()
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Define your class names based on your model training
            classes = ["Natural", "Glam", "Minimalist"] 
            
            st.success(f"Result: {classes[np.argmax(prediction)]}")
        except Exception as e:
            st.error(f"Error during analysis: {e}")
