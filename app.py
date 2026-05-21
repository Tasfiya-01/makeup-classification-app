import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ১. মডেল লোড করার ফাংশন (ক্যাশিং যাতে বারবার লোড না হয়)
@st.cache_resource
def load_model():
    # এখানে 'GetItem' লেয়ারের জন্য কাস্টম অবজেক্ট হ্যান্ডেল করা হয়েছে
    # যদি আপনার মডেলে নির্দিষ্ট কোনো কাস্টম লেয়ার থাকে, তা এখানে যোগ করুন
    model = tf.keras.models.load_model('model.h5', custom_objects={'GetItem': tf.keras.layers.Lambda})
    return model

# ২. ইমেজ প্রিপ্রসেসিং ফাংশন
def preprocess_image(image):
    image = image.resize((224, 224))  # আপনার মডেলের ইনপুট সাইজ অনুযায়ী পরিবর্তন করুন
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ৩. অ্যাপের ইন্টারফেস
st.title("মেকআপ ক্লাসিফায়ার অ্যাপ")
st.write("আপনার ইমেজের মেকআপ স্টাইল শনাক্ত করুন।")

uploaded_file = st.file_uploader("একটি ছবি আপলোড করুন...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='আপলোড করা ছবি', use_column_width=True)
    
    # মডেল লোড করা
    model = load_model()
    
    # প্রেডিকশন
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    
    # আউটপুট দেখানো
    st.write("প্রেডিকশন সম্পন্ন!")
    # আপনার ক্লাসের লিস্ট অনুযায়ী আউটপুট দেখান
    classes = ["Natural", "Glam", "Minimalist"] # আপনার মডেলের ক্লাসগুলো এখানে দিন
    st.success(f"শনাক্তকৃত স্টাইল: {classes[np.argmax(prediction)]}")
