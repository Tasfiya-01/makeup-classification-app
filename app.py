import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import plotly.graph_objects as go

st.set_page_config(
    page_title="Makeup Style Classifier",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Lato:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
    .stApp { background: linear-gradient(160deg, #fdf6f0 0%, #fce8e8 50%, #f5e6f0 100%); }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid #f0d9e5;
    }
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #b5547a !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
    }
    h3, h4 { color: #7a3d5c !important; font-family: 'Playfair Display', serif !important; }
    .result-card {
        background: rgba(255,255,255,0.75);
        border: 1.5px solid #e8b4cc;
        border-radius: 20px;
        padding: 28px 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(181,84,122,0.10);
    }
    .confidence-badge {
        background: linear-gradient(90deg, #d4799f, #b5547a);
        color: white;
        padding: 8px 22px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 700;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(181,84,122,0.25);
    }
    div[data-testid="stInfo"] {
        background: rgba(255,240,248,0.8);
        border: 1px solid #e8b4cc;
        color: #7a3d5c;
        border-radius: 12px;
    }
    hr { border-color: #f0d0e0 !important; }
</style>
""", unsafe_allow_html=True)

CLASSES = [
    "bold_makeup", "casual_makeup", "evening_glamour_makeup", "evening_makeup",
    "fantasy_makeup", "no_makeup", "smokey_eyes_makeup", "vintage_makeup"
]

CLASS_INFO = {
    "bold_makeup":            {"emoji": "💋", "desc": "Vibrant, striking colors with high contrast"},
    "casual_makeup":          {"emoji": "🌸", "desc": "Everyday natural look, light and fresh"},
    "evening_glamour_makeup": {"emoji": "✨", "desc": "Glamorous, luxurious look for special occasions"},
    "evening_makeup":         {"emoji": "🌙", "desc": "Elegant and polished for evening events"},
    "fantasy_makeup":         {"emoji": "🦋", "desc": "Creative, artistic and imaginative styles"},
    "no_makeup":              {"emoji": "🪷", "desc": "Bare face or minimal skincare only"},
    "smokey_eyes_makeup":     {"emoji": "🖤", "desc": "Dark, sultry eye looks with dramatic effect"},
    "vintage_makeup":         {"emoji": "🎭", "desc": "Classic retro styles from past decades"},
}


@st.cache_resource
def load_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model... please wait.")
        file_id = "1vACDcidGM2r41GAMqbKRySnIbJ6y7JAg"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image.convert("RGB")) / 255.0
    return np.expand_dims(img_array, axis=0)


def make_chart(prediction):
    scores = [float(prediction[0][i]) * 100 for i in range(len(CLASSES))]
    labels = [f"{CLASS_INFO[c]['emoji']} {c.replace('_', ' ').title()}" for c in CLASSES]
    colors = ["#b5547a" if s == max(scores) else "#f0c8dc" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="#e8b4cc", width=1)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#7a3d5c", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a3d5c"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 115]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#7a3d5c")),
        margin=dict(l=10, r=10, t=10, b=10), height=350,
    )
    return fig


with st.sidebar:
    st.markdown("## 💄 About")
    st.markdown("This AI classifier detects **8 makeup styles** from your photo.")
    st.markdown("---")
    st.markdown("### 🎨 Style Guide")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"**{info['emoji']} {cls.replace('_',' ').title()}**")
        st.caption(info["desc"])
    st.markdown("---")
    st.caption("Built with TensorFlow + Streamlit")


st.markdown("# 💄 Makeup Style Classifier")
st.markdown("#### Upload a face photo — AI will detect the makeup style instantly.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

with col2:
    st.markdown("### 🔍 Analysis Result")
    if not uploaded_file:
        st.info("👈 Upload an image on the left to get started.")
    else:
        with st.spinner("Analyzing makeup style..."):
            try:
                model = load_model()
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)

                idx = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100
                top_class = CLASSES[idx]
                info = CLASS_INFO[top_class]

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:3rem">{info['emoji']}</div>
                    <div style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#7a3d5c; margin:8px 0">
                        {top_class.replace('_',' ').title()}
                    </div>
                    <div class="confidence-badge">🎯 {confidence:.1f}% Confidence</div>
                    <div style="color:#a06080; margin-top:8px; font-size:0.9rem">{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 🏆 Top 3 Predictions")
                top3 = np.argsort(prediction[0])[::-1][:3]
                for rank, i in enumerate(top3):
                    medal = ["🥇", "🥈", "🥉"][rank]
                    cls = CLASSES[i]
                    score = float(prediction[0][i]) * 100
                    st.markdown(f"{medal} **{cls.replace('_',' ').title()}** — `{score:.1f}%`")

                st.markdown("#### 📊 All Class Scores")
                st.plotly_chart(make_chart(prediction), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
