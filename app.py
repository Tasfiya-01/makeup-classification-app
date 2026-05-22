import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown
import plotly.graph_objects as go

st.set_page_config(
    page_title="Makeup Style Classifier",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    h1 { 
        background: linear-gradient(90deg, #e94560, #f5a7b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }
    .result-card {
        background: rgba(233, 69, 96, 0.15);
        border: 2px solid #e94560;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .confidence-badge {
        background: linear-gradient(90deg, #e94560, #c62a47);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

CLASSES = [
    "bold_makeup",
    "casual_makeup",
    "evening_glamour_makeup",
    "evening_makeup",
    "fantasy_makeup",
    "no_makeup",
    "smokey_eyes_makeup",
    "vintage_makeup"
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


@tf.keras.utils.register_keras_serializable()
class GetItem(tf.keras.layers.Layer):
    def call(self, inputs, idx=0):
        return inputs[idx]

    def get_config(self):
        return super().get_config()


@st.cache_resource
def load_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model... please wait.")
        file_id = "https://drive.google.com/drive/folders/1okMi772YSi2zUz6Ys9beeBF0lynKmMNu?usp=drive_link"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )
    return model


def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image.convert("RGB")) / 255.0
    return np.expand_dims(img_array, axis=0)


def make_chart(prediction):
    scores = [float(prediction[0][i]) * 100 for i in range(len(CLASSES))]
    labels = [f"{CLASS_INFO[c]['emoji']} {c.replace('_', ' ').title()}" for c in CLASSES]
    colors = ["#e94560" if s == max(scores) else "#3a3a6e" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="#ffffff22", width=1)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 115]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
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
        st.image(image, caption="Uploaded Image", use_container_width=True)

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
                    <div style="font-size:1.5rem; font-weight:800; color:white; margin:8px 0">
                        {top_class.replace('_',' ').title()}
                    </div>
                    <div class="confidence-badge">🎯 {confidence:.1f}% Confidence</div>
                    <div style="color:#ccc; margin-top:8px">{info['desc']}</div>
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
