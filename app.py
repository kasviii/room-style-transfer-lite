import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AI Room Style Transfer", page_icon="🎨", layout="centered")
st.title("🎨 AI Room Style Transfer")
st.markdown("Transform your room into a work of art using neural style transfer.")
st.divider()

STYLES = {
    "Van Gogh — Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "Hokusai — The Great Wave": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg",
    "Picasso — Les Demoiselles": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Les_Demoiselles_d%27Avignon.jpg/600px-Les_Demoiselles_d%27Avignon.jpg",
}

@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(img, max_dim=512):
    if isinstance(img, Image.Image):
        img = img.convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0
    scale = max_dim / max(img.shape[:2])
    new_h = int(img.shape[0] * scale)
    new_w = int(img.shape[1] * scale)
    img = tf.image.resize(img, [new_h, new_w])
    return tf.expand_dims(img, 0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if tensor.ndim == 4:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# ── UI ──
content_file = st.file_uploader("📷 Upload your room photo", type=["jpg","jpeg","png"])

st.subheader("Choose a style")
style_choice = st.radio("Preset styles", list(STYLES.keys()))
custom_style = st.file_uploader("Or upload your own style image", type=["jpg","jpeg","png"])

if content_file and st.button("✨ Apply Style Transfer", type="primary"):
    content_img = Image.open(content_file)

    if custom_style:
        style_img_pil = Image.open(custom_style)
    else:
        import requests
        r = requests.get(STYLES[style_choice], headers={'User-Agent': 'Mozilla/5.0'})
        style_img_pil = Image.open(io.BytesIO(r.content))

    col1, col2 = st.columns(2)
    with col1:
        st.image(content_img, caption="Your Room", use_container_width=True)
    with col2:
        st.image(style_img_pil, caption="Style Reference", use_container_width=True)

    st.divider()

    with st.spinner("Loading model..."):
        model = load_model()

    with st.spinner("Applying style transfer..."):
        content_tensor = load_img(content_img)
        style_tensor = load_img(style_img_pil, max_dim=256)
        stylized = model(tf.constant(content_tensor), tf.constant(style_tensor))[0]
        result = tensor_to_image(stylized)

    st.image(result, caption="✨ Styled Room", use_container_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("⬇️ Download Result", buf.getvalue(), "styled_room.png", "image/png")
