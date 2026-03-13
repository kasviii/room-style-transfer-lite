# 🎨 AI Room Style Transfer (Lite)

A neural style transfer web app that transforms room photos into artistic styles using VGG19.

🚀 **Live Demo:** [room-style-transfer-lite-kasviii.streamlit.app](https://room-style-transfer-lite-kasviii.streamlit.app)

## How It Works
Upload a room photo, pick an artistic style or upload your own, and the app applies neural style transfer to blend the artistic style onto your room while preserving its structure.

## Features
- 3 preset artistic styles (Van Gogh, Hokusai, Picasso)
- Custom style image upload
- Download styled result

## Model
- Architecture: VGG19 (pretrained on ImageNet)
- Method: Neural Style Transfer (Gatys et al., 2015)
- Optimized for CPU deployment (reduced resolution for free-tier memory constraints)
- Full resolution version available in: [room-style-transfer](https://github.com/kasviii/room-style-transfer)

## Stack
- Python, TensorFlow, Streamlit
- Deployed on Streamlit Community Cloud

## Note
This is a memory-optimized lite version for free-tier deployment. Output resolution is reduced to fit within Streamlit Cloud's 1GB RAM limit. The full VGG19 implementation with higher quality results is in the linked repo above.
