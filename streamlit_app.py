# streamlit_app.py

import streamlit as st
import tempfile
from PIL import Image
from tyre_defect_detector import TyreDefectDetector

# Page configuration
st.set_page_config(page_title="Tyre Defect Detection", layout="centered")

# Combined custom CSS
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 60px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: -webkit-linear-gradient(45deg, #00BFFF, #8A2BE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 20px;
            margin-bottom: 40px;
        }
        .instruction-text {
            font-size: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            color: #444;
            margin-bottom: 20px;
        }
        .uploaded-label > label {
            font-size: 18px !important;
            font-weight: bold;
            color: #2E86C1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .centered-button .stDownloadButton {
            display: block;
            margin: auto;
        }
    </style>

    <div class="title">Tyre Defect Detection</div>
    <div class="instruction-text">📤 Upload an X-ray image of a defect tyre.</div>
""", unsafe_allow_html=True)

# Upload image
with st.container():
    st.markdown('<div class="uploaded-label">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" Upload tyre X-ray image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

# Handle image processing
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        temp_input_path = temp_input_file.name

    # Run defect detection
    detector = TyreDefectDetector(model_path="best.pt")
    output_image_path = detector.run_detection_pipeline(temp_input_path)

    result_img = Image.open(output_image_path)
    st.image(result_img, caption="🛠️ Detected Output", use_container_width=True)

    # Stylish download button
    with open(output_image_path, "rb") as f:
        st.markdown('<div class="centered-button">', unsafe_allow_html=True)
        st.download_button("⬇️ Download Result Image", f, file_name="annotated_tyre.jpg", mime="image/jpeg")
        st.markdown('</div>', unsafe_allow_html=True)
