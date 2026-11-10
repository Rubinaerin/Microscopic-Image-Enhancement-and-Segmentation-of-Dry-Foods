import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.pipeline import FoodQualityPipeline

st.set_page_config(page_title="Gıda Kalite Kontrol - MVP", layout="wide")

st.title("🔬 Kuru Gıda Kalite Kontrol İstasyonu")
st.markdown("Mikroskobik/makro gıda görüntüsü yükleyerek yüzey anomalilerini inceleyin.")

# Sol Menü: Parametre Ayarları
st.sidebar.header("⚙️ Pipeline Parametreleri")
blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", min_value=1, max_value=15, value=5, step=2)
open_ksize = st.sidebar.slider("Morphological Opening Kernel", min_value=1, max_value=15, value=3, step=2)
close_ksize = st.sidebar.slider("Morphological Closing Kernel", min_value=1, max_value=15, value=5, step=2)

uploaded_file = st.file_uploader("Görüntü Yükle (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Yüklenen dosyayı OpenCV formatına (BGR) çevir
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Pipeline'ı çalıştır
    pipeline = FoodQualityPipeline()
    results = pipeline.run(
        img_bgr,
        blur_ksize=blur_ksize,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
    )

    # Yan yana 3 sütunlu gösterim
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Orijinal Görüntü")
        st.image(results["original_rgb"], use_container_width=True)

    with col2:
        st.subheader("İyileştirilmiş (Enhanced)")
        st.image(results["enhanced"], use_container_width=True, clamp=True)

    with col3:
        st.subheader("İkili Maske (Binary Mask)")
        st.image(results["final_mask"], use_container_width=True, clamp=True)
else:
    st.info("Lütfen analiz etmek için yukarıdan bir gıda fotoğrafı yükleyin.")