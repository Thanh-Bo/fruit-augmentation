"""
app.py - Ứng dụng Streamlit demo phân loại trái cây.

Chức năng:
- Phân loại ảnh trái cây bằng mô hình CNN.
- Xem trước ảnh tăng cường dữ liệu (augmentation).
- Hiển thị kết quả đánh giá mô hình.
- Thông tin hệ thống.
"""

import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH, CLASS_INDICES_PATH,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, EPOCHS, FINE_TUNE_EPOCHS,
    PHASE1_LR, PHASE2_LR, LOW_CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    RESULTS_DIR,
    TEST_DIR
)

# ========================
# Custom CSS - Light Theme
# ========================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc !important;
    }

    /* Main Header styling */
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        padding: 32px 36px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: white !important;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 12px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
        line-height: 1.6;
        color: #e8f5e9 !important;
    }

    /* Stat card */
    .stat-card {
        background: white;
        padding: 20px 16px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02), 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.08);
        border-color: #a5d6a7;
    }
    .stat-card .stat-value {
        font-size: 26px;
        font-weight: 700;
        color: #2e7d32;
        margin: 0;
    }
    .stat-card .stat-label {
        font-size: 12px;
        color: #64748b;
        margin: 6px 0 0 0;
        font-weight: 500;
    }

    /* Result card */
    .result-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Confidence badge */
    .confidence-high {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .confidence-medium {
        background: #fff8e1;
        color: #f57f17;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #ffe082;
    }
    .confidence-low {
        background: #ffebee;
        color: #c62828;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }

    /* Info box & Warning box */
    .info-box {
        background: #e8f5e9;
        padding: 16px 20px;
        border-radius: 12px;
        border-left: 5px solid #43a047;
        margin: 16px 0;
        color: #1b5e20;
        font-size: 14px;
        font-weight: 500;
    }
    .warn-box {
        background: #fff3e0;
        padding: 16px 20px;
        border-radius: 12px;
        border-left: 5px solid #fb8c00;
        margin: 16px 0;
        color: #e65100;
        font-size: 14px;
        font-weight: 500;
    }

    /* Section title */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] h2 {
        color: #1b5e20 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] h3 {
        color: #2e7d32 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        margin-top: 16px;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #334155 !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stSidebar"] .stMarkdown li {
        color: #334155 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
        margin: 16px 0 !important;
    }

    /* Header bar transparent */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.8) !important;
        backdrop-filter: blur(8px);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #64748b !important;
        transition: all 0.2s ease;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #2e7d32 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* File Uploader style override */
    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2e7d32;
    }
    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
        border-radius: 8px;
        padding: 12px;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #1b5e20 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25);
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #2e7d32 !important;
        color: #2e7d32 !important;
    }

    /* Pyplot background */
    .stPyplot {
        background: white;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        border: 1px solid #e2e8f0;
    }

    /* Custom progress bar styles */
    .bar-container {
        background-color: #e2e8f0;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 6px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================
# Hàm tiện ích (giữ nguyên logic ML)
# ========================

@st.cache_resource
def load_model():
    """Load model từ file .h5 (cache để không load lại mỗi lần)."""
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_class_indices():
    """Load class_indices từ file JSON (cache)."""
    if not os.path.exists(CLASS_INDICES_PATH):
        return None
    with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
        index_to_class = json.load(f)
    index_to_class = {int(k): v for k, v in index_to_class.items()}
    return index_to_class


def preprocess_image(image_input):
    """
    Tiền xử lý ảnh từ file upload hoặc đường dẫn file.
    Sử dụng MobileNetV2 preprocess_input (scale về [-1, 1]).
    Returns: (img_array, pil_image)
    """
    pil_image = Image.open(image_input).convert('RGB')
    img_resized = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, pil_image


@st.cache_data
def get_sample_images():
    """Lấy danh sách ảnh mẫu đại diện từ thư mục test."""
    samples = {}
    sample_classes = ["Apple", "Banana", "Orange", "Strawberry", "Mango"]
    for c in sample_classes:
        folder_path = os.path.join(TEST_DIR, c)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if files:
                samples[c] = os.path.join(folder_path, files[0])
    return samples


def predict(model, img_array, index_to_class):
    """
    Dự đoán class của ảnh.
    Returns: (predicted_class, confidence, all_probabilities)
    """
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = index_to_class[predicted_index]
    return predicted_class, confidence, predictions


def get_confidence_level(confidence):
    """Trả về (label, css_class) dựa trên mức độ tin cậy."""
    if confidence >= 0.80:
        return "Mức độ tin cậy cao", "confidence-high"
    elif confidence >= 0.50:
        return "Mức độ tin cậy trung bình", "confidence-medium"
    else:
        return "Mức độ tin cậy thấp", "confidence-low"


# ========================
# Augmentation preview helpers
# ========================
def generate_augmented_variants(pil_image):
    """
    Tạo các biến thể augmentation từ ảnh gốc (dùng Pillow, không dùng TF).
    Returns dict: {tên_kỹ_thuật: PIL.Image}
    """
    variants = {}

    # Lật ngang
    variants["Lật ngang"] = ImageOps.mirror(pil_image)

    # Xoay ảnh (30 độ)
    variants["Xoay ảnh (30°)"] = pil_image.rotate(
        30, expand=True, fillcolor=(255, 255, 255)
    )

    # Tăng độ sáng
    enhancer = ImageEnhance.Brightness(pil_image)
    variants["Tăng độ sáng"] = enhancer.enhance(1.5)

    # Giảm độ sáng
    variants["Giảm độ sáng"] = enhancer.enhance(0.5)

    # Tăng độ tương phản
    contrast = ImageEnhance.Contrast(pil_image)
    variants["Tăng độ tương phản"] = contrast.enhance(1.8)

    # Ảnh xám
    variants["Ảnh xám"] = ImageOps.grayscale(pil_image).convert('RGB')

    return variants


# ========================
# Matplotlib chart helpers
# ========================
def plot_top5_prediction(class_names, probs, predicted_class):
    """
    Vẽ biểu đồ top 5 xác suất dự đoán, highlight class được chọn.
    """
    # Lấy top 5 index
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_names = [class_names[i] for i in top5_idx]
    top5_vals = [probs[i] * 100 for i in top5_idx]

    colors = [
        '#43a047' if class_names[i] == predicted_class else '#bdbdbd'
        for i in top5_idx
    ]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.barh(
        top5_names[::-1], top5_vals[::-1],
        color=colors[::-1], height=0.55
    )

    for bar, val in zip(bars, top5_vals[::-1]):
        ax.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontsize=10, color='#333'
        )

    ax.set_xlabel('Xác suất (%)', fontsize=11, color='#555')
    ax.set_title(
        'Top 5 kết quả dự đoán', fontsize=13,
        fontweight='bold', color='#1a1a2e', pad=10
    )
    ax.set_xlim(0, max(top5_vals) * 1.25)
    ax.tick_params(labelsize=10, colors='#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e0e0e0')
    ax.spines['bottom'].set_color('#e0e0e0')

    plt.tight_layout()
    return fig


# ========================
# Sidebar
# ========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## Phân loại trái cây")
        st.markdown("---")

        st.markdown("### Thông tin đề tài")
        st.markdown("""
        - **Đề tài:** Xây dựng ứng dụng tăng cường dữ liệu ảnh và phân loại ảnh trái cây
        - **Môn học:** Khai phá dữ liệu
        """)

        st.markdown("---")

        st.markdown("### Công nghệ sử dụng")
        st.markdown("""
        - Python & Streamlit
        - TensorFlow / Keras
        - MobileNetV2 Transfer Learning
        - Data Augmentation
        """)

        st.markdown("---")

        st.markdown("### Dữ liệu & Mô hình")
        st.markdown(f"""
        - **Dataset:** Fruits-360 (Kaggle)
        - **Kích thước ảnh gốc:** 100x100 pixel
        - **Kích thước đầu vào:** {IMG_SIZE}x{IMG_SIZE} pixel
        - **Epochs:** {EPOCHS} + {FINE_TUNE_EPOCHS} fine-tune
        - **Learning rate:** {PHASE1_LR} / {PHASE2_LR}
        - **Mô hình:** MobileNetV2 Transfer Learning
        """)

        st.markdown("---")

        with st.expander("Danh sách 15 lớp phân loại", expanded=False):
            st.markdown("\n".join([f"- **{name}**" for name in CLASS_NAMES]))

        st.markdown("---")
        st.caption("Ứng dụng phát triển cho mục đích học tập.")


# ========================
# Header
# ========================
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Ứng dụng tăng cường dữ liệu ảnh và phân loại trái cây</h1>
        <p>
            Ứng dụng sử dụng MobileNetV2 Transfer Learning để phân loại ảnh trái cây.
            Dữ liệu được tăng cường bằng các kỹ thuật xoay, lật, zoom,
            dịch ảnh và thay đổi độ sáng nhằm cải thiện khả năng học
            của mô hình.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_stat_cards():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-value">15</p>
            <p class="stat-label">Lớp trái cây</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-value">MobileNetV2</p>
            <p class="stat-label">Mô hình phân loại</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-value">Fruits-360</p>
            <p class="stat-label">Bộ dữ liệu huấn luyện</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <p class="stat-value">Streamlit</p>
            <p class="stat-label">Giao diện ứng dụng</p>
        </div>
        """, unsafe_allow_html=True)


# ========================
# Tab 1: Phân loại ảnh
# ========================
def render_tab_classification():
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    st.markdown(
        '<p class="section-title">1. Tải ảnh lên hoặc chọn ảnh mẫu</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh trái cây để phân loại (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Hỗ trợ định dạng JPG, JPEG, PNG, WEBP",
        key="classify_uploader",
        label_visibility="collapsed"
    )

    # Reset selected sample if a new file is uploaded
    if uploaded_file is not None:
        st.session_state.selected_sample = None

    # Get and show sample gallery
    samples = get_sample_images()
    if samples:
        st.markdown("<p style='font-size:13px;font-weight:600;margin:12px 0 6px 0;color:#64748b;'>Hoặc chọn nhanh từ thư viện ảnh chạy thử mẫu:</p>", unsafe_allow_html=True)
        cols_samples = st.columns(len(samples))
        for idx, (class_name, img_path) in enumerate(samples.items()):
            with cols_samples[idx]:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                    if st.button(f"Chọn {class_name}", key=f"btn_sample_{class_name}", use_container_width=True):
                        st.session_state.selected_sample = img_path
                        st.rerun()
                except Exception:
                    pass

    # Determine source image
    image_input = None
    if uploaded_file is not None:
        image_input = uploaded_file
    elif st.session_state.selected_sample is not None:
        image_input = st.session_state.selected_sample

    # Display preview and classification result if image exists
    if image_input is not None:
        st.markdown("---")
        
        # UI layout for preview and processing indicator
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.markdown('<p class="section-title" style="margin-top:0;">Ảnh đang phân tích</p>', unsafe_allow_html=True)
            try:
                pil_img = Image.open(image_input).convert('RGB')
                st.image(pil_img, use_container_width=True)
                
                # Show clear sample button if using sample image
                if st.session_state.selected_sample is not None and uploaded_file is None:
                    if st.button("Xóa ảnh mẫu", key="clear_sample_btn", use_container_width=True):
                        st.session_state.selected_sample = None
                        st.rerun()
            except Exception as e:
                st.error(f"Không thể mở ảnh: {e}")
                return

        with col_info:
            model = load_model()
            index_to_class = load_class_indices()

            if model is None:
                st.markdown("""
                <div class="warn-box">
                    <b>Chưa tìm thấy mô hình đã huấn luyện.</b><br>
                    Vui lòng chạy quá trình huấn luyện trước khi sử dụng tính năng này.
                </div>
                """, unsafe_allow_html=True)
                return

            if index_to_class is None:
                st.markdown("""
                <div class="warn-box">
                    <b>Chưa tìm thấy file ánh xạ lớp.</b><br>
                    Vui lòng huấn luyện mô hình để tạo file class_indices.json.
                </div>
                """, unsafe_allow_html=True)
                return

            with st.spinner("Đang phân tích ảnh bằng mô hình CNN..."):
                img_array, pil_image = preprocess_image(image_input)
                predicted_class, confidence, all_probs = predict(
                    model, img_array, index_to_class
                )

            # Show results card
            conf_label, conf_css = get_confidence_level(confidence)
            class_names_list = [
                index_to_class[i] for i in range(len(all_probs))
            ]

            st.markdown('<p class="section-title" style="margin-top:0;">Kết quả phân loại</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; align-items:center; gap:32px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:180px;">
                        <p style="font-size:12px; color:#64748b; margin:0; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Loại quả dự đoán</p>
                        <p style="font-size:32px; font-weight:800; color:#2e7d32; margin:6px 0;">{predicted_class}</p>
                        <span class="{conf_css}">{conf_label}</span>
                    </div>
                    <div style="flex:1; min-width:150px; border-left:1px solid #e2e8f0; padding-left:32px;">
                        <p style="font-size:12px; color:#64748b; margin:0; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Độ tin cậy</p>
                        <p style="font-size:36px; font-weight:800; color:#1e293b; margin:6px 0;">{confidence*100:.2f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Cảnh báo nếu confidence thấp
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.markdown(f"""
                <div class="warn-box">
                    <b>⚠️ Mô hình không chắc chắn về dự đoán này.</b><br>
                    Confidence ({confidence*100:.1f}%) thấp hơn ngưỡng an toàn ({LOW_CONFIDENCE_THRESHOLD*100:.0f}%).<br>
                    Ảnh có thể khác phân phối Fruits-360 hoặc không thuộc 15 class được huấn luyện.
                </div>
                """, unsafe_allow_html=True)

        # Charts & Detailed Probabilities Grid side-by-side
        st.markdown("---")
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            st.markdown(
                '<p class="section-title">Top 5 dự đoán nhiều khả năng nhất</p>',
                unsafe_allow_html=True
            )
            fig = plot_top5_prediction(
                class_names_list, all_probs, predicted_class
            )
            st.pyplot(fig)
            plt.close()

        with col_table:
            st.markdown(
                '<p class="section-title">Phân phối xác suất 15 lớp quả</p>',
                unsafe_allow_html=True
            )

            num_classes = len(all_probs)
            cols_per_row = 3
            num_rows = (num_classes + cols_per_row - 1) // cols_per_row

            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    i = row * cols_per_row + col_idx
                    if i >= num_classes:
                        break
                    class_name = class_names_list[i]
                    prob = all_probs[i] * 100

                    is_pred = (class_name == predicted_class)
                    fill_color = "#2e7d32" if is_pred else "#94a3b8"
                    bg_color = "#e8f5e9" if is_pred else "#ffffff"
                    border_style = "2px solid #2e7d32" if is_pred else "1px solid #e2e8f0"
                    text_weight = "700" if is_pred else "500"

                    cols[col_idx].markdown(f"""
                    <div style="
                        background-color: {bg_color};
                        padding: 10px 8px;
                        border-radius: 10px;
                        border: {border_style};
                        margin-bottom: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.01);
                    ">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="font-weight:{text_weight}; font-size:11px; color:#334155;">{class_name}</span>
                            <span style="font-weight:700; font-size:11px; color:{fill_color};">{prob:.1f}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill" style="background-color:{fill_color}; width:{prob}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "Lưu ý: Mô hình chỉ được huấn luyện trên dataset Fruits-360 (nền trắng, studio). "
            "Ảnh ngoài đời thực có thể cho kết quả không chính xác. "
            "Mô hình chỉ phân loại được 15 loại trái cây kể trên — "
            "ảnh các loại khác sẽ bị gán nhầm vào 1 trong 15 class."
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; background-color:white; border-radius:12px; border:1px solid #e2e8f0; margin-top:20px;">
            <p style="color:#64748b; font-size:15px; margin:0; font-weight:500;">
                Vui lòng kéo thả ảnh, nhấn chọn file hoặc click chọn một ảnh mẫu thử nghiệm ở trên để bắt đầu phân loại tự động.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ========================
# Tab 2: Tăng cường dữ liệu
# ========================
def render_tab_augmentation():
    st.markdown("""
    <div class="info-box">
        <b>Tăng cường dữ liệu ảnh</b> giúp tạo ra nhiều biến thể
        từ ảnh gốc, giúp mô hình học tốt hơn và giảm hiện tượng
        quá khớp (overfitting).
    </div>
    """, unsafe_allow_html=True)

    aug_file = st.file_uploader(
        "Tải lên một ảnh để xem các biến thể tăng cường",
        type=["jpg", "jpeg", "png", "webp"],
        help="Hỗ trợ định dạng JPG, JPEG, PNG, WEBP",
        key="augment_uploader"
    )

    if aug_file is not None:
        pil_image = Image.open(aug_file).convert('RGB')
        variants = generate_augmented_variants(pil_image)

        st.markdown("---")
        st.markdown(
            '<p class="section-title">Ảnh gốc</p>',
            unsafe_allow_html=True
        )
        st.image(pil_image, width=250)

        st.markdown("---")
        st.markdown(
            '<p class="section-title">Các biến thể sau tăng cường</p>',
            unsafe_allow_html=True
        )

        # Hiển thị grid 3 cột x 2 hàng
        variant_items = list(variants.items())
        for row_start in range(0, len(variant_items), 3):
            cols = st.columns(3)
            for col_idx in range(3):
                idx = row_start + col_idx
                if idx >= len(variant_items):
                    break
                name, img = variant_items[idx]
                with cols[col_idx]:
                    st.image(img, caption=name, use_container_width=True)
    else:
        st.markdown("""
        <div class="result-card" style="text-align:center;padding:40px;">
            <p style="color:#888;font-size:15px;margin:0;">
                Vui lòng tải lên một ảnh để xem các biến thể
                tăng cường dữ liệu.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ========================
# Tab 3: Kết quả mô hình
# ========================
def render_tab_results():
    st.markdown("### Kết quả đánh giá mô hình")

    # Accuracy/Loss chart
    st.markdown(
        '<p class="section-title">Biểu đồ Accuracy/Loss</p>',
        unsafe_allow_html=True
    )
    acc_loss_path = os.path.join(RESULTS_DIR, 'accuracy_loss.png')
    if os.path.exists(acc_loss_path):
        st.image(acc_loss_path, use_container_width=True)
    else:
        st.markdown("""
        <div class="warn-box">
            Chưa tìm thấy biểu đồ accuracy/loss.
            Vui lòng chạy <code>python train_model.py</code> để tạo.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Confusion matrix
    st.markdown(
        '<p class="section-title">Ma trận nhầm lẫn</p>',
        unsafe_allow_html=True
    )
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        st.markdown("""
        <div class="warn-box">
            Chưa tìm thấy ma trận nhầm lẫn.
            Vui lòng chạy <code>python evaluate_model.py</code> để tạo.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Classification report
    st.markdown(
        '<p class="section-title">Báo cáo phân loại</p>',
        unsafe_allow_html=True
    )
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        st.code(report_content, language=None)
    else:
        st.markdown("""
        <div class="warn-box">
            Chưa tìm thấy báo cáo phân loại.
            Vui lòng chạy <code>python evaluate_model.py</code> để tạo.
        </div>
        """, unsafe_allow_html=True)

    # Augmentation preview
    st.markdown("---")
    st.markdown(
        '<p class="section-title">Xem trước tăng cường dữ liệu</p>',
        unsafe_allow_html=True
    )
    aug_path = os.path.join(RESULTS_DIR, 'augmentation_preview.png')
    if os.path.exists(aug_path):
        st.image(aug_path, use_container_width=True)
    else:
        st.markdown("""
        <div class="warn-box">
            Chưa tìm thấy ảnh xem trước tăng cường dữ liệu.
            Vui lòng chạy <code>python augment_preview.py</code> để tạo.
        </div>
        """, unsafe_allow_html=True)


# ========================
# Tab 4: Thông tin hệ thống
# ========================
def render_tab_system_info():
    st.markdown("### Thông tin hệ thống")

    info_items = [
        ("Dữ liệu", "Fruits-360 (Kaggle)"),
        ("Mô hình", "MobileNetV2 Transfer Learning"),
        ("Kích thước ảnh đầu vào", f"{IMG_SIZE} x {IMG_SIZE} pixel"),
        ("Số lượng lớp", str(len(CLASS_NAMES))),
        ("Định dạng ảnh hỗ trợ", "JPG, JPEG, PNG, WEBP"),
        ("Số epoch huấn luyện", f"{EPOCHS} + {FINE_TUNE_EPOCHS} fine-tune (có EarlyStopping)"),
        ("Batch size", str(BATCH_SIZE)),
        (
            "Kỹ thuật tăng cường",
            "Xoay, Zoom, Dịch ảnh, Lật ngang, "
            "Thay đổi độ sáng, Kéo nghiêng"
        ),
        ("Tối ưu hóa", f"Adam (lr={PHASE1_LR} / {PHASE2_LR})"),
        ("Hàm mất mát", "Categorical Crossentropy"),
    ]

    for label, value in info_items:
        st.markdown(f"""
        <div class="result-card"
             style="padding:14px 20px;margin-bottom:8px;">
            <span style="font-weight:600;color:#555;">{label}:</span>
            <span style="color:#1a1a2e;margin-left:8px;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Danh sách lớp phân loại")
    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        col_idx = i % 3
        cols[col_idx].markdown(f"- **{name}**")


# ========================
# MAIN
# ========================
def main():
    # Cấu hình trang
    st.set_page_config(
        page_title="Phân loại trái cây - CNN",
        layout="wide"
    )

    # Inject CSS
    inject_css()

    # Sidebar
    render_sidebar()

    # Header
    render_header()

    # Stat cards
    render_stat_cards()

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Phân loại ảnh",
        "Tăng cường dữ liệu",
        "Kết quả mô hình",
        "Thông tin hệ thống"
    ])

    with tab1:
        render_tab_classification()

    with tab2:
        render_tab_augmentation()

    with tab3:
        render_tab_results()

    with tab4:
        render_tab_system_info()


if __name__ == "__main__":
    main()

