"""
config.py - File cấu hình chung cho toàn bộ project.
Chứa tất cả các tham số, đường dẫn và cài đặt cho project phân loại trái cây.
"""

import os

# ========================
# Cấu hình ảnh và huấn luyện
# ========================

# Kích thước ảnh đầu vào cho model (pixel x pixel)
IMG_SIZE = 224

# Batch size - số ảnh xử lý trong mỗi lần cập nhật gradient
BATCH_SIZE = 16

# Số epoch huấn luyện Phase 1 (classification head)
EPOCHS = 20

# Số epoch fine-tune (mở khóa base model)
FINE_TUNE_EPOCHS = 10

# Số class cần phân loại
NUM_CLASSES = 15

# Số ảnh tối đa lấy từ mỗi class cho pilot (để phù hợp với máy cá nhân)
MAX_IMAGES_PER_CLASS = 1000

# Learning rate cho Phase 1 (train classification head)
PHASE1_LR = 0.001

# Learning rate cho Phase 2 (fine-tune base model)
PHASE2_LR = 0.00003

# Tỷ lệ lớp cuối của base model được fine-tune (0.35 = 35%)
FINE_TUNE_RATIO = 0.35

# Ngưỡng confidence thấp để cảnh báo khi dự đoán
LOW_CONFIDENCE_THRESHOLD = 0.70

# ========================
# Tỷ lệ chia dữ liệu
# ========================

# Tỷ lệ train (validation = 1 - train_ratio)
TRAIN_RATIO = 0.8

# ========================
# Danh sách class và folder nguồn
# ========================

# Danh sách tên 15 class trái cây cần phân loại
CLASS_NAMES = [
    "Apple", "Banana", "Orange", "Mango", "Strawberry",
    "Watermelon", "Pineapple", "Kiwi", "Lemon", "Cherry",
    "Grape", "Peach", "Pear", "Blueberry", "Avocado"
]

# Ánh xạ tên class sang tên folder thực tế trong dataset Fruits-360
# QUAN TRỌNG: Mỗi class có thể gồm NHIỀU folder (biến thể của cùng 1 loại quả)
SOURCE_FOLDERS = {
    "Apple": [
        "Apple 5", "Apple 6", "Apple 7", "Apple 8", "Apple 9",
        "Apple 10", "Apple 11", "Apple 12", "Apple 13", "Apple 14",
        "Apple 17", "Apple 18", "Apple 19", "Apple 20", "Apple 21",
        "Apple 22", "Apple 23",
        "Apple Braeburn 1", "Apple Crimson Snow 1",
        "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
        "Apple Granny Smith 1", "Apple Pink Lady 1",
        "Apple Red 1", "Apple Red 2", "Apple Red 3",
        "Apple Red Delicious 1",
        "Apple Red Yellow 1", "Apple Red Yellow 2"
    ],
    "Banana": [
        "Banana 1", "Banana 3", "Banana 4",
        "Banana Lady Finger 1", "Banana Red 1"
    ],
    "Orange": [
        "Orange 1", "Orange 2", "Orange 3",
        "orange 4", "Orange peeled 1"
    ],
    "Mango": [
        "Mango 1", "Mango Red 1"
    ],
    "Strawberry": [
        "Strawberry 1", "Strawberry 2", "Strawberry 3",
        "Strawberry Wedge 1"
    ],
    "Watermelon": [
        "Watermelon 1"
    ],
    "Pineapple": [
        "Pineapple 1", "Pineapple Mini 1"
    ],
    "Kiwi": [
        "Kiwi 1"
    ],
    "Lemon": [
        "Lemon 1", "Lemon Meyer 1"
    ],
    "Cherry": [
        "Cherry 1", "Cherry 2", "Cherry 3", "Cherry 4", "Cherry 5",
        "Cherry Rainier 1", "Cherry Rainier 2", "Cherry Rainier 3",
        "Cherry Sour 1",
        "Cherry Wax 1", "Cherry Wax 2",
        "Cherry Wax Black 1", "Cherry Wax Red 1",
        "Cherry Wax Red 2", "Cherry Wax Red 3", "Cherry Wax Yellow 1"
    ],
    "Grape": [
        "Grape 1", "Grape Blue 1", "Grape Pink 1", "Grape pink 2",
        "Grape White 1", "Grape White 2", "Grape White 3", "Grape White 4"
    ],
    "Peach": [
        "Peach 1", "Peach 2", "Peach 3", "Peach 4",
        "Peach 5", "Peach 6", "Peach Flat 1"
    ],
    "Pear": [
        "Pear 1", "Pear 2", "Pear 3", "Pear 5", "Pear 6",
        "Pear 7", "Pear 8", "Pear 9", "Pear 10", "Pear 11",
        "Pear 12", "Pear 13", "Pear 14",
        "Pear Abate 1", "Pear Forelle 1", "Pear Kaiser 1",
        "Pear Monster 1", "Pear Red 1", "Pear Stone 1",
        "Pear Williams 1"
    ],
    "Blueberry": [
        "Blueberry 1"
    ],
    "Avocado": [
        "Avocado 1", "Avocado 2",
        "Avocado Black 1", "Avocado Black 2",
        "Avocado Green 1"
    ]
}

# ========================
# Đường dẫn thư mục
# ========================

# Thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục dataset
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Thư mục dataset gốc (từ Kaggle) - nằm trong thư mục fruits-360_100x100
RAW_DIR = os.path.join(BASE_DIR, "fruits-360_100x100", "fruits-360")
RAW_TRAIN_DIR = os.path.join(RAW_DIR, "Training")
RAW_TEST_DIR = os.path.join(RAW_DIR, "Test")

# Thư mục chứa ảnh đã chọn (15 class)
SELECTED_DIR = os.path.join(DATASET_DIR, "selected")

# Thư mục train, validation, test
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Thư mục lưu model
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_cnn_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Thư mục lưu kết quả đánh giá
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Thư mục ảnh mẫu
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

# ========================
# Data Augmentation (tăng cường dữ liệu)
# ========================

# Cấu hình augmentation cho tập train (vừa phải, cân bằng)
AUGMENTATION_CONFIG = {
    "rotation_range": 30,         # Xoay ảnh ngẫu nhiên trong khoảng [-30, 30] độ
    "zoom_range": 0.3,            # Zoom ảnh ngẫu nhiên trong khoảng [0.7, 1.3]
    "width_shift_range": 0.15,    # Dịch ảnh theo chiều ngang tối đa 15%
    "height_shift_range": 0.15,   # Dịch ảnh theo chiều dọc tối đa 15%
    "horizontal_flip": True,      # Lật ảnh theo chiều ngang
    "brightness_range": [0.7, 1.3], # Thay đổi độ sáng trong khoảng [0.7, 1.3]
    "shear_range": 10,            # Kéo nghiêng ảnh 10 độ
    "fill_mode": "nearest"        # Cách điền pixel trong khi biến đổi
}
