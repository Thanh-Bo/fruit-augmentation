"""
predict.py - Dự đoán loại trái cây từ một ảnh đơn lẻ.

Chức năng:
1. Load model đã huấn luyện.
2. Load file class_indices.json.
3. Đọc và tiền xử lý ảnh đầu vào.
4. Dự đoán class và in kết quả chi tiết.
"""

import os
import json
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    MODEL_PATH, CLASS_INDICES_PATH,
    IMG_SIZE, LOW_CONFIDENCE_THRESHOLD,
    SAMPLE_DIR
)

# Đường dẫn ảnh cần dự đoán (có thể thay đổi)
IMAGE_PATH = os.path.join(SAMPLE_DIR, "test_image.jpg")


def check_prerequisites():
    """
    Kiểm tra các file cần thiết đã tồn tại chưa.

    Returns:
        tuple: (bool, str|None) — (all_ok, found_image_path or None)
    """
    all_ok = True
    found_image = None

    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy model: {MODEL_PATH}")
        print("  Hãy chạy train_model.py trước.")
        all_ok = False

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"[LỖI] Không tìm thấy class_indices: {CLASS_INDICES_PATH}")
        print("  Hãy chạy train_model.py trước.")
        all_ok = False

    # Kiểm tra ảnh có tồn tại không (hỗ trợ nhiều định dạng)
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    for ext in image_extensions:
        test_path = os.path.join(SAMPLE_DIR, f"test_image{ext}")
        if os.path.exists(test_path):
            found_image = test_path
            break
    # Cũng kiểm tra IMAGE_PATH gốc
    if found_image is None and os.path.exists(IMAGE_PATH):
        found_image = IMAGE_PATH

    if found_image is None:
        print(f"[LỖI] Không tìm thấy ảnh trong thư mục: {SAMPLE_DIR}/")
        print(f"  Hãy đặt ảnh muốn dự đoán vào thư mục {SAMPLE_DIR}/")
        print(f"  Tên file: test_image.jpg, test_image.jpeg, test_image.png, hoặc test_image.webp")
        print(f"  Hoặc sửa biến IMAGE_PATH trong predict.py")
        all_ok = False

    return all_ok, found_image


def load_model_and_classes():
    """
    Load model và class_indices.

    Returns:
        tuple: (model, index_to_class_dict)
    """
    print(f"Đang load model từ: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model đã load thành công!")

    print(f"Đang load class indices từ: {CLASS_INDICES_PATH}")
    with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
        index_to_class = json.load(f)

    # Đảm bảo key là int (JSON lưu key dạng string)
    index_to_class = {int(k): v for k, v in index_to_class.items()}
    print(f"[OK] Class indices: {index_to_class}")

    return model, index_to_class


def preprocess_image(image_path):
    """
    Đọc và tiền xử lý ảnh:
    - Resize về IMG_SIZE x IMG_SIZE.
    - Chuyển sang RGB.
    - Chuyển thành numpy array.
    - Chuẩn hóa về [0, 1].

    Args:
        image_path: Đường dẫn đến file ảnh.

    Returns:
        np.ndarray: Ảnh đã tiền xử lý, shape (1, IMG_SIZE, IMG_SIZE, 3).
    """
    # Mở ảnh bằng PIL
    img = Image.open(image_path)

    # Chuyển sang RGB (phòng trường hợp ảnh RGBA hoặc grayscale)
    img = img.convert('RGB')

    # Resize về kích thước chuẩn
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    # Chuyển thành numpy array, chuẩn hóa với MobileNetV2 preprocess_input
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def predict_image(model, img_array, index_to_class):
    """
    Dự đoán class của ảnh.

    Args:
        model: Model đã load.
        img_array: Ảnh đã tiền xử lý.
        index_to_class: Dict ánh xạ index -> class_name.

    Returns:
        tuple: (predicted_class, confidence, all_probs)
    """
    # Dự đoán
    predictions = model.predict(img_array, verbose=0)[0]

    # Lấy class có xác suất cao nhất
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    predicted_class = index_to_class[predicted_index]

    return predicted_class, confidence, predictions


def main():
    """
    Hàm chính - dự đoán ảnh đơn lẻ.
    """
    print("\n" + "#" * 60)
    print("#  DỰ ĐOÁN ẢNH TRÁI CÂY")
    print("#" * 60)

    # Kiểm tra điều kiện tiên quyết
    all_ok, image_path = check_prerequisites()
    if not all_ok:
        return

    print(f"\nẢnh đầu vào: {image_path}")

    # Load model và class indices
    model, index_to_class = load_model_and_classes()

    # Tiền xử lý ảnh
    print(f"\nĐang tiền xử lý ảnh...")
    img_array = preprocess_image(image_path)
    print(f"[OK] Ảnh đã được resize về {IMG_SIZE}x{IMG_SIZE} và chuẩn hóa.")

    # Dự đoán
    print("\nĐang dự đoán...")
    predicted_class, confidence, all_probs = predict_image(
        model, img_array, index_to_class
    )

    # In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("=" * 60)
    print(f"\n  >>> Loại trái cây: {predicted_class}")
    print(f"  >>> Độ tin cậy:    {confidence*100:.2f}%")

    # Cảnh báo nếu confidence thấp
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        print("\n" + "!" * 60)
        print("[CẢNH BÁO] Mô hình không chắc chắn về dự đoán này!")
        print(f"  Confidence ({confidence*100:.2f}%) < Ngưỡng ({LOW_CONFIDENCE_THRESHOLD*100:.0f}%)")
        print("  Ảnh có thể khác phân phối Fruits-360 hoặc không thuộc 15 class.")
        print("!" * 60)

    # In top 5 xác suất cao nhất
    print("\n" + "-" * 60)
    print("TOP 5 DỰ ĐOÁN")
    print("-" * 60)
    print(f"{'Hạng':<6} {'Class':<15} {'Xác suất':<15} {'Bar'}")
    print("-" * 60)

    top5_idx = np.argsort(all_probs)[-5:][::-1]
    for rank, idx in enumerate(top5_idx, 1):
        class_name = index_to_class[idx]
        prob = all_probs[idx]
        bar = "█" * int(prob * 40)
        marker = " <-- DỰ ĐOÁN" if class_name == predicted_class else ""
        print(f"  #{rank}   {class_name:<15} {prob*100:>5.2f}%      {bar}{marker}")

    print("\n[HOÀN THÀNH] Dự đoán kết thúc!")


if __name__ == "__main__":
    main()
