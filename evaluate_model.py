"""
evaluate_model.py - Đánh giá mô hình trên tập test.

Chức năng:
1. Load model đã huấn luyện.
2. Load dữ liệu test.
3. Dự đoán và tính các chỉ số đánh giá:
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, F1-score
   - Classification Report
4. Lưu kết quả vào results/.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    TEST_DIR,
    MODEL_PATH, CLASS_INDICES_PATH,
    IMG_SIZE, BATCH_SIZE,
    RESULTS_DIR
)


def check_prerequisites():
    """
    Kiểm tra các điều kiện tiên quyết:
    - Model đã tồn tại.
    - Dữ liệu test đã tồn tại.
    - File class_indices.json đã tồn tại.

    Returns:
        bool: True nếu tất cả OK, False nếu thiếu.
    """
    all_ok = True

    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy model: {MODEL_PATH}")
        print("  Hãy chạy train_model.py trước để huấn luyện model.")
        all_ok = False

    if not os.path.exists(TEST_DIR):
        print(f"[LỖI] Không tìm thấy thư mục test: {TEST_DIR}")
        print("  Hãy chạy prepare_dataset.py trước.")
        all_ok = False

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"[LỖI] Không tìm thấy file class_indices: {CLASS_INDICES_PATH}")
        print("  Hãy chạy train_model.py trước.")
        all_ok = False

    return all_ok


def load_test_data():
    """
    Load dữ liệu test và nhãn từ TEST_DIR.

    Returns:
        tuple: (X_test, y_true, class_labels)
    """
    print("\nĐang load dữ liệu test...")

    # Không augmentation cho tập test, dùng MobileNetV2 preprocess
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    # Lấy toàn bộ dữ liệu từ generator
    # Reset generator
    test_generator.reset()

    # Dự đoán theo batch
    X_test = []
    y_true = []

    total_batches = len(test_generator)
    for i in range(total_batches):
        X_batch, y_batch = next(test_generator)
        X_test.append(X_batch)
        y_true.append(y_batch)

        if (i + 1) % 10 == 0 or i == total_batches - 1:
            print(f"  Đã xử lý {i + 1}/{total_batches} batches")

    X_test = np.concatenate(X_test, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # Chuyển y_true từ one-hot thành label index
    y_true_labels = np.argmax(y_true, axis=1)

    # Lay class labels tu generator
    generator_labels = list(test_generator.class_indices.keys())

    # Validate class order voi class_indices.json
    with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
        saved_indices = json.load(f)
    saved_labels = [saved_indices[str(i)] for i in range(len(saved_indices))]

    if generator_labels != saved_labels:
        print("[CANH BAO] Thu tu class trong test_generator khac class_indices.json!")
        print(f"  Generator: {generator_labels}")
        print(f"  JSON:      {saved_labels}")
        print("  Se dung thu tu tu class_indices.json de bao dam nhan dung.")
        class_labels = saved_labels
    else:
        class_labels = generator_labels
        print(f"[OK] Thu tu class khop voi class_indices.json")

    print(f"  So anh test: {len(X_test)}")

    return X_test, y_true_labels, class_labels


def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Vẽ và lưu confusion matrix.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        class_labels: Danh sách tên class.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Số lượng ảnh'}
    )
    plt.title('Confusion Matrix - Ma trận nhầm lẫn', fontsize=15, fontweight='bold')
    plt.xlabel('Dự đoán (Predicted)', fontsize=12)
    plt.ylabel('Thực tế (Actual)', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Đã lưu confusion matrix tại: {output_path}")


def save_classification_report_text(y_true, y_pred, class_labels):
    """
    Lưu classification report dạng text vào file.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        class_labels: Danh sách tên class.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_labels,
        digits=4
    )

    output_path = os.path.join(RESULTS_DIR, 'classification_report.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT - BÁO CÁO PHÂN LOẠI\n")
        f.write("=" * 60 + "\n\n")

        # Thêm accuracy tổng
        acc = accuracy_score(y_true, y_pred)
        f.write(f"Accuracy (Độ chính xác tổng): {acc:.4f} ({acc*100:.2f}%)\n\n")

        # Thêm precision, recall, f1 macro
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        f.write(f"Macro Precision: {precision_macro:.4f}\n")
        f.write(f"Macro Recall:    {recall_macro:.4f}\n")
        f.write(f"Macro F1-score:  {f1_macro:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write(str(report))

    print(f"[OK] Đã lưu classification report tại: {output_path}")


def main():
    """
    Hàm chính - đánh giá mô hình.
    """
    print("\n" + "#" * 60)
    print("#  ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("#" * 60)

    # Kiểm tra điều kiện tiên quyết
    if not check_prerequisites():
        return

    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Bước 1: Load model
    print(f"\nĐang load model từ: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model đã được load thành công!")

    # Bước 2: Load dữ liệu test
    X_test, y_true_labels, class_labels = load_test_data()

    # Bước 3: Dự đoán trên tập test
    print("\nĐang dự đoán trên tập test...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    print("[OK] Dự đoán hoàn tất!")

    # Bước 4: Tính các chỉ số đánh giá
    acc = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='macro')
    recall = recall_score(y_true_labels, y_pred_labels, average='macro')
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

    # Bước 5: In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 60)
    print(f"Accuracy  (Độ chính xác):    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (Macro):            {precision:.4f}")
    print(f"Recall    (Macro):            {recall:.4f}")
    print(f"F1-score  (Macro):            {f1:.4f}")

    # In chi tiết từng class
    print("\n" + "-" * 60)
    print("CHI TIẾT TỪNG CLASS")
    print("-" * 60)

    # Tinh precision, recall, f1 cho tung class
    precision_per_class = np.asarray(precision_score(
        y_true_labels, y_pred_labels, average=None
    ))
    recall_per_class = np.asarray(recall_score(
        y_true_labels, y_pred_labels, average=None
    ))
    f1_per_class = np.asarray(f1_score(
        y_true_labels, y_pred_labels, average=None
    ))

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 51)
    for i, class_name in enumerate(class_labels):
        print(f"{class_name:<15} {precision_per_class[i]:<12.4f} "
              f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")

    # Bước 6: Vẽ confusion matrix
    plot_confusion_matrix(y_true_labels, y_pred_labels, class_labels)

    # Bước 7: Lưu classification report
    save_classification_report_text(y_true_labels, y_pred_labels, class_labels)

    print("\n[HOÀN THÀNH] Đánh giá model hoàn tất!")
    print("Kết quả đã được lưu trong thư mục results/.")


if __name__ == "__main__":
    main()
