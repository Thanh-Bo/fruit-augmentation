"""
augment_preview.py - Hiển thị ảnh gốc và ảnh sau khi tăng cường dữ liệu.

Chức năng:
1. Tự động tìm một ảnh mẫu từ dataset/train/Apple.
2. Áp dụng các kỹ thuật data augmentation.
3. Hiển thị 1 ảnh gốc và 5 ảnh đã augmentation.
4. Lưu kết quả vào results/augmentation_preview.png.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR, RESULTS_DIR, IMG_SIZE,
    CLASS_NAMES,
    AUGMENTATION_CONFIG
)


def find_sample_image():
    """
    Tìm một ảnh mẫu từ dataset/train.
    Ưu tiên lấy từ class Apple, nếu không có thì tìm class khác.

    Returns:
        tuple: (image_path, class_name) hoặc (None, None) nếu không tìm thấy.
    """
    # Thử tìm ảnh trong từng class theo thứ tự ưu tiên
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if not os.path.exists(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            img_path = os.path.join(class_dir, images[0])
            return img_path, class_name

    return None, None


def load_and_preprocess_image(image_path):
    """
    Load ảnh và chuyển thành numpy array (0-255).

    Args:
        image_path: Đường dẫn file ảnh.

    Returns:
        np.ndarray: Ảnh với giá trị pixel [0, 255], shape (IMG_SIZE, IMG_SIZE, 3).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    return img_array


def generate_augmented_images(img_array, num_samples=5):
    """
    Tạo các ảnh augmentation từ ảnh gốc.

    Args:
        img_array: Ảnh gốc, shape (H, W, 3).
        num_samples: Số ảnh augmentation cần tạo.

    Returns:
        list: Danh sách các ảnh augmentation (numpy arrays).
    """
    # Tạo ImageDataGenerator với cấu hình augmentation
    datagen = ImageDataGenerator(
        rotation_range=AUGMENTATION_CONFIG["rotation_range"],
        zoom_range=AUGMENTATION_CONFIG["zoom_range"],
        width_shift_range=AUGMENTATION_CONFIG["width_shift_range"],
        height_shift_range=AUGMENTATION_CONFIG["height_shift_range"],
        horizontal_flip=AUGMENTATION_CONFIG["horizontal_flip"],
        brightness_range=AUGMENTATION_CONFIG["brightness_range"],
        shear_range=AUGMENTATION_CONFIG["shear_range"],
        fill_mode=AUGMENTATION_CONFIG["fill_mode"]
    )

    # Thêm batch dimension cho ảnh: (1, H, W, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Tạo augmented images
    augmented_images = []
    aug_iter = datagen.flow(img_batch, batch_size=1)

    for i in range(num_samples):
        aug_img = next(aug_iter)[0]  # Lấy ảnh đầu tiên trong batch
        # Đảm bảo giá trị trong [0, 255]
        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        augmented_images.append(aug_img)

    return augmented_images


def plot_augmentation_preview(original_img, augmented_images, class_name):
    """
    Vẽ và lưu ảnh so sánh: ảnh gốc + ảnh augmentation.

    Args:
        original_img: Ảnh gốc (numpy array).
        augmented_images: Danh sách ảnh đã augmentation.
        class_name: Tên class của ảnh.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f'Data Augmentation Preview - {class_name}\n'
        'Ảnh gốc và các biến thể sau tăng cường dữ liệu',
        fontsize=14, fontweight='bold'
    )

    # Ảnh gốc ở vị trí đầu tiên
    axes[0, 0].imshow(original_img.astype(np.uint8))
    axes[0, 0].set_title('ẢNH GỐC (Original)', fontsize=11, color='green',
                         fontweight='bold')
    axes[0, 0].axis('off')

    # Tên các kỹ thuật augmentation
    aug_names = [
        'Xoay (Rotation)',
        'Zoom',
        'Dịch ngang (Width Shift)',
        'Dịch dọc (Height Shift)',
        'Lật ngang (Horizontal Flip)'
    ]

    # Hiển thị 5 ảnh augmentation
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, (pos, aug_img, name) in enumerate(
        zip(positions, augmented_images, aug_names)
    ):
        axes[pos[0], pos[1]].imshow(aug_img)
        axes[pos[0], pos[1]].set_title(f'{name}', fontsize=10, color='blue')
        axes[pos[0], pos[1]].axis('off')

    plt.tight_layout()

    # Lưu ảnh
    output_path = os.path.join(RESULTS_DIR, 'augmentation_preview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Đã lưu augmentation preview tại: {output_path}")


def main():
    """
    Hàm chính - tạo ảnh preview augmentation.
    """
    print("\n" + "#" * 60)
    print("#  PREVIEW DATA AUGMENTATION")
    print("#" * 60)

    # Kiểm tra thư mục train
    if not os.path.exists(TRAIN_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục train: {TRAIN_DIR}")
        print("Hãy chạy prepare_dataset.py trước.")
        return

    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Tìm ảnh mẫu
    print("\nĐang tìm ảnh mẫu...")
    image_path, class_name = find_sample_image()

    if image_path is None:
        print("[LỖI] Không tìm thấy ảnh nào trong dataset/train/")
        print("Hãy chạy prepare_dataset.py trước.")
        return

    print(f"[OK] Đã chọn ảnh: {image_path}")
    print(f"  Class: {class_name}")

    # Load ảnh
    print("\nĐang load ảnh...")
    original_img = load_and_preprocess_image(image_path)

    # Tạo augmented images
    print("Đang tạo các ảnh augmentation...")
    augmented_images = generate_augmented_images(original_img, num_samples=5)
    print(f"[OK] Đã tạo {len(augmented_images)} ảnh augmentation.")

    # In các kỹ thuật augmentation đã sử dụng
    print("\nCác kỹ thuật augmentation đã áp dụng:")
    for key, value in AUGMENTATION_CONFIG.items():
        print(f"  - {key}: {value}")

    # Vẽ và lưu preview
    plot_augmentation_preview(original_img, augmented_images, class_name)

    print("\n[HOÀN THÀNH] Mở file results/augmentation_preview.png để xem kết quả.")


if __name__ == "__main__":
    main()
