"""
prepare_dataset.py - Chuan bi dataset cho project phan loai trai cay.

Chuc nang:
1. Kiem tra dataset goc Fruits-360 co ton tai khong.
2. Chon ra 15 class tu dataset goc.
3. Copy anh tu Training vao dataset/selected.
4. Chia selected thanh train (70%), validation (30%).
5. Copy anh tu Test goc vao dataset/test (test that, khong tron).
"""

import os
import shutil
import random
from config import (
    RAW_TRAIN_DIR, RAW_TEST_DIR,
    SELECTED_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR,
    SOURCE_FOLDERS, CLASS_NAMES,
    MAX_IMAGES_PER_CLASS,
    TRAIN_RATIO
)

# Đặt seed để kết quả có thể tái lập
random.seed(42)


def check_raw_dataset():
    """
    Kiểm tra dataset gốc đã được đặt đúng vị trí chưa.

    Returns:
        bool: True nếu dataset tồn tại, False nếu không.
    """
    print("=" * 60)
    print("KIỂM TRA DATASET GỐC")
    print("=" * 60)

    if not os.path.exists(RAW_TRAIN_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục: {RAW_TRAIN_DIR}")
        print("Bạn cần đặt dataset vào đúng vị trí sau:")
        print(f"  {RAW_TRAIN_DIR}")
        print(f"  {RAW_TEST_DIR}")
        print("\nCau truc yeu cau:")
        print("  fruits-360_100x100/fruits-360/")
        print("      +-- Training/")
        print("      │   ├── Apple 5/")
        print("      │   ├── Banana 1/")
        print("      │   ├── Orange 1/")
        print("      │   ├── Mango 1/")
        print("      │   ├── Strawberry 1/")
        print("      │   └── ...")
        print("      └── Test/")
        print("          ├── Apple 5/")
        print("          ├── Banana 1/")
        print("          └── ...")
        return False

    if not os.path.exists(RAW_TEST_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục: {RAW_TEST_DIR}")
        return False

    print(f"[OK] Đã tìm thấy Training tại: {RAW_TRAIN_DIR}")
    print(f"[OK] Đã tìm thấy Test tại: {RAW_TEST_DIR}")

    # Liệt kê các folder có trong Training
    train_folders = [f for f in os.listdir(RAW_TRAIN_DIR)
                     if os.path.isdir(os.path.join(RAW_TRAIN_DIR, f))]
    test_folders = [f for f in os.listdir(RAW_TEST_DIR)
                    if os.path.isdir(os.path.join(RAW_TEST_DIR, f))]

    print(f"\nSố folder trong Training: {len(train_folders)}")
    print(f"Số folder trong Test: {len(test_folders)}")

    return True


def clean_directories():
    """
    Xóa dữ liệu cũ trong các thư mục selected, train, validation, test
    để tránh trùng lặp khi chạy lại script.
    """
    dirs_to_clean = [SELECTED_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[CLEAN] Đã xóa thư mục cũ: {d}")
        os.makedirs(d, exist_ok=True)


def copy_images_to_selected():
    """
    Copy anh TU TRAINING goc vao dataset/selected.
    KHONG tron voi Test - Test se duoc xu ly rieng.

    Voi moi class:
    - Tim folder tuong ung trong RAW_TRAIN_DIR.
    - Copy toi da MAX_IMAGES_PER_CLASS anh vao dataset/selected/<class_name>.
    - Neu khong tim thay folder, in canh bao chi tiet.

    Returns:
        dict: Thong ke so anh da copy cho moi class.
    """
    print("\n" + "=" * 60)
    print("COPY ANH TU TRAINING VAO SELECTED")
    print("=" * 60)

    stats = {}
    all_found = True

    for class_name in CLASS_NAMES:
        dest_dir = os.path.join(SELECTED_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        source_folder_names = SOURCE_FOLDERS.get(class_name, [])
        copied_count = 0
        found_any = False

        for folder_name in source_folder_names:
            # CHI tim trong Training, KHONG tron voi Test
            source_dir = os.path.join(RAW_TRAIN_DIR, folder_name)

            if not os.path.exists(source_dir):
                continue

            found_any = True
            images = [f for f in os.listdir(source_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in images:
                if copied_count >= MAX_IMAGES_PER_CLASS:
                    break

                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)

                base, ext = os.path.splitext(img_name)
                if os.path.exists(dst_path):
                    dst_path = os.path.join(dest_dir, f"{base}_{folder_name}{ext}")

                shutil.copy2(src_path, dst_path)
                copied_count += 1

            if copied_count >= MAX_IMAGES_PER_CLASS:
                break

        stats[class_name] = copied_count

        if not found_any:
            all_found = False
            print(f"\n[CẢNH BÁO] Khong tim thay folder cho class '{class_name}'!")
            print(f"  Da tim cac folder: {source_folder_names}")
            print(f"  Trong Training: {RAW_TRAIN_DIR}")
            print(f"  => Hay kiem tra va sua SOURCE_FOLDERS trong config.py")
            print(f"  => Cac folder hien co trong Training:")
            if os.path.exists(RAW_TRAIN_DIR):
                for f in sorted(os.listdir(RAW_TRAIN_DIR)):
                    if os.path.isdir(os.path.join(RAW_TRAIN_DIR, f)):
                        print(f"       - {f}")
        else:
            print(f"  [{class_name}] Da copy {copied_count} anh")

    if not all_found:
        print("\n" + "!" * 60)
        print("KHONG TIM THAY MOT SO FOLDER!")
        print("Hay sua SOURCE_FOLDERS trong config.py cho dung voi dataset cua ban.")
        print("!" * 60)

    return stats


def copy_test_from_raw():
    """
    Copy anh TU TEST GOC vao dataset/test (test that, khong tron voi train).
    
    Voi moi class:
    - Tim folder tuong ung trong RAW_TEST_DIR.
    - Copy toi da MAX_IMAGES_PER_CLASS anh vao dataset/test/<class_name>.

    Returns:
        dict: Thong ke so anh test cho moi class.
    """
    print("\n" + "=" * 60)
    print("COPY ANH TU TEST GOC VAO DATASET/TEST")
    print("=" * 60)

    test_stats = {}

    for class_name in CLASS_NAMES:
        dest_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        source_folder_names = SOURCE_FOLDERS.get(class_name, [])
        copied_count = 0

        for folder_name in source_folder_names:
            source_dir = os.path.join(RAW_TEST_DIR, folder_name)

            if not os.path.exists(source_dir):
                continue

            images = [f for f in os.listdir(source_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in images:
                if copied_count >= MAX_IMAGES_PER_CLASS:
                    break

                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)

                base, ext = os.path.splitext(img_name)
                if os.path.exists(dst_path):
                    dst_path = os.path.join(dest_dir, f"{base}_{folder_name}{ext}")

                shutil.copy2(src_path, dst_path)
                copied_count += 1

            if copied_count >= MAX_IMAGES_PER_CLASS:
                break

        test_stats[class_name] = copied_count
        print(f"  [{class_name}] Test: {copied_count} anh")

    return test_stats


def split_train_validation():
    """
    Chia anh tu dataset/selected thanh train va validation.
    KHONG tao test tu selected - test lay tu Test goc.

    Voi moi class:
    - Doc danh sach anh trong dataset/selected/<class>.
    - Xao tron va chia thanh train, validation theo ty le.

    Returns:
        dict: Thong ke so anh train/val cho moi class.
    """
    print("\n" + "=" * 60)
    print("CHIA DU LIEU TRAIN / VALIDATION (TU TRAINING GOC)")
    print("=" * 60)
    print(f"Ty le: Train={TRAIN_RATIO}, Validation={1 - TRAIN_RATIO:.2f}")

    split_stats = {}

    for class_name in CLASS_NAMES:
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VALIDATION_DIR, class_name)

        for d in [train_class_dir, val_class_dir]:
            os.makedirs(d, exist_ok=True)

        selected_class_dir = os.path.join(SELECTED_DIR, class_name)
        if not os.path.exists(selected_class_dir):
            print(f"  [{class_name}] Khong co anh trong selected!")
            split_stats[class_name] = {"train": 0, "val": 0}
            continue

        images = [f for f in os.listdir(selected_class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = total - train_count

        train_images = images[:train_count]
        val_images = images[train_count:]

        for img_list, dest_dir in [
            (train_images, train_class_dir),
            (val_images, val_class_dir)
        ]:
            for img_name in img_list:
                src_path = os.path.join(selected_class_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)
                shutil.copy2(src_path, dst_path)

        split_stats[class_name] = {"train": train_count, "val": val_count}

        print(f"  [{class_name}] Tong: {total} | Train: {train_count} | Val: {val_count}")

    return split_stats


def print_summary(selected_stats, split_stats, test_stats):
    """
    In bang tong ket du lieu.

    Args:
        selected_stats: Thong ke anh trong selected (tu Training goc).
        split_stats: Thong ke anh train/val.
        test_stats: Thong ke anh test (tu Test goc).
    """
    print("\n" + "=" * 60)
    print("TONG KET DU LIEU")
    print("=" * 60)
    print(f"{'Class':<15} {'Selected':<12} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 57)

    total_selected = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASS_NAMES:
        sel = selected_stats.get(class_name, 0)
        sp = split_stats.get(class_name, {"train": 0, "val": 0})
        train = sp["train"]
        val = sp["val"]
        test = test_stats.get(class_name, 0)

        total_selected += sel
        total_train += train
        total_val += val
        total_test += test

        print(f"{class_name:<15} {sel:<12} {train:<10} {val:<10} {test:<10}")

    print("-" * 57)
    print(f"{'TONG':<15} {total_selected:<12} {total_train:<10} "
          f"{total_val:<10} {total_test:<10}")
    print(f"\nTrain + Val (tu Training goc): {total_train + total_val}")
    print(f"Test (tu Test goc - KHONG tron): {total_test}")


def main():
    """
    Hàm chính - thực hiện toàn bộ quy trình chuẩn bị dataset.
    """
    print("\n" + "#" * 60)
    print("#  CHUẨN BỊ DATASET CHO PHÂN LOẠI TRÁI CÂY")
    print("#" * 60)

    # Bước 1: Kiểm tra dataset gốc
    if not check_raw_dataset():
        print("\n[THOÁT] Vui lòng đặt dataset đúng vị trí và thử lại.")
        return

    # Bước 2: Xóa dữ liệu cũ
    clean_directories()

    # Buoc 3: Copy anh tu Training vao selected
    selected_stats = copy_images_to_selected()

    # Kiem tra co class nao khong co anh khong
    if all(count == 0 for count in selected_stats.values()):
        print("\n[LOI NGHIEM TRONG] Khong copy duoc anh nao!")
        print("Hay kiem tra SOURCE_FOLDERS trong config.py.")
        return

    # Buoc 4: Chia selected thanh train + validation
    split_stats = split_train_validation()

    # Buoc 5: Copy anh tu Test goc vao dataset/test (test that)
    test_stats = copy_test_from_raw()

    # Buoc 6: In tong ket
    print_summary(selected_stats, split_stats, test_stats)

    print("\n[HOÀN THÀNH] Dataset đã sẵn sàng!")
    print("Bạn có thể chạy train_model.py để huấn luyện mô hình.")


if __name__ == "__main__":
    main()
