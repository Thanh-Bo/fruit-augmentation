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
    MAX_IMAGES_PER_CLASS, IMG_SIZE,
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
    Copy ảnh TỪ TRAINING gốc vào dataset/selected.
    KHÔNG trộn với Test - Test sẽ được xử lý riêng.

    Với mỗi class:
    - Đọc tất cả ảnh từ mọi folder nguồn có tồn tại.
    - Shuffle ảnh trong từng folder.
    - Chia quota đều cho các folder.
    - Nếu folder nào thiếu ảnh thì phân bổ phần còn thiếu cho các folder còn dư.
    - In summary chi tiết số ảnh lấy từ từng folder biến thể.

    Returns:
        dict: Thống kê số ảnh đã copy cho mỗi class, kèm chi tiết từng folder.
    """
    print("\n" + "=" * 60)
    print("COPY ẢNH TỪ TRAINING VÀO SELECTED")
    print("=" * 60)
    print(f"MAX_IMAGES_PER_CLASS = {MAX_IMAGES_PER_CLASS}")
    print("Chiến lược: chia đều quota cho các folder biến thể, shuffle trong folder.\n")

    stats = {}
    all_found = True

    for class_name in CLASS_NAMES:
        dest_dir = os.path.join(SELECTED_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        source_folder_names = SOURCE_FOLDERS.get(class_name, [])

        # Bước 1: Đọc tất cả ảnh từ mọi folder nguồn tồn tại
        folder_images = {}  # {folder_name: [list of image filenames]}
        for folder_name in source_folder_names:
            source_dir = os.path.join(RAW_TRAIN_DIR, folder_name)
            if not os.path.exists(source_dir):
                continue
            images = [f for f in os.listdir(source_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                folder_images[folder_name] = images

        if not folder_images:
            all_found = False
            stats[class_name] = {"total": 0, "folders": {}}
            print(f"\n[CẢNH BÁO] Không tìm thấy folder cho class '{class_name}'!")
            print(f"  Đã tìm các folder: {source_folder_names}")
            print(f"  Trong Training: {RAW_TRAIN_DIR}")
            print(f"  => Hãy kiểm tra và sửa SOURCE_FOLDERS trong config.py")
            if os.path.exists(RAW_TRAIN_DIR):
                print(f"  => Các folder hiện có trong Training:")
                for f in sorted(os.listdir(RAW_TRAIN_DIR)):
                    if os.path.isdir(os.path.join(RAW_TRAIN_DIR, f)):
                        print(f"       - {f}")
            continue

        num_folders = len(folder_images)

        # Bước 2: Shuffle ảnh trong từng folder
        for folder_name in folder_images:
            random.shuffle(folder_images[folder_name])

        # Bước 3: Phân bổ quota đều cho các folder
        base_quota = MAX_IMAGES_PER_CLASS // num_folders
        remaining = MAX_IMAGES_PER_CLASS - base_quota * num_folders

        # Phân bổ ban đầu
        quotas = {}
        for i, folder_name in enumerate(folder_images):
            quotas[folder_name] = base_quota + (1 if i < remaining else 0)

        # Bước 4: Điều chỉnh quota - folder nào thiếu thì phân bổ cho folder dư
        # Lặp đến khi không còn thay đổi
        changed = True
        while changed:
            changed = False
            # Tìm folder thiếu (có ít ảnh hơn quota) và folder dư (có nhiều hơn quota)
            deficit_total = 0
            surplus_folders = []
            for folder_name in folder_images:
                available = len(folder_images[folder_name])
                quota = quotas[folder_name]
                if available < quota:
                    deficit_total += quota - available
                    quotas[folder_name] = available  # Gán lại = số ảnh thực có
                elif available > quota:
                    surplus_folders.append((folder_name, available - quota))

            if deficit_total > 0 and surplus_folders:
                changed = True
                # Phân bổ deficit cho các folder dư, mỗi folder thêm tối đa phần dư của nó
                for folder_name, surplus in surplus_folders:
                    if deficit_total <= 0:
                        break
                    add = min(surplus, deficit_total)
                    quotas[folder_name] += add
                    deficit_total -= add

        # Bước 5: Copy ảnh theo quota đã điều chỉnh
        folder_detail = {}
        total_copied = 0

        for folder_name in source_folder_names:
            if folder_name not in folder_images:
                folder_detail[folder_name] = {"available": 0, "copied": 0, "exists": False}
                continue

            available = len(folder_images[folder_name])
            quota = quotas.get(folder_name, 0)
            copied = 0

            for img_name in folder_images[folder_name][:quota]:
                src_path = os.path.join(RAW_TRAIN_DIR, folder_name, img_name)
                dst_path = os.path.join(dest_dir, img_name)

                base, ext = os.path.splitext(img_name)
                if os.path.exists(dst_path):
                    dst_path = os.path.join(dest_dir, f"{base}_{folder_name}{ext}")

                shutil.copy2(src_path, dst_path)
                copied += 1

            total_copied += copied
            folder_detail[folder_name] = {"available": available, "copied": copied, "exists": True}

        stats[class_name] = {"total": total_copied, "folders": folder_detail}

        # In chi tiết
        print(f"  [{class_name}] Tổng: {total_copied}/{MAX_IMAGES_PER_CLASS} ảnh "
              f"(từ {num_folders} folder biến thể)")
        for folder_name in source_folder_names:
            if folder_name in folder_detail:
                fd = folder_detail[folder_name]
                if fd["exists"]:
                    print(f"      {folder_name:<35} có {fd['available']:>5} ảnh → lấy {fd['copied']:>5}")
                else:
                    print(f"      {folder_name:<35} [KHÔNG TỒN TẠI]")
        print()

    if not all_found:
        print("\n" + "!" * 60)
        print("KHÔNG TÌM THẤY MỘT SỐ FOLDER!")
        print("Hãy sửa SOURCE_FOLDERS trong config.py cho đúng với dataset của bạn.")
        print("!" * 60)

    return stats


def copy_test_from_raw():
    """
    Copy ảnh TỪ TEST GỐC vào dataset/test (test thật, không trộn với train).

    Với mỗi class:
    - Đọc tất cả ảnh từ mọi folder nguồn có tồn tại trong RAW_TEST_DIR.
    - Shuffle ảnh trong từng folder.
    - Chia quota đều cho các folder (giống logic copy_images_to_selected).
    - In summary chi tiết số ảnh lấy từ từng folder biến thể.

    Returns:
        dict: Thống kê số ảnh test cho mỗi class, kèm chi tiết từng folder.
    """
    print("\n" + "=" * 60)
    print("COPY ẢNH TỪ TEST GỐC VÀO DATASET/TEST")
    print("=" * 60)
    print(f"MAX_IMAGES_PER_CLASS = {MAX_IMAGES_PER_CLASS}")
    print("Chiến lược: chia đều quota cho các folder biến thể, shuffle trong folder.\n")

    test_stats = {}

    for class_name in CLASS_NAMES:
        dest_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        source_folder_names = SOURCE_FOLDERS.get(class_name, [])

        # Bước 1: Đọc tất cả ảnh từ mọi folder nguồn tồn tại trong Test
        folder_images = {}
        for folder_name in source_folder_names:
            source_dir = os.path.join(RAW_TEST_DIR, folder_name)
            if not os.path.exists(source_dir):
                continue
            images = [f for f in os.listdir(source_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                folder_images[folder_name] = images

        if not folder_images:
            test_stats[class_name] = {"total": 0, "folders": {}}
            print(f"  [{class_name}] Test: KHÔNG tìm thấy folder nào trong Test gốc")
            continue

        num_folders = len(folder_images)

        # Bước 2: Shuffle ảnh trong từng folder
        for folder_name in folder_images:
            random.shuffle(folder_images[folder_name])

        # Bước 3: Phân bổ quota đều
        base_quota = MAX_IMAGES_PER_CLASS // num_folders
        remaining = MAX_IMAGES_PER_CLASS - base_quota * num_folders

        quotas = {}
        for i, folder_name in enumerate(folder_images):
            quotas[folder_name] = base_quota + (1 if i < remaining else 0)

        # Bước 4: Điều chỉnh quota - folder thiếu → phân bổ cho folder dư
        changed = True
        while changed:
            changed = False
            deficit_total = 0
            surplus_folders = []
            for folder_name in folder_images:
                available = len(folder_images[folder_name])
                quota = quotas[folder_name]
                if available < quota:
                    deficit_total += quota - available
                    quotas[folder_name] = available
                elif available > quota:
                    surplus_folders.append((folder_name, available - quota))

            if deficit_total > 0 and surplus_folders:
                changed = True
                for folder_name, surplus in surplus_folders:
                    if deficit_total <= 0:
                        break
                    add = min(surplus, deficit_total)
                    quotas[folder_name] += add
                    deficit_total -= add

        # Bước 5: Copy ảnh theo quota
        folder_detail = {}
        total_copied = 0

        for folder_name in source_folder_names:
            if folder_name not in folder_images:
                folder_detail[folder_name] = {"available": 0, "copied": 0, "exists": False}
                continue

            available = len(folder_images[folder_name])
            quota = quotas.get(folder_name, 0)
            copied = 0

            for img_name in folder_images[folder_name][:quota]:
                src_path = os.path.join(RAW_TEST_DIR, folder_name, img_name)
                dst_path = os.path.join(dest_dir, img_name)

                base, ext = os.path.splitext(img_name)
                if os.path.exists(dst_path):
                    dst_path = os.path.join(dest_dir, f"{base}_{folder_name}{ext}")

                shutil.copy2(src_path, dst_path)
                copied += 1

            total_copied += copied
            folder_detail[folder_name] = {"available": available, "copied": copied, "exists": True}

        test_stats[class_name] = {"total": total_copied, "folders": folder_detail}

        # In chi tiết
        print(f"  [{class_name}] Test: {total_copied}/{MAX_IMAGES_PER_CLASS} ảnh "
              f"(từ {num_folders} folder)")
        for folder_name in source_folder_names:
            if folder_name in folder_detail:
                fd = folder_detail[folder_name]
                if fd["exists"]:
                    print(f"      {folder_name:<35} có {fd['available']:>5} ảnh → lấy {fd['copied']:>5}")
                else:
                    print(f"      {folder_name:<35} [KHÔNG TỒN TẠI]")
        print()

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
    In bảng tổng kết dữ liệu.

    Args:
        selected_stats: Thống kê ảnh trong selected (từ Training gốc) - dict {class: {total, folders}}.
        split_stats: Thống kê ảnh train/val.
        test_stats: Thống kê ảnh test (từ Test gốc) - dict {class: {total, folders}}.
    """
    print("\n" + "=" * 60)
    print("TỔNG KẾT DỮ LIỆU")
    print("=" * 60)
    print(f"{'Class':<15} {'Selected':<12} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 57)

    total_selected = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASS_NAMES:
        sel_info = selected_stats.get(class_name, {"total": 0})
        sel = sel_info["total"] if isinstance(sel_info, dict) else sel_info
        sp = split_stats.get(class_name, {"train": 0, "val": 0})
        train = sp["train"]
        val = sp["val"]
        test_info = test_stats.get(class_name, {"total": 0})
        test = test_info["total"] if isinstance(test_info, dict) else test_info

        total_selected += sel
        total_train += train
        total_val += val
        total_test += test

        print(f"{class_name:<15} {sel:<12} {train:<10} {val:<10} {test:<10}")

    print("-" * 57)
    print(f"{'TỔNG':<15} {total_selected:<12} {total_train:<10} "
          f"{total_val:<10} {total_test:<10}")
    print(f"\nTrain + Val (từ Training gốc): {total_train + total_val}")
    print(f"Test (từ Test gốc - KHÔNG trộn): {total_test}")
    print(f"\nTỷ lệ Train/Val: {TRAIN_RATIO}/{1-TRAIN_RATIO:.2f}")
    print(f"MAX_IMAGES_PER_CLASS: {MAX_IMAGES_PER_CLASS}")
    print(f"IMG_SIZE: {IMG_SIZE}")


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
    all_zero = all(
        (info["total"] if isinstance(info, dict) else info) == 0
        for info in selected_stats.values()
    )
    if all_zero:
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
