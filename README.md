# Xây dựng ứng dụng tăng cường dữ liệu ảnh và phân loại ảnh trái cây

## Mục tiêu đề tài

Xây dựng ứng dụng sử dụng **Transfer Learning với MobileNetV2** kết hợp với **Data Augmentation** để phân loại 15 loại trái cây: Apple, Avocado, Banana, Blueberry, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Strawberry, Watermelon.

## Dataset sử dụng

- **Fruits-360** từ Kaggle: https://www.kaggle.com/moltean/fruits
- **Phiên bản**: `fruits-360_100x100` (ảnh 100x100 pixels)
- **Lý do chọn**: Dataset chất lượng cao, ảnh chụp thực tế trái cây trên nền trắng, có nhiều class, thuận tiện cho việc học tập và nghiên cứu.

### Vì sao chọn bản 100x100?

Bản `fruits-360_100x100` (ảnh 100x100) có dung lượng nhẹ hơn bản gốc (100MB vs ~1GB), phù hợp với máy cá nhân có CPU Intel Core i5, RAM 16GB, không có GPU NVIDIA.

## 15 Lớp sử dụng

| STT | Class       | Tên tiếng Việt |
|-----|-------------|----------------|
| 1   | Apple       | Táo            |
| 2   | Avocado     | Bơ             |
| 3   | Banana      | Chuối          |
| 4   | Blueberry   | Việt quất      |
| 5   | Cherry      | Anh đào        |
| 6   | Grape       | Nho            |
| 7   | Kiwi        | Kiwi           |
| 8   | Lemon       | Chanh          |
| 9   | Mango       | Xoài           |
| 10  | Orange      | Cam            |
| 11  | Peach       | Đào            |
| 12  | Pear        | Lê             |
| 13  | Pineapple   | Dứa            |
| 14  | Strawberry  | Dâu tây        |
| 15  | Watermelon  | Dưa hấu        |

## Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Python 3.10+** | Ngôn ngữ lập trình chính |
| **TensorFlow/Keras** | Xây dựng và huấn luyện mô hình |
| **MobileNetV2 (Transfer Learning)** | Mô hình pre-trained trên ImageNet, fine-tune cho trái cây |
| **Streamlit** | Xây dựng ứng dụng web demo |
| **NumPy** | Xử lý mảng và tính toán số |
| **PIL/Pillow** | Đọc và xử lý ảnh |
| **Matplotlib + Seaborn** | Trực quan hóa dữ liệu |
| **Scikit-learn** | Đánh giá mô hình (metrics) |

## Cấu trúc thư mục

```
fruit_augmentation_classification/
│
├── fruits-360_100x100/           <-- Dataset gốc (đặt ở đây)
│   └── fruits-360/
│       ├── Training/
│       │   ├── Apple 5/
│       │   ├── Banana 1/
│       │   ├── Orange 1/
│       │   ├── Mango 1/
│       │   ├── Strawberry 1/
│       │   └── ...
│       └── Test/
│           ├── Apple 5/
│           ├── Banana 1/
│           └── ...
│
├── dataset/
│   ├── selected/                 <-- Ảnh 15 class đã chọn
│   │   ├── Apple/
│   │   ├── Avocado/
│   │   ├── Banana/
│   │   ├── Blueberry/
│   │   ├── Cherry/
│   │   ├── Grape/
│   │   ├── Kiwi/
│   │   ├── Lemon/
│   │   ├── Mango/
│   │   ├── Orange/
│   │   ├── Peach/
│   │   ├── Pear/
│   │   ├── Pineapple/
│   │   ├── Strawberry/
│   │   └── Watermelon/
│   │
│   ├── train/                    <-- Dữ liệu train (70%)
│   ├── validation/               <-- Dữ liệu validation (30%)
│   └── test/                     <-- Dữ liệu test (từ Test gốc, KHÔNG trộn)
│
├── model/
│   ├── fruit_cnn_model.h5        <-- Mô hình đã huấn luyện
│   └── class_indices.json        <-- Ánh xạ class -> index
│
├── results/                      <-- Kết quả đánh giá
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── augmentation_preview.png
│
├── sample_images/                <-- Ảnh mẫu để test predict.py
│
├── config.py                     <-- Cấu hình chung
├── prepare_dataset.py            <-- Chuẩn bị dataset
├── train_model.py                <-- Huấn luyện Transfer Learning
├── evaluate_model.py             <-- Đánh giá mô hình
├── predict.py                    <-- Dự đoán ảnh đơn
├── augment_preview.py            <-- Xem ảnh augmentation
├── app.py                        <-- Ứng dụng Streamlit
├── requirements.txt              <-- Thư viện cần cài
├── .gitignore                    <-- Loại trừ dataset khỏi Git
└── README.md                     <-- File này
```

## Mô tả các file Python

| File | Chức năng |
|------|-----------|
| `config.py` | File cấu hình trung tâm: đường dẫn, tham số huấn luyện, danh sách class, cấu hình augmentation |
| `prepare_dataset.py` | Đọc dataset gốc từ `fruits-360_100x100/`, chọn 15 class, copy vào `dataset/selected/`, chia train/val/test |
| `train_model.py` | Xây dựng mô hình MobileNetV2 Transfer Learning, huấn luyện 2 pha, lưu model và biểu đồ |
| `evaluate_model.py` | Đánh giá mô hình trên tập test: accuracy, confusion matrix, precision/recall/F1 từng class |
| `predict.py` | Dự đoán 1 ảnh đơn lẻ từ dòng lệnh, in kết quả chi tiết |
| `augment_preview.py` | Tạo ảnh so sánh: 1 ảnh gốc + 5 ảnh augmentation, lưu vào `results/` |
| `app.py` | Ứng dụng Streamlit 4 tab: phân loại ảnh, tăng cường dữ liệu, kết quả mô hình, thông tin hệ thống |

---

## Hướng dẫn chạy từng bước

### Điều kiện tiên quyết

- Đã cài Python 3.10 trở lên
- Dataset `fruits-360_100x100` đã được giải nén vào thư mục `fruits-360_100x100/` trong project
- Cấu trúc dataset phải có `fruits-360/Training/` và `fruits-360/Test/`

Kiểm tra nhanh: mở `fruits-360_100x100/fruits-360/Training/` phải thấy các folder như `Apple 5`, `Banana 1`,...

---

### Bước 1: Mở terminal tại thư mục project

```bash
cd E:\kbdl\fruit_augmentation_classification
```

---

### Bước 2: Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu máy bạn dùng Python 3.12+, có thể gặp lỗi với tensorflow. Dùng lệnh này thay thế:

```bash
pip install tensorflow==2.16.1 streamlit numpy pillow matplotlib opencv-python scikit-learn seaborn
```

---

### Bước 3: Chuẩn bị dataset

```bash
python prepare_dataset.py
```

Script này sẽ:
- Kiểm tra `fruits-360_100x100/fruits-360/Training/` và `Test/` có tồn tại không
- Tìm các folder tương ứng với 15 class trong dataset
- Nếu không tìm thấy folder nào, sẽ in cảnh báo kèm danh sách folder hiện có để bạn sửa `config.py`
- Copy tối đa 400 ảnh/class vào `dataset/selected/`
- Chia selected thành train (70%) và validation (30%)
- Copy ảnh từ Test gốc vào `dataset/test/` (test thật, không trộn với train)

**Lưu ý**: Nếu script báo không tìm thấy folder, mở `config.py` và sửa `SOURCE_FOLDERS` cho đúng tên folder thực tế.

---

### Bước 4: Huấn luyện mô hình Transfer Learning

```bash
python train_model.py
```

Quá trình huấn luyện (2 pha tự động):
- **Phase 1 — Huấn luyện Classification Head (base đóng băng)**:
  - Chỉ các lớp Dense mới được huấn luyện (~130K tham số)
  - Base MobileNetV2 trích xuất đặc trưng từ ảnh
  - Tối đa 25 epoch, có EarlyStopping (patience=5)
  - ReduceLROnPlateau: giảm learning rate khi val_loss ngừng cải thiện
- **Phase 2 — Fine-tune Base Model**:
  - Mở khóa 20% lớp cuối của MobileNetV2
  - Huấn luyện với learning rate thấp (0.0001)
  - Tối đa 7 epoch, có EarlyStopping (patience=5)
- ModelCheckpoint: lưu model tốt nhất vào `model/fruit_cnn_model.h5`
- Lưu biểu đồ accuracy/loss vào `results/accuracy_loss.png` (có đánh dấu điểm bắt đầu fine-tune)

Thời gian ước tính: 25-40 phút trên CPU (i5-13500H, 16GB RAM).

Lần chạy đầu tiên sẽ tự động tải MobileNetV2 weights (~14MB) từ internet.

---

### Bước 5: Đánh giá mô hình

```bash
python evaluate_model.py
```

Kết quả:
- Test Accuracy
- Confusion Matrix -> `results/confusion_matrix.png`
- Precision, Recall, F1-score (macro + từng class)
- Classification Report -> `results/classification_report.txt`

---

### Bước 6: Xem ảnh augmentation

```bash
python augment_preview.py
```

Tạo ảnh so sánh: 1 ảnh gốc + 5 ảnh sau augmentation -> `results/augmentation_preview.png`

---

### Bước 7: Dự đoán ảnh đơn lẻ

Trước tiên, copy một ảnh trái cây (jpg/png) vào `sample_images/` và đặt tên là `test_image.jpg`.

Sau đó chạy:

```bash
python predict.py
```

Kết quả in ra:
- Tên class dự đoán
- Độ tin cậy (confidence %)
- Xác suất chi tiết từng class

---

### Bước 8: Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`, upload ảnh trái cây và xem kết quả dự đoán.

Ứng dụng có 4 tab chính:
- **Phân loại ảnh**: Upload ảnh và dự đoán loại trái cây
- **Tăng cường dữ liệu**: Xem các biến thể augmentation từ ảnh gốc
- **Kết quả mô hình**: Xem biểu đồ accuracy/loss, confusion matrix, báo cáo phân loại
- **Thông tin hệ thống**: Thông tin về dataset, mô hình, tham số huấn luyện

---

## Data Augmentation (Tăng cường dữ liệu)

Data Augmentation là kỹ thuật tạo ra các biến thể của ảnh gốc bằng cách áp dụng các phép biến đổi ngẫu nhiên. Điều này giúp:

- **Tăng số lượng dữ liệu huấn luyện** mà không cần thu thập thêm ảnh
- **Giảm overfitting** — mô hình học được các đặc trưng bất biến với phép biến đổi
- **Cải thiện khả năng tổng quát hóa** của mô hình

### Các kỹ thuật đã sử dụng

| Kỹ thuật | Tham số | Mô tả |
|----------|---------|-------|
| **Rotation** | ±30° | Xoay ảnh ngẫu nhiên |
| **Zoom** | 0.7x - 1.3x | Phóng to/thu nhỏ |
| **Width Shift** | ±15% | Dịch ảnh theo chiều ngang |
| **Height Shift** | ±15% | Dịch ảnh theo chiều dọc |
| **Horizontal Flip** | Có | Lật ảnh theo chiều ngang |
| **Brightness** | 0.7 - 1.3 | Thay đổi độ sáng |
| **Shear** | 10° | Kéo nghiêng ảnh |

---

## Kiến trúc mô hình (Transfer Learning MobileNetV2)

Mô hình sử dụng **Transfer Learning** với MobileNetV2 pre-trained trên ImageNet (1.4 triệu ảnh, 1000 class). Kiến trúc gồm 2 phần:

### Base Model: MobileNetV2 (Pre-trained, đóng băng ban đầu)

MobileNetV2 là mạng CNN nhẹ được Google thiết kế cho thiết bị di động, đã được huấn luyện trên ImageNet. Base model đã biết cách trích xuất các đặc trưng cơ bản như cạnh, góc, texture, hình dạng, màu sắc.

### Classification Head (Huấn luyện mới)

```
MobileNetV2 Base (frozen, pre-trained)
  ↓
GlobalAveragePooling2D
  ↓
Dense(128) + ReLU
  ↓
Dropout(0.4)
  ↓
Dense(15) + Softmax
  ↓
Output: Xác suất 15 class
```

### Quy trình huấn luyện 2 pha

| Pha | Mô tả | Epoch |
|-----|-------|-------|
| **Phase 1** | Huấn luyện Classification Head (base đóng băng) | 25 (có EarlyStopping) |
| **Phase 2** | Fine-tune 25% lớp cuối của MobileNetV2 | 7 (có EarlyStopping) |

- **Phase 1**: Chỉ các lớp Dense mới được huấn luyện. Base MobileNetV2 hoạt động như "máy ảnh thông minh" trích xuất đặc trưng.
- **Phase 2**: Mở khóa 25% lớp cuối của MobileNetV2, huấn luyện với learning rate rất thấp (0.0001) để thích nghi đặc trưng với ảnh trái cây.

### Thông số huấn luyện

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Kích thước ảnh đầu vào | 160×160 pixels | Upscale từ 100×100 |
| Batch size | 32 | |
| Phase 1 epochs | 25 | Có EarlyStopping (patience=5) |
| Phase 2 epochs | 7 | Có EarlyStopping (patience=5) |
| Optimizer | Adam | Phase 1: lr=0.001, Phase 2: lr=0.0001 |
| Dropout | 0.4 | Trên classification head |
| Fine-tune layers | 25% lớp cuối | ~40 lớp được mở khóa |
| Số lớp | 15 | Apple, Avocado, Banana, Blueberry, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Strawberry, Watermelon |
| Ảnh/class | Tối đa 400 | Train 70% (~280 ảnh), Validation 30% (~120 ảnh) |
| Thời gian huấn luyện | ~30-40 phút | Trên CPU i5-13500H, 16GB RAM |

---

## Các chỉ số đánh giá

### Accuracy (Độ chính xác)

Tỷ lệ dự đoán đúng trên tổng số mẫu:
```
Accuracy = (TP + TN) / Total
```

### Precision (Độ chính xác dự đoán)

Trong số các mẫu được dự đoán là class X, bao nhiêu % đúng:
```
Precision = TP / (TP + FP)
```

### Recall (Độ bao phủ)

Trong số các mẫu thực sự thuộc class X, bao nhiêu % được phát hiện:
```
Recall = TP / (TP + FN)
```

### F1-score

Trung bình điều hòa của Precision và Recall:
```
F1 = 2 x Precision x Recall / (Precision + Recall)
```

### Confusion Matrix (Ma trận nhầm lẫn)

Bảng thể hiện số lượng dự đoán đúng/sai cho từng class.
- Hàng = Actual (thực tế)
- Cột = Predicted (dự đoán)
- Đường chéo chính = dự đoán đúng

---

## Lưu ý

1. **Mô hình chỉ phân loại được 15 class đã huấn luyện**: Apple, Avocado, Banana, Blueberry, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Strawberry, Watermelon.
2. **Nếu upload ảnh trái cây khác** (ví dụ: sầu riêng, mít, xoài tượng), mô hình vẫn sẽ cố gắng dự đoán thành một trong 15 class trên.
3. **Ảnh nên chụp rõ nét**, trái cây ở giữa khung hình, nền đơn giản để có kết quả tốt nhất.
4. **Nếu không tìm thấy folder**: Kiểm tra và sửa `SOURCE_FOLDERS` trong `config.py` cho khớp với dataset của bạn.
5. **Khi chạy lại prepare_dataset.py**: Script sẽ tự động xóa dữ liệu cũ trong `dataset/selected/`, `train/`, `validation/`, `test/` để tránh trùng lặp.

---

## Hướng phát triển

- **Thử nghiệm các kiến trúc khác**: EfficientNetB0, ResNet50 để so sánh hiệu năng
- **Tăng độ phân giải ảnh**: Sử dụng bản Fruits-360 gốc (1000x1000) để cải thiện độ chính xác
- **Cải thiện giao diện**: Thêm tính năng chụp ảnh từ webcam
- **Triển khai ứng dụng**: Đưa lên Streamlit Cloud, Hugging Face Spaces
- **Thêm chức năng**: Nhận diện độ chín của trái cây, phát hiện trái cây hỏng
- **Grad-CAM Visualization**: Hiển thị vùng ảnh mà mô hình tập trung để đưa ra dự đoán

---

## Kết quả đánh giá mô hình

Mô hình đạt **68.28%** accuracy trên tập test (5034 ảnh).

### Tổng quan

| Chỉ số | Giá trị |
|--------|---------|
| Test Accuracy | 68.28% |
| Macro Precision | 0.7579 |
| Macro Recall | 0.7250 |
| Macro F1-score | 0.7255 |

### Chi tiết từng lớp

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Kiwi | 1.00 | 1.00 | 1.00 | 156 |
| Strawberry | 0.99 | 1.00 | 0.99 | 400 |
| Pineapple | 1.00 | 0.97 | 0.99 | 329 |
| Watermelon | 0.96 | 1.00 | 0.98 | 157 |
| Avocado | 0.96 | 0.99 | 0.97 | 400 |
| Blueberry | 0.98 | 0.97 | 0.97 | 154 |
| Banana | 1.00 | 0.66 | 0.80 | 400 |
| Mango | 0.78 | 0.66 | 0.72 | 308 |
| Pear | 0.54 | 0.74 | 0.63 | 400 |
| Grape | 0.83 | 0.45 | 0.59 | 400 |
| Lemon | 0.41 | 0.83 | 0.55 | 330 |
| Orange | 0.74 | 0.40 | 0.52 | 400 |
| Cherry | 0.46 | 0.43 | 0.44 | 400 |
| Apple | 0.40 | 0.37 | 0.38 | 400 |
| Peach | 0.31 | 0.41 | 0.35 | 400 |

### Nhận xét

- **Các lớp dễ phân biệt** (Kiwi, Strawberry, Pineapple, Watermelon, Avocado, Blueberry): F1-score > 0.95, mô hình hoạt động rất tốt.
- **Các lớp khó** (Apple, Peach, Cherry, Orange, Pear): Đây là các loại quả tròn, màu sắc tương đồng, bị giới hạn bởi độ phân giải 100×100 pixels. Đây là hạn chế của dataset, không phải của mô hình.
- **So với Custom CNN ban đầu (53.30%)**: Transfer Learning MobileNetV2 cải thiện **+15%** accuracy.

---

## Hướng dẫn sử dụng cho nhóm (GitHub)

### Thành viên clone project về máy (3 bước)

```bash
# 1. Clone repository
git clone https://github.com/Thanh-Bo/fruit-augmentation.git
cd fruit_augmentation_classification

# 2. Tải dataset Fruits-360_100x100 từ Kaggle:
#    https://www.kaggle.com/moltean/fruits
#    Giải nén vào thư mục project.
#    Kết quả: fruits-360_100x100/fruits-360/Training/ và Test/

# 3. Cài đặt + chuẩn bị + chạy
pip install -r requirements.txt
python prepare_dataset.py
streamlit run app.py
```

**Không cần huấn luyện lại** — file `model/fruit_cnn_model.h5` (~15MB) đã có sẵn trong repository.

### Các file/thư mục không có trong Git (.gitignore)

| Thư mục/File | Lý do |
|-------------|-------|
| `fruits-360_100x100/` | Dataset 100MB, tải riêng từ Kaggle |
| `dataset/` | Sinh ra từ `prepare_dataset.py`, không cần push |
| `__pycache__/`, `*.pyc` | Cache Python |

### Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|----------|
| `No module named 'tensorflow'` | Chưa cài thư viện | `pip install -r requirements.txt` |
| `Không tìm thấy thư mục Training/` | Dataset chưa giải nén đúng vị trí | Kiểm tra `fruits-360_100x100/fruits-360/Training/` tồn tại |
| `Không tìm thấy folder cho class X` | Tên folder trong dataset khác `SOURCE_FOLDERS` | Mở `config.py`, sửa `SOURCE_FOLDERS` khớp với tên thực tế |
| `Model không load được` | Thiếu file `model/fruit_cnn_model.h5` | Clone lại repo hoặc chạy `python train_model.py` |
| `Streamlit báo use_container_width deprecated` | Phiên bản Streamlit mới | Cảnh báo, không ảnh hưởng chức năng |
| Python 3.12+ không cài được TensorFlow | TensorFlow chưa hỗ trợ Python 3.12 | Dùng Python 3.10-3.11, hoặc `pip install tensorflow==2.16.1` |

---

**Môn học:** Khai phá dữ liệu

**Dataset:** [Fruits-360](https://www.kaggle.com/moltean/fruits)

**Framework:** TensorFlow/Keras + Streamlit

---
