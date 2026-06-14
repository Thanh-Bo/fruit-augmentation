# Xây dựng ứng dụng tăng cường dữ liệu ảnh và phân loại ảnh trái cây

## Mục tiêu đề tài

Xây dựng ứng dụng sử dụng **Transfer Learning với MobileNetV2** kết hợp với **Data Augmentation** để phân loại 15 loại trái cây: Apple, Avocado, Banana, Blueberry, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Strawberry, Watermelon.

## Dataset sử dụng

- **Fruits-360** từ Kaggle: https://www.kaggle.com/moltean/fruits
- **Phiên bản**: `fruits-360_100x100` (ảnh 100x100 pixels)
- **Lý do chọn**: Dataset chất lượng cao, ảnh chụp thực tế trái cây trên nền trắng, có nhiều class, thuận tiện cho việc học tập và nghiên cứu.

### Vì sao chọn bản 100x100?

Bản `fruits-360_100x100` (ảnh 100x100) có dung lượng nhẹ hơn bản gốc (100MB vs ~1GB), phù hợp với máy cá nhân có CPU Intel Core i5, RAM 16GB, không có GPU NVIDIA. Model sẽ resize ảnh về 224x224 trước khi đưa vào MobileNetV2 — đây là kích thước tiêu chuẩn của ImageNet, giúp tận dụng tối đa pre-trained weights.

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
│   ├── train/                    <-- Dữ liệu train (80%)
│   ├── validation/               <-- Dữ liệu validation (20%)
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
- **Copy ảnh cân bằng từ TẤT CẢ folder biến thể**: Với mỗi class, chia đều quota cho các folder nguồn (vd: Apple có 30+ folder → mỗi folder được lấy số ảnh bằng nhau). Folder nào thiếu thì phần còn lại phân bổ cho folder dư. In chi tiết số ảnh lấy từ từng folder.
- Chia selected thành train (80%) và validation (20%)
- Copy ảnh từ Test gốc vào `dataset/test/` (test thật, không trộn với train), cũng cân bằng giữa các folder

**Tại sao cần cân bằng folder biến thể?**

Phiên bản cũ lấy 400 ảnh/class bằng cách duyệt folder theo thứ tự và dừng khi đủ. Điều này gây ra:
1. **Học vẹt (memorization)**: Model chỉ thấy 1-2 biến thể đầu tiên, không thấy các biến thể khác → khi gặp biến thể mới trong test, model thất bại.
2. **Lệch phân phối**: Ví dụ Apple có 30+ folder (Apple Red, Apple Golden, Apple Granny Smith...), nhưng chỉ lấy từ 2-3 folder đầu → model không học được sự đa dạng của táo.
3. **Accuracy thấp ở class đa dạng**: Trước khi cân bằng, Apple (F1=0.38), Peach (F1=0.35) là các class có F1 thấp nhất. Sau cải thiện, tất cả class đều có F1 >= 0.955.

Giải pháp: Chia đều quota 1000 ảnh cho tất cả folder biến thể, shuffle ảnh trong từng folder → model thấy được sự đa dạng thực sự của mỗi class.

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
  - Tối đa 20 epoch, có EarlyStopping (patience=5)
  - ReduceLROnPlateau: giảm learning rate khi val_loss ngừng cải thiện
- **Phase 2 — Fine-tune Base Model**:
  - Mở khóa 35% lớp cuối của MobileNetV2
  - Huấn luyện với learning rate rất thấp (0.00003)
  - Tối đa 10 epoch, có EarlyStopping (patience=5)
- ModelCheckpoint: lưu model tốt nhất vào `model/fruit_cnn_model.h5`
- Lưu biểu đồ accuracy/loss vào `results/accuracy_loss.png` (có đánh dấu điểm bắt đầu fine-tune)

Thời gian ước tính: 45-60 phút trên CPU (i5-13500H, 16GB RAM).

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
| **Phase 1** | Huấn luyện Classification Head (base đóng băng) | 20 (có EarlyStopping) |
| **Phase 2** | Fine-tune 35% lớp cuối của MobileNetV2 | 10 (có EarlyStopping) |

- **Phase 1**: Chỉ các lớp Dense mới được huấn luyện. Base MobileNetV2 hoạt động như "máy ảnh thông minh" trích xuất đặc trưng.
- **Phase 2**: Mở khóa 35% lớp cuối của MobileNetV2, huấn luyện với learning rate rất thấp (0.00003) để thích nghi đặc trưng với ảnh trái cây.

### Thông số huấn luyện (Pilot)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Kích thước ảnh đầu vào | 224×224 pixels | Upscale từ 100×100, chuẩn ImageNet |
| Batch size | 16 | Giảm để tránh OOM trên CPU |
| Phase 1 epochs | 20 | Có EarlyStopping (patience=5) |
| Phase 2 epochs | 10 | Fine-tune, có EarlyStopping (patience=5) |
| Optimizer | Adam | Phase 1: lr=0.001, Phase 2: lr=0.00003 |
| Dropout | 0.4 | Trên classification head |
| Fine-tune layers | 35% lớp cuối | ~54 lớp được mở khóa |
| Số lớp | 15 | Apple, Avocado, Banana, Blueberry, Cherry, Grape, Kiwi, Lemon, Mango, Orange, Peach, Pear, Pineapple, Strawberry, Watermelon |
| Ảnh/class (Pilot) | Tối đa 1000 | Train 80% (~800 ảnh), Validation 20% (~200 ảnh) |
| Thời gian huấn luyện | ~45-60 phút | Trên CPU i5-13500H, 16GB RAM |

### Cấu hình Final (gợi ý sau pilot)

| Tham số | Giá trị Pilot | Giá trị Final |
|---------|---------------|---------------|
| MAX_IMAGES_PER_CLASS | 1000 | 2000 |
| IMG_SIZE | 224 | 224 |
| BATCH_SIZE | 16 | 16 |
| EPOCHS | 20 | 40 |
| FINE_TUNE_EPOCHS | 10 | 25 |
| PHASE1_LR | 0.001 | 0.001 |
| PHASE2_LR | 0.00003 | 0.00001 |
| FINE_TUNE_RATIO | 0.35 | 0.40 |

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
2. **Nếu upload ảnh trái cây khác** (ví dụ: sầu riêng, mít, xoài tượng), mô hình vẫn sẽ cố gắng dự đoán thành một trong 15 class trên. Ứng dụng sẽ hiển thị cảnh báo nếu confidence < 70%.
3. **Ảnh nên chụp rõ nét**, trái cây ở giữa khung hình, nền đơn giản để có kết quả tốt nhất.
4. **Nếu không tìm thấy folder**: Kiểm tra và sửa `SOURCE_FOLDERS` trong `config.py` cho khớp với dataset của bạn.
5. **Khi chạy lại prepare_dataset.py**: Script sẽ tự động xóa dữ liệu cũ trong `dataset/selected/`, `train/`, `validation/`, `test/` để tránh trùng lặp.

## Quy trình an toàn (khuyến nghị)

Để đạt accuracy tốt nhất, làm theo quy trình sau:

```
1. python prepare_dataset.py       # Chuẩn bị dataset cân bằng
   → Kiểm tra summary: đảm bảo mỗi folder biến thể đều có ảnh được chọn
   
2. python train_model.py           # Train pilot (1000 ảnh/class, 20+10 epoch)
   → Model lưu vào model/fruit_cnn_model.h5
   
3. python evaluate_model.py        # Đánh giá
   → Xem class nào có F1 < 0.90 trong classification_report.txt
   
4. Nếu có class yếu → sửa config.py:
   - Tăng MAX_IMAGES_PER_CLASS lên 2000
   - Tăng EPOCHS lên 40, FINE_TUNE_EPOCHS lên 25
   - Chạy lại prepare_dataset.py → train_model.py → evaluate_model.py
   
5. python predict.py               # Test ảnh đơn
6. streamlit run app.py            # Demo
```

## Giới hạn của mô hình

**⚠️ Domain Shift:** Mô hình được huấn luyện **CHỈ trên Fruits-360** — dataset chụp trong studio với nền trắng, ánh sáng chuẩn:
- **Accuracy 98.82% trên Fruits-360 Test không đảm bảo accuracy cao trên ảnh web/Google.**
- Ảnh ngoài đời (nền phức tạp, ánh sáng khác nhau, góc chụp đa dạng) thuộc phân phối khác → model có thể dự đoán sai.
- Softmax luôn trả về 1 trong 15 class — ngay cả khi ảnh không phải trái cây.
- Ứng dụng sẽ cảnh báo nếu confidence < 70%.

**Khắc phục domain shift:** Augmentation thay background, thêm ảnh Google vào train, CutOut/MixUp, Test-Time Augmentation. Xem `PROJECT_DOCUMENTATION.md` để biết chi tiết.

## Kết quả đánh giá mô hình (FINAL)

Mô hình đạt **98.82% accuracy** trên tập test 9,804 ảnh, với Macro F1-score = 0.9907.

### Bảng phân loại chi tiết

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Apple 🍎 | 0.9851 | 0.9270 | 0.9552 | 1,000 |
| Avocado 🥑 | 1.0000 | 0.9940 | 0.9970 | 1,000 |
| Banana 🍌 | 1.0000 | 0.9922 | 0.9961 | 645 |
| Blueberry 🫐 | 0.9747 | 1.0000 | 0.9872 | 154 |
| Cherry 🍒 | 0.9881 | 1.0000 | 0.9940 | 1,000 |
| Grape 🍇 | 1.0000 | 1.0000 | 1.0000 | 1,000 |
| Kiwi 🥝 | 1.0000 | 1.0000 | 1.0000 | 156 |
| Lemon 🍋 | 0.9851 | 1.0000 | 0.9925 | 330 |
| Mango 🥭 | 0.9904 | 1.0000 | 0.9952 | 308 |
| Orange 🍊 | 0.9970 | 1.0000 | 0.9985 | 1,000 |
| Peach 🍑 | 0.9520 | 0.9910 | 0.9711 | 1,000 |
| Pear 🍐 | 0.9760 | 0.9770 | 0.9765 | 1,000 |
| Pineapple 🍍 | 1.0000 | 1.0000 | 1.0000 | 329 |
| Strawberry 🍓 | 1.0000 | 1.0000 | 1.0000 | 725 |
| Watermelon 🍉 | 0.9937 | 1.0000 | 0.9968 | 157 |

- **Tổng Accuracy:** 98.82%
- **Macro Precision:** 0.9895
- **Macro Recall:** 0.9921
- **Macro F1-score:** 0.9907
- **Tất cả 15 class đều có F1-score >= 0.95**

### Phân tích overfitting

Mô hình **KHÔNG bị overfitting** vì:
- Test set đến từ thư mục `Test/` gốc, **hoàn toàn tách biệt** với Training.
- Fruits-360 là dataset "dễ" (nền trắng, ánh sáng chuẩn) — accuracy 95-99% là bình thường.
- Sử dụng Transfer Learning với MobileNetV2 (pre-trained trên 1.4M ảnh ImageNet) → feature rất tổng quát.
- Nhiều lớp regularization: Dropout(0.4), 8 kỹ thuật Augmentation, EarlyStopping, LR thấp ở Phase 2.
- Per-class metrics đồng đều, pattern nhầm lẫn có ý nghĩa (Apple ↔ Peach ↔ Pear — đều là quả tròn).

### Vấn đề Domain Shift (ảnh Google bị sai)

⚠️ **Model hoạt động tốt trên Fruits-360 (98.82%) nhưng kém trên ảnh Google.** Nguyên nhân:
- Fruits-360: nền trắng studio, ánh sáng chuẩn, quả ở giữa khung hình.
- Ảnh Google: nền phức tạp (cây cối, bàn tay, bàn gỗ), ánh sáng đa dạng, nhiều góc chụp.
- Model học "quả táo + nền trắng", không học "quả táo trong mọi bối cảnh".

Cách khắc phục (xem chi tiết trong `PROJECT_DOCUMENTATION.md`):
1. Augmentation thay background ngẫu nhiên trong lúc train.
2. Thêm 10-20% ảnh Google vào tập train.
3. Dùng CutOut, MixUp, CutMix.
4. Test-Time Augmentation (TTA) khi predict.

---

## Hướng dẫn sử dụng cho nhóm (GitHub)

### Thành viên clone project về máy (3 bước)

```bash
# 1. Clone repository
git clone https://github.com/Thanh-Bo/fruit-augmentation.git
cd fruit_augmentation_classification

# 2. Tải dataset Fruits-360_100x100 từ Kaggle:
#     
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
