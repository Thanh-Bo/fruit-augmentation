# Giải thích chi tiết mô hình phân loại ảnh trái cây

> **Mục đích của file này:** Giải thích sâu cách mô hình hoạt động **theo đúng source code**, không lặp lại thông tin đã có trong `README.md`. Đọc file này khi bạn muốn hiểu kỹ thuật để thuyết trình, bảo vệ đồ án, hoặc debug.

---

## Mục lục

1. [Luồng dữ liệu thực tế đi qua project](#1-luồng-dữ-liệu-thực-tế-đi-qua-project)
2. [Pipeline huấn luyện mô hình chi tiết](#2-pipeline-huấn-luyện-mô-hình-chi-tiết)
3. [Kiến trúc mô hình theo đúng code](#3-kiến-trúc-mô-hình-theo-đúng-code)
4. [Transfer Learning và Fine-tuning trong project](#4-transfer-learning-và-fine-tuning-trong-project)
5. [Tiền xử lý ảnh và ảnh hưởng đến kết quả](#5-tiền-xử-lý-ảnh-và-ảnh-hưởng-đến-kết-quả)
6. [Cách model dự đoán một ảnh mới](#6-cách-model-dự-đoán-một-ảnh-mới)
7. [Cách đánh giá mô hình trong code](#7-cách-đánh-giá-mô-hình-trong-code)
8. [Vai trò của Data Augmentation trong project này](#8-vai-trò-của-data-augmentation-trong-project-này)
9. [Điểm mạnh và điểm yếu của mô hình](#9-điểm-mạnh-và-điểm-yếu-của-mô-hình)
10. [Những điểm cần kiểm tra trong source code](#10-những-điểm-cần-kiểm-tra-trong-source-code)
11. [Nếu giảng viên hỏi thì trả lời thế nào](#11-nếu-giảng-viên-hỏi-thì-trả-lời-thế-nào)
12. [Cách trình bày kết quả mô hình trong báo cáo](#12-cách-trình-bày-kết-quả-mô-hình-trong-báo-cáo)
13. [Tóm tắt mô hình trong 10 câu](#13-tóm-tắt-mô-hình-trong-10-câu)

---

## 1. Luồng dữ liệu thực tế đi qua project

Dữ liệu trong project đi qua **5 giai đoạn chính**, mỗi giai đoạn tương ứng với một file Python:

### Giai đoạn 1: Chuẩn bị dữ liệu thô (`prepare_dataset.py`)

**Đầu vào:**
- `fruits-360_100x100/fruits-360/Training/` — chứa ~131 thư mục con, mỗi thư mục là một biến thể của một loại quả (ví dụ: `Apple 5`, `Apple Red 1`, `Banana 1`,...)
- `fruits-360_100x100/fruits-360/Test/` — cấu trúc tương tự Training nhưng là ảnh test độc lập
- `config.py` — định nghĩa `SOURCE_FOLDERS` ánh xạ mỗi class (vd: "Apple") sang danh sách tên thư mục thực tế trong dataset

**Xử lý:**

| Bước | Hàm | Mô tả |
|------|------|-------|
| 1 | `check_raw_dataset()` | Kiểm tra `Training/` và `Test/` có tồn tại không |
| 2 | `clean_directories()` | Xóa toàn bộ `dataset/selected/`, `dataset/train/`, `dataset/validation/`, `dataset/test/` cũ (dùng `shutil.rmtree`) để tránh trùng lặp |
| 3 | `copy_images_to_selected()` | Duyệt từng class → duyệt từng folder nguồn → copy tối đa `MAX_IMAGES_PER_CLASS` (400) ảnh từ **chỉ Training gốc** vào `dataset/selected/<class_name>/` |
| 4 | `split_train_validation()` | Với mỗi class trong `selected/`, đọc danh sách ảnh → `random.shuffle(images)` với `seed=42` → chia 70% vào `train/`, 30% vào `validation/` |
| 5 | `copy_test_from_raw()` | Copy ảnh từ **Test gốc** (KHÔNG từ Training) vào `dataset/test/<class_name>/`, cũng tối đa 400 ảnh/class |

**Đầu ra:**
```
dataset/
├── selected/Apple/       (400 ảnh, từ Training gốc)
├── selected/Avocado/     (400 ảnh)
├── ...
├── train/Apple/          (~280 ảnh = 70% của 400)
├── train/Avocado/        (~280 ảnh)
├── ...
├── validation/Apple/     (~120 ảnh = 30% của 400)
├── validation/Avocado/   (~120 ảnh)
├── ...
├── test/Apple/           (tối đa 400 ảnh, từ Test gốc)
├── test/Avocado/         (tối đa 400 ảnh, từ Test gốc)
└── ...
```

> **Quan trọng:** Test set đến từ thư mục `Test/` gốc của Fruits-360, **hoàn toàn tách biệt** với Training gốc. Train và validation được chia từ cùng một nguồn (Training gốc), nhưng test thì không. Đây là cách làm đúng để tránh **data leakage**.

### Giai đoạn 2: Huấn luyện (`train_model.py`)

**Đầu vào:** `dataset/train/` và `dataset/validation/` (thư mục chứa ảnh đã chia)

**Xử lý:** Chi tiết ở [Mục 2](#2-pipeline-huấn-luyện-mô-hình-chi-tiết)

**Đầu ra:**
- `model/fruit_cnn_model.h5` — model Keras đã huấn luyện (~15MB)
- `model/class_indices.json` — ánh xạ `{0: "Apple", 1: "Avocado", ...}`
- `results/accuracy_loss.png` — biểu đồ accuracy/loss

### Giai đoạn 3: Đánh giá (`evaluate_model.py`)

**Đầu vào:** `model/fruit_cnn_model.h5`, `dataset/test/`

**Xử lý:** Chi tiết ở [Mục 7](#7-cách-đánh-giá-mô-hình-trong-code)

**Đầu ra:**
- `results/confusion_matrix.png`
- `results/classification_report.txt`

### Giai đoạn 4: Dự đoán ảnh đơn (`predict.py`)

**Đầu vào:** `model/fruit_cnn_model.h5`, `model/class_indices.json`, một file ảnh `.jpg`

**Xử lý:** Chi tiết ở [Mục 6](#6-cách-model-dự-đoán-một-ảnh-mới)

**Đầu ra:** In ra console tên class dự đoán, confidence, và xác suất 15 class

### Giai đoạn 5: Ứng dụng web (`app.py`)

**Đầu vào:** Ảnh upload từ người dùng qua giao diện Streamlit

**Xử lý:** Gọi cùng hàm `preprocess_input` + `model.predict()` như `predict.py`, hiển thị kết quả lên web

**Đầu ra:** Giao diện web với 4 tab: Phân loại ảnh, Tăng cường dữ liệu, Kết quả mô hình, Thông tin hệ thống

---

## 2. Pipeline huấn luyện mô hình chi tiết

### 2.1 Cách load dataset

Code dùng `ImageDataGenerator.flow_from_directory()` của Keras — đây là cách load ảnh **từng batch một** (không load toàn bộ vào RAM):

```python
# Trong train_model.py, hàm create_data_generators()
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,                          # Thư mục gốc chứa các subfolder theo class
    target_size=(IMG_SIZE, IMG_SIZE),   # Resize ảnh về 160x160
    batch_size=BATCH_SIZE,              # 32 ảnh/batch
    class_mode='categorical',           # Nhãn dạng one-hot (15 chiều)
    shuffle=True,                       # Xáo trộn sau mỗi epoch
    color_mode='rgb'                    # Đảm bảo 3 kênh màu
)
```

`flow_from_directory` tự động gán class index **theo thứ tự alphabet** của tên thư mục. Ví dụ: `Apple/` → index 0, `Avocado/` → index 1, `Banana/` → index 2,... Kết quả này được lưu vào `class_indices.json` để dùng khi predict.

### 2.2 Cách áp dụng augmentation

Augmentation **chỉ áp dụng cho tập train**, KHÔNG áp dụng cho validation và test. Lý do: validation và test cần phản ánh dữ liệu thực tế (không biến dạng) để đánh giá khách quan.

| Tham số | Giá trị | Ý nghĩa thực tế |
|---------|---------|-----------------|
| `rotation_range=30` | ±30° | Ảnh có thể bị xoay tối đa 30 độ theo chiều kim đồng hồ hoặc ngược lại. Quả táo nghiêng vẫn là quả táo |
| `zoom_range=0.3` | 0.7x – 1.3x | Ảnh có thể bị phóng to hoặc thu nhỏ trong khoảng 70%-130%. Mô phỏng việc chụp gần/xa |
| `width_shift_range=0.15` | ±15% chiều ngang | Ảnh dịch ngang, mô phỏng quả không nằm giữa khung hình |
| `height_shift_range=0.15` | ±15% chiều dọc | Ảnh dịch dọc |
| `horizontal_flip=True` | Lật ngang | Quan trọng với trái cây vì quả đối xứng trái-phải |
| `brightness_range=[0.7, 1.3]` | 70%-130% độ sáng | Mô phỏng điều kiện ánh sáng khác nhau |
| `shear_range=10` | ±10° kéo nghiêng | Biến dạng góc nhìn |
| `fill_mode='nearest'` | Điền pixel lân cận | Khi xoay/dịch để lại khoảng trống, dùng pixel gần nhất để lấp |

### 2.3 Cách build MobileNetV2

Code nằm trong `build_mobilenetv2_model()`. Các bước:

1. **Load base model không có top:** `MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3), alpha=1.0)`
   - `weights='imagenet'` → tải权重 đã huấn luyện trên ImageNet (1.4 triệu ảnh, 1000 class)
   - `include_top=False` → bỏ phần Dense + Softmax cuối cùng (chỉ giữ phần CNN trích xuất đặc trưng)
   - `alpha=1.0` → dùng MobileNetV2 kích thước chuẩn (không thu nhỏ)

2. **Đóng băng toàn bộ base:** `base_model.trainable = False` — không layer nào trong MobileNetV2 được cập nhật ở phase 1

3. **Thêm classification head mới:**
   ```
   Input(160, 160, 3)
     → MobileNetV2 (frozen, training=False)
     → GlobalAveragePooling2D
     → Dense(128, relu)
     → Dropout(0.4)
     → Dense(15, softmax)
   ```

4. **Compile model:** `Adam(lr=0.001)`, loss=`categorical_crossentropy`, metrics=`accuracy`

### 2.4 Loss function — Categorical Crossentropy

Model dùng **categorical crossentropy** vì đây là bài toán **phân loại nhiều lớp (multi-class) với nhãn one-hot**.

Công thức cho một mẫu:

$$L = -\sum_{i=1}^{15} y_i \cdot \log(\hat{y}_i)$$

Trong đó:
- $y_i$ là nhãn thực tế (one-hot: 1 cho class đúng, 0 cho 14 class còn lại)
- $\hat{y}_i$ là xác suất model dự đoán cho class $i$

Ví dụ: Ảnh quả táo → nhãn thật là `[1,0,0,...,0]` (Apple ở index 0). Nếu model dự đoán Apple với xác suất 0.8, loss = $-\log(0.8) = 0.223$. Nếu model dự đoán Apple với xác suất 0.2, loss = $-\log(0.2) = 1.609$ (cao hơn → model bị phạt nặng hơn).

### 2.5 Optimizer — Adam

Adam (Adaptive Moment Estimation) kết hợp 2 ý tưởng:
- **Momentum:** Tích lũy gradient theo thời gian để tránh dao động
- **RMSprop:** Tự động điều chỉnh learning rate cho từng tham số

Trong code:
- **Phase 1:** `Adam(learning_rate=0.001)` — learning rate cao vì head mới được khởi tạo ngẫu nhiên, cần học nhanh
- **Phase 2:** `Adam(learning_rate=0.0001)` — learning rate thấp hơn 10 lần vì base đã được huấn luyện trên ImageNet, chỉ cần điều chỉnh nhẹ

### 2.6 Callback — tác dụng thực tế

| Callback | Phase | Tác dụng thực tế |
|----------|-------|------------------|
| `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` | 1 & 2 | Nếu val_loss không giảm trong 5 epoch liên tiếp → dừng sớm. `restore_best_weights=True` nghĩa là sau khi dừng, model quay về trạng thái tốt nhất (val_loss thấp nhất), không phải trạng thái ở epoch cuối |
| `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)` | Chỉ Phase 1 | Nếu val_loss không giảm trong 3 epoch → giảm learning rate xuống 1 nửa (×0.5). Tối thiểu không xuống dưới 1e-6. Điều này giúp model "tinh chỉnh" khi đã gần điểm tối ưu |
| `ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)` | 1 & 2 | Lưu model mỗi khi val_accuracy đạt kỷ lục mới. File `fruit_cnn_model.h5` luôn là phiên bản có val_accuracy cao nhất |

### 2.7 Hai pha huấn luyện

**Phase 1: Huấn luyện Classification Head (base đóng băng)**

- `base_model.trainable = False` → 154 lớp của MobileNetV2 bị đóng băng
- Chỉ ~130K tham số mới (trong Dense + Softmax) được huấn luyện
- Base model chỉ đóng vai trò "máy trích xuất đặc trưng cố định"
- Mỗi ảnh đi qua MobileNetV2 → nhận được 1 vector đặc trưng 1280 chiều (từ layer cuối của base) → vector này đi qua Dense(128) → Softmax(15)
- Tối đa 25 epoch, nhưng EarlyStopping thường dừng sớm hơn

**Phase 2: Fine-tune một phần Base Model**

- `base_model.trainable = True` → mở khóa toàn bộ base
- Nhưng sau đó **đóng băng lại 75% lớp đầu tiên**:
  ```python
  fine_tune_at = int(len(base_model.layers) * 0.75)  # ~115/154 lớp bị đóng băng
  for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False
  ```
  → Chỉ ~39 lớp cuối của MobileNetV2 được phép cập nhật
- Compile lại với `Adam(lr=0.0001)` (thấp hơn 10 lần)
- Tối đa 7 epoch

---

## 3. Kiến trúc mô hình theo đúng code

### 3.1 Input Layer

```python
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
# IMG_SIZE = 160 (từ config.py)
# 3 = RGB (3 kênh màu)
```

Mỗi ảnh đầu vào có kích thước **160×160 pixels, 3 kênh màu RGB**. Giá trị pixel sau `preprocess_input` nằm trong khoảng **[-1, 1]**.

### 3.2 MobileNetV2 Base

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3),
    alpha=1.0
)
x = base_model(inputs, training=False)  # training=False để dùng BatchNorm ở chế độ inference
```

MobileNetV2 dùng **depthwise separable convolution** và **inverted residual blocks** (bottleneck). Cụ thể:

- **Depthwise separable convolution:** Thay vì dùng 1 filter 3×3×C (C kênh đầu vào) cho mỗi kênh đầu ra, MobileNetV2 tách thành 2 bước: (1) depthwise conv: mỗi kênh có filter riêng 3×3, (2) pointwise conv: dùng 1×1 để trộn thông tin giữa các kênh. Cách này giảm số tham số ~8-9 lần so với convolution thông thường.

- **Inverted residual block:** Khác với ResNet truyền thống (nén → xử lý → mở rộng), MobileNetV2 làm ngược lại: **mở rộng → xử lý → nén**. Điều này giúp giữ được nhiều thông tin hơn trong không gian trung gian.

- **ReLU6:** Dùng ReLU6 (cắt tại giá trị 6) thay vì ReLU thông thường để ổn định hơn khi chạy trên thiết bị di động với độ chính xác thấp (float16/int8).

Kết quả: ảnh 160×160×3 sau khi qua MobileNetV2 (không top) trở thành feature map **5×5×1280**:
- 5×5: kích thước không gian (giảm 32 lần: 160 → 80 → 40 → 20 → 10 → 5)
- 1280: số kênh đặc trưng (tăng dần qua các block)

### 3.3 GlobalAveragePooling2D

```python
x = GlobalAveragePooling2D(name='gap')(x)
# Input: (batch, 5, 5, 1280)
# Output: (batch, 1280)
```

Layer này **tính trung bình** của mỗi kênh trên toàn bộ không gian 5×5:

$$\text{output}[k] = \frac{1}{5 \times 5} \sum_{i=1}^{5} \sum_{j=1}^{5} \text{feature\_map}[i, j, k]$$

Tại sao dùng GAP thay vì Flatten?
- Flatten sẽ tạo ra 5×5×1280 = **32,000** tham số đầu vào cho Dense layer → dễ overfitting
- GAP chỉ tạo ra **1,280** giá trị → ít tham số hơn, chống overfitting tốt hơn

Mỗi giá trị trong vector 1280 chiều này **đại diện cho mức độ "kích hoạt" trung bình** của một đặc trưng trên toàn bộ ảnh. Ví dụ: kênh số 50 có thể mã hóa "màu đỏ", kênh số 200 mã hóa "texture nhẵn".

### 3.4 Dense(128) + ReLU

```python
x = Dense(128, activation='relu', name='dense_head')(x)
# Input: (batch, 1280)
# Output: (batch, 128)
# Số tham số: 1280 × 128 + 128(bias) = 163,968
```

Layer này học cách **kết hợp** 1280 đặc trưng từ MobileNetV2 thành 128 đặc trưng mới, phù hợp riêng cho bài toán phân loại trái cây. Ví dụ: nó có thể học rằng "màu đỏ" + "hình tròn" + "có cuống" = Apple.

ReLU (Rectified Linear Unit): $f(x) = \max(0, x)$. Nếu đầu vào âm → output = 0; nếu dương → giữ nguyên. Điều này tạo ra **tính phi tuyến** (non-linearity), cho phép mạng học được các mối quan hệ phức tạp.

### 3.5 Dropout(0.4)

```python
x = Dropout(0.4, name='dropout_head')(x)
```

**Khi train (training=True):** Mỗi neuron trong 128 neuron của Dense có **40% xác suất bị tắt** (set về 0) trong mỗi batch. Điều này buộc mạng không được phụ thuộc vào bất kỳ neuron cụ thể nào, mà phải phân tán thông tin ra nhiều neuron.

**Khi predict (training=False):** Tất cả neuron đều hoạt động, nhưng output được nhân với (1 - 0.4) = 0.6 để bù trừ.

Tại sao cần Dropout? Vì Dense(128) có ~164K tham số — tương đối nhiều so với lượng dữ liệu (~4200 ảnh train). Nếu không có Dropout, model có thể "ghi nhớ" từng ảnh train thay vì học đặc trưng tổng quát.

### 3.6 Dense(15) + Softmax

```python
outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
# NUM_CLASSES = 15
# Input: (batch, 128)
# Output: (batch, 15)
# Số tham số: 128 × 15 + 15(bias) = 1,935
```

Softmax chuyển 15 giá trị thô (logits) thành 15 xác suất **có tổng bằng 1**:

$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{15} e^{z_j}}$$

Trong đó $z_i$ là logit cho class $i$, $\hat{y}_i$ là xác suất dự đoán.

Ví dụ: Nếu logit của Apple là 5.2, Avocado là 1.3, các class khác ~0:
- $e^{5.2} \approx 181$, $e^{1.3} \approx 3.67$, các $e^0 = 1$
- $\hat{y}_{Apple} = 181 / (181 + 3.67 + 13 \times 1) \approx 0.92$ (92%)
- $\hat{y}_{Avocado} \approx 0.02$ (2%)

**Đặc điểm quan trọng của Softmax:** Luôn có một class được dự đoán, ngay cả khi ảnh không thuộc 15 loại nào. Nếu ảnh con mèo được đưa vào, 15 xác suất sẽ phân bố "gần đều" (mỗi class ~6.7%), và class cao nhất có thể là bất kỳ class nào — nhưng model vẫn trả về class đó. Đây là hạn chế cố hữu của Softmax.

---

## 4. Transfer Learning và Fine-tuning trong project

### 4.1 Tại sao không train toàn bộ model ngay từ đầu?

Nếu train MobileNetV2 từ đầu (random weights) với chỉ ~4,200 ảnh trái cây:
- MobileNetV2 có ~2.2 triệu tham số → cần hàng trăm nghìn ảnh để train từ đầu
- 4200 ảnh là quá ít → model sẽ overfitting rất nặng (accuracy train cao, accuracy test thấp)
- Thời gian train trên CPU sẽ rất lâu (hàng giờ hoặc hơn)

**Transfer Learning** giải quyết vấn đề này: MobileNetV2 đã được huấn luyện trên ImageNet (1.4 triệu ảnh, 1000 class). Các lớp đầu của nó đã biết cách phát hiện cạnh, góc, texture, hình dạng cơ bản — những đặc trưng này **dùng chung được** cho hầu hết các bài toán thị giác máy tính, bao gồm cả phân loại trái cây.

### 4.2 Khác biệt giữa hai giai đoạn

| Tiêu chí | Phase 1: Train Classification Head | Phase 2: Fine-tune Base |
|----------|------------------------------------|------------------------|
| **Phần được train** | Chỉ Dense(128) + Dropout + Softmax(15) | 25% lớp cuối của MobileNetV2 + toàn bộ head |
| **Số tham số trainable** | ~165,903 | ~1,200,000 |
| **Learning rate** | 0.001 (cao) | 0.0001 (thấp hơn 10 lần) |
| **Callbacks** | EarlyStopping + ReduceLROnPlateau + Checkpoint | EarlyStopping + Checkpoint (KHÔNG có ReduceLROnPlateau) |
| **Base model** | Đóng băng hoàn toàn (`trainable=False`) | Mở 25% lớp cuối |
| **Mục đích** | Học cách kết hợp đặc trưng có sẵn từ ImageNet để phân loại trái cây | Tinh chỉnh đặc trưng MobileNetV2 cho phù hợp với ảnh trái cây |

### 4.3 Tại sao fine-tune dùng learning rate thấp?

Khi bạn mở khóa một phần base model:
- Các layer này đã được huấn luyện kỹ trên ImageNet → weights đang ở vị trí "tốt"
- Nếu dùng learning rate cao (0.001), gradient lớn có thể **phá hủy** những gì base đã học → "catastrophic forgetting"
- Learning rate thấp (0.0001) giúp điều chỉnh weights **từ từ**, chỉ thích nghi với đặc thù của ảnh trái cây mà không quên kiến thức từ ImageNet

### 4.4 Rủi ro nếu mở khóa quá nhiều layer

Nếu mở khóa toàn bộ 154 lớp của MobileNetV2 (thay vì chỉ 25% cuối):

1. **Overfitting nghiêm trọng:** ~2.2 triệu tham số cùng được train trên ~4200 ảnh → model "học thuộc lòng" ảnh train, không tổng quát hóa được
2. **Catastrophic forgetting:** Các lớp đầu (phát hiện cạnh, góc cơ bản) bị thay đổi → mất đi kiến thức quý giá từ ImageNet
3. **Thời gian train tăng gấp 3-4 lần trên CPU**

Nguyên tắc chung: **Các lớp càng gần input thì càng "tổng quát"** (cạnh, góc, màu), **các lớp càng gần output thì càng "đặc thù"** (hình dạng quả táo cụ thể). Vì vậy chỉ nên fine-tune các lớp gần output — đó là lý do code chỉ mở 25% lớp cuối.

---

## 5. Tiền xử lý ảnh và ảnh hưởng đến kết quả

### 5.1 Resize ảnh

Ảnh gốc Fruits-360 có kích thước **100×100 pixels**. Tuy nhiên, trong code:

```python
# config.py
IMG_SIZE = 160

# train_model.py
target_size=(IMG_SIZE, IMG_SIZE)  # = (160, 160)
```

**Tại sao resize lên 160×160?**

MobileNetV2 yêu cầu input tối thiểu **32×32** nhưng kích thước input trong code thường là 96, 128, 160, 192, hoặc 224. Các lý do cụ thể cho việc chọn 160:

1. **Downsampling factor:** MobileNetV2 giảm kích thước không gian 32 lần (160→80→40→20→10→5). Với ảnh 100×100, sau 32 lần giảm sẽ còn 3.125 → làm tròn xuống 3 → quá nhỏ, mất nhiều thông tin. Với 160×160, output feature map là 5×5 → còn đủ thông tin không gian.

2. **Padding:** 160 chia hết cho 32 (160 = 5 × 32), đảm bảo không có padding không đều ở các layer.

3. **Cân bằng:** 224×224 (kích thước gốc ImageNet) sẽ nặng hơn về tính toán, trong khi 160×160 đủ để phân biệt các loại quả mà vẫn nhẹ.

> **Lưu ý:** `app.py` sidebar hiển thị "Kích thước đầu vào: 128x128 pixel" — đây là thông tin **sai** so với code thực tế. Code dùng `IMG_SIZE=160` từ `config.py`. Đây là một bug hiển thị trong giao diện Streamlit.

### 5.2 Hàm preprocess_input của MobileNetV2

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

Hàm này thực hiện phép biến đổi:

```
pixel_mới = (pixel_cũ / 127.5) - 1.0
```

Kết quả:
- Pixel gốc 0 → -1.0
- Pixel gốc 127.5 → 0.0
- Pixel gốc 255 → 1.0

→ Dữ liệu nằm trong khoảng **[-1, 1]** thay vì [0, 255] hoặc [0, 1].

**Tại sao cần chuẩn hóa về [-1, 1]?**

1. **MobileNetV2 được train với chuẩn này trên ImageNet** — nếu dùng chuẩn khác, weights pre-trained sẽ không hoạt động đúng (vì chúng "kỳ vọng" input trong khoảng [-1, 1])
2. **Ổn định số học:** Giá trị lớn (0-255) gây ra gradient lớn → huấn luyện không ổn định. Giá trị nhỏ quanh 0 giúp gradient ổn định hơn
3. **Hội tụ nhanh hơn:** Khi input có mean ≈ 0 và variance ≈ 1, optimizer (Adam) hoạt động hiệu quả hơn

> **Quan trọng:** Cùng một hàm `preprocess_input` được dùng ở cả `train_model.py`, `evaluate_model.py`, `predict.py` và `app.py`. Điều này đảm bảo **preprocessing nhất quán** giữa lúc train và lúc predict.

### 5.3 Pipeline tiền xử lý đầy đủ

```
Ảnh gốc (100×100, RGB, 0-255)
  → Resize về 160×160 (dùng LANCZOS interpolation)
  → Chuyển sang float32
  → preprocess_input: scale về [-1, 1]
  → Đưa vào model
```

---

## 6. Cách model dự đoán một ảnh mới

### 6.1 Trong `predict.py` (dòng lệnh)

**Bước 1: Đọc ảnh**
```python
img = Image.open(image_path)         # Mở file bằng PIL
img = img.convert('RGB')             # Đảm bảo 3 kênh (phòng ảnh RGBA/xám)
```

**Bước 2: Resize**
```python
img = img.resize((160, 160), Image.Resampling.LANCZOS)
```
LANCZOS là thuật toán nội suy chất lượng cao — giữ được chi tiết tốt hơn BILINEAR khi phóng to ảnh.

**Bước 3: Chuẩn hóa**
```python
img_array = np.array(img, dtype=np.float32)       # Shape: (160, 160, 3)
img_array = np.expand_dims(img_array, axis=0)      # Shape: (1, 160, 160, 3)
img_array = preprocess_input(img_array)            # Scale về [-1, 1]
```

**Bước 4: Dự đoán**
```python
predictions = model.predict(img_array, verbose=0)[0]  # [0] lấy batch đầu tiên
# predictions là numpy array 15 phần tử, mỗi phần tử là xác suất 0-1, tổng = 1
```

**Bước 5: Lấy class cao nhất**
```python
predicted_index = np.argmax(predictions)   # Vị trí có xác suất cao nhất
confidence = predictions[predicted_index]  # Giá trị xác suất tại vị trí đó
predicted_class = index_to_class[predicted_index]  # Ánh xạ index → tên class
```

### 6.2 Trong `app.py` (Streamlit)

Quy trình tương tự `predict.py`, cùng dùng `preprocess_input` của MobileNetV2. Điểm khác:
- Ảnh đến từ `st.file_uploader` (người dùng upload) hoặc từ ảnh mẫu trong `test/`
- Có `@st.cache_resource` để cache model (không load lại mỗi lần người dùng tương tác)
- Hiển thị kết quả bằng HTML/CSS thay vì in ra console
- Có thêm biểu đồ top-5 predictions và thanh tiến trình xác suất cho từng class

### 6.3 Confidence nghĩa là gì?

**Confidence** là xác suất Softmax cho class được chọn. Ví dụ:
- Nếu confidence = 0.95 → model "tự tin" 95% rằng ảnh này là class X
- Nếu confidence = 0.35 → model khá "bối rối", chỉ chắc 35%

**Lưu ý quan trọng:** Confidence cao KHÔNG đồng nghĩa với dự đoán đúng. Model có thể "tự tin sai" (high confidence, wrong prediction) — đặc biệt với ảnh ngoài 15 class đã train.

Code phân loại confidence trong `app.py`:
```python
if confidence >= 0.80:      → "Mức độ tin cậy cao" (xanh lá)
elif confidence >= 0.50:    → "Mức độ tin cậy trung bình" (vàng)
else:                       → "Mức độ tin cậy thấp" (đỏ)
```

### 6.4 Vì sao model vẫn dự đoán một trong 15 class ngay cả khi ảnh không phải trái cây?

Vì Softmax **luôn tạo ra phân phối xác suất có tổng = 1** trên 15 class. Không có class thứ 16 là "không phải trái cây" hay "không biết".

Nếu bạn đưa ảnh con mèo:
- MobileNetV2 vẫn trích xuất đặc trưng (cạnh, góc, texture,...)
- Dense layers vẫn cố gắng khớp vào 15 class đã học
- Softmax vẫn chọn ra 1 class có xác suất cao nhất (dù có thể chỉ ~15-20%)
- Kết quả: Model dự đoán "đây là Cherry" với confidence 18%

Đây là hạn chế thiết kế. Để khắc phục, cần thêm **ngưỡng confidence** (chỉ chấp nhận nếu confidence > 70%) hoặc thêm cơ chế **out-of-distribution detection**.

---

## 7. Cách đánh giá mô hình trong code

### 7.1 Load tập test

```python
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,          # KHÔNG xáo trộn — quan trọng để khớp nhãn
)

# Lấy toàn bộ dữ liệu ra khỏi generator
X_test, y_true = [], []
for i in range(len(test_generator)):
    X_batch, y_batch = next(test_generator)
    X_test.append(X_batch)
    y_true.append(y_batch)

X_test = np.concatenate(X_test, axis=0)        # (5034, 160, 160, 3)
y_true_labels = np.argmax(y_true, axis=1)       # Chuyển one-hot → index (5034,)
```

**Lưu ý:** `shuffle=False` rất quan trọng vì nhãn từ `flow_from_directory` có thứ tự cố định theo alphabet thư mục. Nếu shuffle, `y_true` sẽ không khớp với thứ tự ảnh.

### 7.2 Dự đoán toàn bộ test set

```python
y_pred_probs = model.predict(X_test, verbose=1)  # (5034, 15) — xác suất 15 class
y_pred_labels = np.argmax(y_pred_probs, axis=1)   # (5034,) — index class dự đoán
```

### 7.3 Confusion matrix được tạo thế nào

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
# cm là ma trận 15×15
# cm[i][j] = số ảnh thực tế class i nhưng bị dự đoán là class j
```

**Cách đọc confusion matrix:**
- **Hàng i** = ảnh thực tế thuộc class i
- **Cột j** = model dự đoán là class j
- **Đường chéo chính** cm[i][i] = dự đoán đúng cho class i
- **Ô ngoài đường chéo** = nhầm lẫn. Ví dụ: cm[Apple][Pear] = 150 nghĩa là 150 ảnh Apple bị dự đoán nhầm thành Pear

Từ confusion matrix của project (kết quả thực tế):

| Cặp dễ nhầm | Số lượng nhầm | Lý do |
|-------------|---------------|-------|
| Apple → Pear | ~120-150 ảnh | Cả hai đều tròn, màu xanh/vàng/đỏ, kích thước tương tự |
| Peach → Apple/Pear | ~120 ảnh | Đào tròn, màu hồng-vàng, dễ nhầm với táo đỏ |
| Orange → Grapefruit/Lemon | ~100 ảnh | Màu cam-vàng, hình tròn |
| Cherry → Apple | ~80 ảnh | Cherry đỏ, tròn, nhỏ → ở 100×100 khó phân biệt kích thước |

### 7.4 Classification report được tính thế nào

```python
from sklearn.metrics import classification_report

# Precision = TP / (TP + FP)
# "Trong số ảnh model nói là Apple, bao nhiêu % đúng?"

# Recall = TP / (TP + FN)
# "Trong số ảnh thực sự là Apple, bao nhiêu % được model phát hiện?"

# F1 = 2 × P × R / (P + R)
# Trung bình điều hòa — cao khi cả P và R đều cao
```

Ví dụ từ kết quả thực tế cho class Apple:
- Precision = 0.40 → Khi model nói "đây là Apple", chỉ 40% là đúng (60% là nhầm từ class khác)
- Recall = 0.37 → Trong 400 ảnh Apple thực sự, model chỉ tìm ra 37% (63% bị bỏ sót)
- F1 = 0.38 → Rất thấp, Apple là class khó phân biệt nhất

Ngược lại, class Kiwi:
- Precision = 1.00, Recall = 1.00, F1 = 1.00 → Hoàn hảo
- Lý do: Kiwi có màu nâu đặc trưng + texture lông — không class nào khác có đặc điểm này

### 7.5 Validation class order

Code có bước kiểm tra thứ tự class giữa `test_generator.class_indices` và `class_indices.json`:

```python
if generator_labels != saved_labels:
    print("[CẢNH BÁO] Thứ tự class không khớp!")
    # Sẽ dùng class_indices.json để đảm bảo đúng nhãn
```

Đây là một safeguard tốt — nếu thứ tự folder thay đổi (ví dụ: thêm/xóa folder trong `test/`), nhãn vẫn được ánh xạ đúng.

---

## 8. Vai trò của Data Augmentation trong project này

### 8.1 Từng phép biến đổi giúp chống overfitting như thế nào

| Phép biến đổi | Cách chống overfitting | Ví dụ cụ thể với ảnh trái cây |
|---------------|------------------------|-------------------------------|
| **Rotation (±30°)** | Model không học vị trí "thẳng đứng" là đặc trưng của quả. Học được rằng quả táo xoay 20° vẫn là quả táo | Trong Fruits-360, tất cả ảnh đều được chụp quả thẳng đứng. Nếu không có rotation, model có thể học "thẳng đứng = táo" → thất bại khi gặp ảnh chụp nghiêng |
| **Width/Height Shift (±15%)** | Model không phụ thuộc vào vị trí quả trong khung hình | Dataset gốc có quả luôn ở giữa. Shift giúp model học rằng quả ở góc trái vẫn là quả đó |
| **Zoom (0.7-1.3x)** | Model học đặc trưng ở nhiều tỷ lệ khác nhau | Quả cherry zoom to không bị nhầm thành táo; quả táo zoom nhỏ không bị nhầm thành cherry |
| **Horizontal Flip** | Hầu hết trái cây đối xứng trái-phải → flip không làm thay đổi lớp | Flip giúp tăng gấp đôi dữ liệu hiệu quả mà không gây hại |
| **Brightness (0.7-1.3x)** | Model không phụ thuộc vào điều kiện ánh sáng cố định | Ảnh chụp trong nhà (tối) và ngoài trời (sáng) đều được nhận diện đúng |
| **Shear (10°)** | Mô phỏng thay đổi góc nhìn | Giống như chụp quả từ góc nghiêng thay vì thẳng từ trên xuống |

### 8.2 Augmentation nào có thể gây hại nếu dùng quá mạnh?

| Phép biến đổi | Ngưỡng an toàn | Nếu quá mạnh | Hậu quả với ảnh trái cây |
|---------------|---------------|--------------|--------------------------|
| **Rotation** | < 45° | > 90° | Quả bị lộn ngược → mất thông tin hình dạng. Quả táo xoay 180° vẫn giống táo nhưng cuống ở dưới → model có thể nhầm |
| **Zoom** | 0.7-1.3x | < 0.5x hoặc > 1.5x | Zoom quá nhỏ → quả chỉ còn vài pixel, mất hết đặc trưng. Zoom quá to → chỉ thấy vỏ, không thấy hình dạng tổng thể |
| **Brightness** | 0.5-1.5 | < 0.3 hoặc > 2.0 | Quá tối → mất màu sắc (đặc trưng quan trọng của trái cây). Quá sáng → ảnh trắng xóa |
| **Horizontal Flip** | An toàn | — | Với trái cây đối xứng, flip hầu như không gây hại |
| **Vertical Flip** | KHÔNG dùng | — | Trái cây không đối xứng trên-dưới (cuống ở trên). Vertical flip sẽ tạo ảnh "quả lộn ngược" → gây hại. Code đã đúng khi không bật `vertical_flip` |

### 8.3 Một điểm khác biệt nhỏ trong app.py

Trong `app.py` Tab 2 (Tăng cường dữ liệu), augmentation demo dùng **Pillow** (không dùng TensorFlow):
```python
variants["Lật ngang"] = ImageOps.mirror(pil_image)
variants["Xoay ảnh (30°)"] = pil_image.rotate(30, expand=True, fillcolor=(255,255,255))
```

Đây chỉ là code demo trực quan cho người dùng xem, **không phải** augmentation thực tế dùng khi train. Augmentation thực tế khi train dùng `ImageDataGenerator` của TensorFlow (trong `train_model.py`).

---

## 9. Điểm mạnh và điểm yếu của mô hình

### 9.1 Điểm mạnh của MobileNetV2 với project này

1. **Nhẹ:** ~2.2M tham số (so với ResNet50: ~23M, VGG16: ~138M). Train được trên CPU i5 trong ~30-40 phút
2. **Transfer learning hiệu quả:** Pre-trained trên ImageNet → base đã biết trích xuất đặc trưng cơ bản → chỉ cần ít dữ liệu để fine-tune
3. **Thiết kế chống overfitting:** Dùng depthwise separable conv (ít tham số) + GAP (thay vì Flatten) + Dropout → ít bị overfitting hơn các kiến trúc nặng
4. **Tốt với dataset nhỏ:** Với ~4200 ảnh train, MobileNetV2 Transfer Learning vẫn cho kết quả khả quan (68.28% test accuracy)

### 9.2 Điểm yếu do ảnh 100×100

1. **Mất chi tiết tinh:** Ảnh 100×100 không đủ độ phân giải để phân biệt texture chi tiết (vd: vân trên vỏ cam vs vỏ chanh, lông trên vỏ đào vs vỏ táo)
2. **Kích thước tương đối bị mất:** Cherry (nhỏ) và Apple (to) đều được resize về 160×160 → model không biết kích thước thật của quả
3. **Resize 100→160:** Phải dùng nội suy để phóng to → ảnh bị "mờ" nhẹ, không sắc nét như ảnh gốc 160×160

### 9.3 Điểm yếu do nền trắng của Fruits-360

1. **Model học nền trắng như một đặc trưng:** Khi đưa ảnh ngoài đời thực (nền phức tạp, nhiều vật thể), model thất bại vì "không thấy nền trắng"
2. **Không có background augmentation:** Code không thêm các phép biến đổi nền (như thay nền ngẫu nhiên) → model không học được cách bỏ qua nền
3. **Thiếu context thực tế:** Ảnh ngoài đời có thể có nhiều quả, tay người cầm, lá cây,... — model chưa từng thấy những thứ này

### 9.4 Rủi ro khi đem model đi nhận diện ảnh ngoài đời thật

| Rủi ro | Mức độ | Mô tả |
|--------|--------|-------|
| **Domain shift** | Cao | Ảnh Fruits-360 (nền trắng, studio) khác xa ảnh đời thật (nền phức tạp). Model sẽ hoạt động kém hơn nhiều |
| **Không có cơ chế từ chối** | Cao | Model luôn dự đoán 1 trong 15 class, ngay cả khi ảnh không phải trái cây |
| **Thiếu đa dạng góc chụp** | Trung bình | Fruits-360 chủ yếu chụp thẳng góc. Ảnh chụp từ trên xuống, góc nghiêng cực đoan có thể gây lỗi |
| **Thiếu đa dạng trạng thái** | Cao | Không có ảnh quả bị cắt, gọt vỏ, dập nát, hoặc chưa chín |

### 9.5 Những class dễ/khó và lý do kỹ thuật

**Dễ dự đoán (F1 > 0.95):**

| Class | F1 | Lý do |
|-------|----|-------|
| Kiwi | 1.00 | Màu nâu, texture lông — đặc trưng độc nhất, không class nào khác có |
| Strawberry | 0.99 | Màu đỏ + hạt li ti + hình dạng thuôn — rất đặc trưng |
| Pineapple | 0.99 | Hình dạng oval + mắt dứa + lá trên đầu — không nhầm với ai |
| Watermelon | 0.98 | Sọc xanh đen đặc trưng + hình tròn lớn |
| Avocado | 0.97 | Màu xanh đậm + hình quả lê thuôn dài |
| Blueberry | 0.97 | Màu xanh tím đậm, nhỏ — màu sắc độc đáo |

**Khó dự đoán (F1 < 0.60):**

| Class | F1 | Lý do kỹ thuật |
|-------|----|----------------|
| Apple | 0.38 | Hình tròn + đỏ/vàng/xanh → trùng đặc điểm với Peach (tròn, hồng-đỏ), Pear (hình gần tròn, vàng-xanh). Ở 100×100, sự khác biệt tinh tế giữa các loại táo-lê-đào bị mờ |
| Peach | 0.35 | Như trên. Đào và táo đỏ rất giống nhau ở độ phân giải thấp |
| Cherry | 0.44 | Tròn + đỏ → giống táo đỏ thu nhỏ. Khi resize về 160×160, sự khác biệt kích thước bị xóa bỏ |
| Orange | 0.52 | Tròn + cam → giống Lemon (khi Lemon chưa chín vàng), giống Peach (màu cam-hồng) |
| Lemon | 0.55 | Tròn/cầu + vàng → giống táo vàng (Apple Golden), cam vàng |
| Grape | 0.59 | Chùm nho có hình dạng không đều, dễ nhầm với quả mọng khác khi chụp gần |

### 9.6 So sánh code và README

| Khía cạnh | README nói | Code thực tế | Khớp? |
|-----------|-----------|-------------|-------|
| Fine-tune layers | 25% lớp cuối | `int(len(base_model.layers) * 0.75)` = 25% cuối | ✓* |
| Phase 2 LR | 0.0001 | `Adam(learning_rate=0.0001)` | ✓ |
| IMG_SIZE | 160×160 | `IMG_SIZE = 160` | ✓ |
| BATCH_SIZE | 32 | `BATCH_SIZE = 32` | ✓ |
| Dropout | 0.4 | `Dropout(0.4)` | ✓ |
| EPOCHS Phase 1 | 25 | `EPOCHS = 25` | ✓ |
| FINE_TUNE_EPOCHS | 7 | `FINE_TUNE_EPOCHS = 7` | ✓ |

> \* Code docstring nói "30% lớp cuối" nhưng comment trong code nói "25% lớp cuối (cân bằng)" và công thức toán học = 25%. README nói 25%. Thực tế là 25%.

> **Khác biệt phát hiện:** `app.py` sidebar hiển thị "Kích thước đầu vào: 128x128 pixel" và "Số epoch huấn luyện: 15 (có EarlyStopping)" và "Adam (learning_rate=0.0005)" — đây là thông tin hiển thị cũ, không khớp với `config.py`. Model thực tế train với 160×160, tối đa 32 epoch (25+7), và learning rate 0.001/0.0001. Các thông tin này KHÔNG ảnh hưởng đến chức năng (vì code dùng giá trị từ `config.py`), nhưng gây nhầm lẫn cho người đọc giao diện.

---

## 10. Những điểm cần kiểm tra trong source code

### 10.1 Data leakage — KHÔNG có ✓

- Test set lấy từ `fruits-360/Test/` (thư mục riêng của Fruits-360)
- Train và validation lấy từ `fruits-360/Training/`, chia 70/30
- Không có ảnh nào từ Test gốc bị trộn vào train/validation
- Kết luận: **Không có data leakage**

### 10.2 Train/validation/test có tách đúng không? — CÓ ✓

- Train: 70% từ Training gốc (mỗi class ~280 ảnh)
- Validation: 30% từ Training gốc (mỗi class ~120 ảnh)
- Test: Từ Test gốc, tách biệt hoàn toàn
- `seed(42)` đảm bảo mỗi lần chạy `prepare_dataset.py` cho cùng một split

### 10.3 Class index có khớp khi train và predict không? — CÓ ✓

- Khi train: `flow_from_directory` gán index theo alphabet thư mục → lưu vào `class_indices.json`
- Khi predict: Load `class_indices.json` → ánh xạ ngược index → tên class
- `evaluate_model.py` có bước kiểm tra chéo giữa `generator.class_indices` và `class_indices.json`
- Kết luận: **Khớp, có safeguard kiểm tra**

### 10.4 Preprocessing lúc train và predict có giống nhau không? — CÓ ✓

- Train: Dùng `preprocess_input` của `mobilenet_v2` trong `ImageDataGenerator`
- Predict (`predict.py`): Dùng `preprocess_input` của `mobilenet_v2`
- App (`app.py`): Dùng `preprocess_input` của `mobilenet_v2`
- Kết luận: **Nhất quán**

### 10.5 Model lưu/load có ổn không? — CÓ ✓

- Lưu: `model.save(MODEL_PATH)` (định dạng H5, lưu toàn bộ weights + kiến trúc + optimizer state) — nhưng thực tế code dùng `ModelCheckpoint` để lưu, cũng là định dạng H5
- Load: `tf.keras.models.load_model(MODEL_PATH)` — load toàn bộ
- File `fruit_cnn_model.h5` đã tồn tại trong repo (~15MB)

### 10.6 App Streamlit có xử lý lỗi upload ảnh không? — Một phần

- Có kiểm tra model và class_indices tồn tại → hiển thị cảnh báo nếu thiếu
- Có try-except khi mở ảnh (`except Exception as e: st.error(...)`)
- Tuy nhiên, **không có** kiểm tra ảnh có đúng định dạng ảnh không (có thể upload file .txt đổi đuôi thành .jpg)
- **Không có** giới hạn kích thước file upload → upload ảnh 50MB có thể gây lỗi

### 10.7 Hard-code đường dẫn gây lỗi không? — Không, dùng `config.py` ✓

Tất cả đường dẫn được định nghĩa trong `config.py` dùng `os.path.join(BASE_DIR, ...)`, với `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` → chạy được trên mọi máy, mọi hệ điều hành.

### 10.8 Các vấn đề nhỏ khác

| Vấn đề | File | Mô tả |
|--------|------|-------|
| `app.py` sidebar sai thông số | `app.py` | Hiển thị IMG_SIZE=128 (sai, thực tế 160), epochs=15 (sai, thực tế tối đa 32), lr=0.0005 (sai, thực tế 0.001/0.0001) |
| `evaluate_model.py` load toàn bộ test vào RAM | `evaluate_model.py` | `np.concatenate` tất cả batch → ~1.5GB RAM cho 5034 ảnh. Với test set lớn hơn có thể gây tràn RAM |
| Docstring không nhất quán | `train_model.py` | Docstring nói "fine-tune 30% lớp cuối" nhưng comment trong code nói "25% lớp cuối" và công thức tính = 25% |
| `augment_preview.py` không dùng `preprocess_input` | `augment_preview.py` | Augmentation preview hiển thị pixel [0,255] — đúng vì chỉ để xem, không ảnh hưởng model |
| Tab 2 `app.py` dùng Pillow, không dùng TF | `app.py` | Augmentation trong app demo bằng Pillow (khác với augmentation TF khi train). Không ảnh hưởng model, chỉ ảnh hưởng trải nghiệm người dùng |

---

## 11. Nếu giảng viên hỏi thì trả lời thế nào

### Q1: Vì sao chọn MobileNetV2 thay vì CNN tự xây?

**Trả lời:** CNN tự xây (từng làm ở giai đoạn trước) cho accuracy ~53%. MobileNetV2 với Transfer Learning đạt ~68% — cải thiện **+15%**. Lý do:

- MobileNetV2 đã được huấn luyện trên ImageNet (1.4 triệu ảnh) → đã biết cách trích xuất đặc trưng cơ bản. CNN tự xây phải học mọi thứ từ đầu với chỉ ~4200 ảnh.
- MobileNetV2 dùng depthwise separable convolution → ít tham số hơn → ít overfitting hơn trên dataset nhỏ.
- MobileNetV2 được thiết kế tối ưu cho thiết bị có tài nguyên hạn chế (như CPU laptop), phù hợp với điều kiện thực tế của đồ án.

### Q2: Vì sao cần Data Augmentation?

**Trả lời:** Fruits-360 là dataset chụp trong studio: nền trắng, ánh sáng chuẩn, quả ở giữa khung hình. Nếu không augmentation, model sẽ học những đặc điểm "giả" này thay vì đặc trưng thực sự của quả. Augmentation (xoay, dịch, zoom, đổi sáng) giúp:

- Tăng dữ liệu ảo mà không cần thu thập thêm → giảm overfitting
- Model học được đặc trưng bất biến: quả táo xoay 20°, tối hơn, hoặc lệch trái vẫn là quả táo
- Cải thiện khả năng tổng quát hóa (generalization)

### Q3: Vì sao chia train/validation/test?

**Trả lời:**

- **Train:** Dùng để cập nhật weights của model
- **Validation:** Dùng trong quá trình train để theo dõi overfitting (qua val_loss), điều chỉnh learning rate (ReduceLROnPlateau), và dừng sớm (EarlyStopping). Validation **không** dùng để cập nhật weights, nhưng **có ảnh hưởng gián tiếp** đến quá trình train qua các callback
- **Test:** Dùng để đánh giá **cuối cùng**, sau khi model đã hoàn toàn huấn luyện xong. Test set **không được dùng** trong bất kỳ quyết định nào khi train → cho biết model hoạt động thế nào trên dữ liệu hoàn toàn mới

### Q4: Vì sao dùng Softmax?

**Trả lời:** Vì đây là bài toán **phân loại đa lớp (multi-class), mỗi ảnh CHỈ thuộc MỘT class**. Softmax đảm bảo:

- Output là phân phối xác suất: tất cả giá trị ≥ 0 và tổng = 1
- Class có xác suất cao nhất được chọn làm dự đoán
- Có thể diễn giải confidence (độ tin cậy) của model

Nếu dùng Sigmoid (cho multi-label, mỗi ảnh có thể thuộc nhiều class), tổng xác suất có thể > 1 hoặc < 1 → không phù hợp.

### Q5: Vì sao dùng Dropout?

**Trả lời:** Dense(128) có ~164K tham số — là layer có nhiều tham số nhất trong classification head. Với chỉ ~4200 ảnh train, 164K tham số dễ dẫn đến overfitting (model "ghi nhớ" ảnh train thay vì học đặc trưng). Dropout(0.4) tắt ngẫu nhiên 40% neuron trong mỗi batch → buộc mỗi neuron phải học đặc trưng hữu ích độc lập, không dựa dẫm vào neuron khác → giảm overfitting.

### Q6: Vì sao một số class có F1 thấp?

**Trả lời:** Các class như Apple (F1=0.38), Peach (F1=0.35), Cherry (F1=0.44) có F1 thấp vì:

- **Nguyên nhân chính:** Ảnh 100×100 pixels không đủ độ phân giải để phân biệt các chi tiết tinh tế giữa quả tròn màu đỏ/vàng (táo vs đào vs cherry vs lê)
- **Nguyên nhân phụ:** Fruits-360 chứa rất nhiều biến thể của Apple (30+ folder: Apple Red, Apple Golden, Apple Granny Smith,...) → nội bộ class Apple đã rất đa dạng → khó học đặc trưng chung
- **Ngược lại:** Kiwi (F1=1.0) có màu nâu + texture lông độc nhất, không class nào giống → model học rất dễ

### Q7: Nếu muốn cải thiện accuracy thì làm gì?

**Trả lời (theo thứ tự ưu tiên):**

1. **Dùng ảnh độ phân giải cao hơn:** Bản Fruits-360 gốc (ảnh ~500×500 hoặc 1000×1000) thay vì bản 100×100 → giữ được chi tiết texture
2. **Thử EfficientNetB0/B1:** Hiệu quả hơn MobileNetV2 trong khi vẫn nhẹ
3. **Thêm background augmentation:** Thay nền trắng bằng nền ngẫu nhiên → model không phụ thuộc vào nền
4. **Tăng dữ liệu:** Dùng toàn bộ ảnh trong Fruits-360 (không giới hạn 400/class)
5. **Fine-tune nhiều layer hơn:** Thử mở 40-50% lớp cuối thay vì 25%
6. **Label smoothing:** Thay vì one-hot cứng [0,0,1,0,...], dùng [0.01, 0.01, 0.86, 0.01,...] → giảm overconfidence
7. **Ensemble:** Kết hợp MobileNetV2 + EfficientNetB0 → dự đoán chính xác hơn

### Q8: Model này có dùng được ngoài thực tế không?

**Trả lời:** Hiện tại **chưa**. Lý do:

- Fruits-360 là dataset phòng thí nghiệm (nền trắng, studio lighting). Ảnh ngoài đời có nền phức tạp, nhiều vật thể, góc chụp đa dạng → model sẽ hoạt động kém
- Thiếu: Background augmentation, ảnh từ nhiều nguồn khác nhau, cơ chế phát hiện "không phải trái cây"
- Để dùng thực tế, cần: (1) Fine-tune thêm với ảnh thực tế, (2) Thêm object detection để tìm vị trí quả trong ảnh, (3) Thêm confidence threshold để từ chối ảnh không phải trái cây

---

## 12. Cách trình bày kết quả mô hình trong báo cáo

Dưới đây là gợi ý cách viết từng phần trong báo cáo đồ án. Văn phong sinh viên, tiếng Việt, rõ ràng.

---

### 12.1 Mô hình đề xuất

> Mô hình đề xuất sử dụng **Transfer Learning với MobileNetV2**, một mạng CNN nhẹ do Google phát triển, được huấn luyện trước trên tập dữ liệu ImageNet (1.4 triệu ảnh, 1000 lớp). Phần base của MobileNetV2 đóng vai trò trích xuất đặc trưng từ ảnh đầu vào. Bên trên base, nhóm xây dựng một classification head mới gồm:
>
> - **GlobalAveragePooling2D:** Chuyển feature map 5×5×1280 thành vector 1280 chiều
> - **Dense(128) + ReLU:** Học cách kết hợp các đặc trưng cho bài toán trái cây
> - **Dropout(0.4):** Giảm overfitting trong quá trình huấn luyện
> - **Dense(15) + Softmax:** Đầu ra là xác suất cho 15 loại trái cây
>
> Mô hình được huấn luyện theo 2 pha:
> - **Pha 1 (Huấn luyện head):** Toàn bộ MobileNetV2 bị đóng băng, chỉ huấn luyện classification head trong tối đa 25 epoch, có EarlyStopping và ReduceLROnPlateau
> - **Pha 2 (Fine-tune):** Mở khóa 25% lớp cuối của MobileNetV2, huấn luyện với learning rate thấp hơn 10 lần (0.0001) trong tối đa 7 epoch
>
> Đầu vào mô hình là ảnh RGB kích thước 160×160 pixels, được chuẩn hóa về khoảng [-1, 1] bằng hàm `preprocess_input` của MobileNetV2. Data augmentation được áp dụng cho tập huấn luyện với các phép biến đổi: xoay ±30°, zoom 0.7-1.3x, dịch ngang/dọc ±15%, lật ngang, thay đổi độ sáng 0.7-1.3x, và kéo nghiêng 10°.

---

### 12.2 Quy trình huấn luyện

> Quy trình huấn luyện được thực hiện qua các bước:
>
> 1. **Chuẩn bị dữ liệu:** Từ dataset Fruits-360 bản 100×100, chọn 15 class trái cây, mỗi class tối đa 400 ảnh. Chia thành train (70%), validation (30%) từ thư mục Training gốc, và test từ thư mục Test gốc (hoàn toàn tách biệt).
>
> 2. **Xây dựng data pipeline:** Sử dụng `ImageDataGenerator` của Keras để load ảnh theo batch (32 ảnh/batch), tự động resize về 160×160, áp dụng augmentation cho tập train và chỉ chuẩn hóa cho tập validation.
>
> 3. **Pha 1:** Huấn luyện classification head với optimizer Adam (lr=0.001), loss function Categorical Crossentropy. Sử dụng 3 callback: EarlyStopping (patience=5, theo dõi val_loss), ReduceLROnPlateau (giảm lr 50% sau 3 epoch không cải thiện), ModelCheckpoint (lưu model tốt nhất theo val_accuracy).
>
> 4. **Pha 2:** Mở khóa 25% lớp cuối của MobileNetV2, compile lại với Adam (lr=0.0001), tiếp tục huấn luyện tối đa 7 epoch với EarlyStopping và ModelCheckpoint.
>
> 5. **Đánh giá:** Sử dụng tập test (5034 ảnh, chưa từng thấy trong quá trình train) để tính Accuracy, Precision, Recall, F1-score và vẽ Confusion Matrix.

---

### 12.3 Đánh giá mô hình

> Mô hình đạt **68.28% accuracy** trên tập test 5034 ảnh, với Macro F1-score = 0.7255. Chi tiết từng lớp cho thấy sự phân hóa rõ rệt:
>
> **Các lớp dễ (F1 > 0.95):** Kiwi (1.00), Strawberry (0.99), Pineapple (0.99), Watermelon (0.98), Avocado (0.97), Blueberry (0.97). Đây là những loại quả có màu sắc hoặc hình dạng đặc trưng, không trùng lặp với các class khác.
>
> **Các lớp khó (F1 < 0.60):** Apple (0.38), Peach (0.35), Cherry (0.44), Orange (0.52), Lemon (0.55), Grape (0.59). Nguyên nhân chính: ở độ phân giải 100×100 pixels, các loại quả tròn có màu đỏ/vàng/cam dễ bị nhầm lẫn với nhau. Apple và Peach đều tròn, màu đỏ-hồng. Cherry tròn đỏ nhưng nhỏ hơn — tuy nhiên sau khi resize về 160×160, thông tin kích thước bị mất.
>
> Confusion matrix xác nhận: Apple thường bị dự đoán nhầm thành Pear và Peach; Peach bị nhầm thành Apple; Orange bị nhầm thành Lemon và Peach.
>
> So với CNN tự xây dựng trước đó (53.30% accuracy), Transfer Learning MobileNetV2 cải thiện đáng kể (+15%), cho thấy hiệu quả của việc tận dụng tri thức từ ImageNet.

---

### 12.4 Nhận xét kết quả

> - MobileNetV2 Transfer Learning hoạt động tốt với dataset nhỏ (~4200 ảnh train), chứng tỏ việc tận dụng pre-trained weights từ ImageNet là chiến lược hiệu quả.
> - Data Augmentation giúp giảm khoảng cách giữa train accuracy và validation accuracy (dấu hiệu của overfitting), đặc biệt quan trọng khi huấn luyện trên dataset studio có điều kiện chụp đồng nhất.
> - Hạn chế lớn nhất đến từ độ phân giải ảnh: 100×100 pixels không đủ để phân biệt các chi tiết texture tinh tế giữa các loại quả tròn cùng màu. Đây là giới hạn của dataset, không phải của kiến trúc mô hình.
> - Mô hình có xu hướng hoạt động tốt với các lớp có đặc trưng "độc nhất" (màu sắc, texture đặc biệt) và kém với các lớp có đặc trưng "chia sẻ" (hình tròn + đỏ/vàng).

---

### 12.5 Hạn chế

> - **Độ phân giải thấp:** Ảnh 100×100 pixels hạn chế khả năng phân biệt chi tiết texture.
> - **Dataset studio:** Fruits-360 chụp trong môi trường kiểm soát (nền trắng, ánh sáng chuẩn). Mô hình có thể hoạt động kém trên ảnh chụp ngoài thực tế.
> - **Số lớp hạn chế:** Chỉ phân loại được 15 loại trái cây đã huấn luyện. Không có cơ chế từ chối ảnh không thuộc 15 lớp.
> - **Thiếu đa dạng trạng thái:** Dataset không chứa ảnh quả bị cắt, dập, hoặc chưa chín.
> - **Thời gian huấn luyện:** ~30-40 phút trên CPU, tuy chấp nhận được nhưng có thể cải thiện nếu có GPU.
> - **Giao diện hiển thị một số thông số chưa chính xác:** Sidebar của ứng dụng Streamlit hiển thị kích thước ảnh 128×128 và learning rate 0.0005, trong khi thực tế model sử dụng ảnh 160×160 và learning rate 0.001/0.0001. Đây là lỗi hiển thị, không ảnh hưởng đến chức năng.

---

### 12.6 Hướng phát triển

> - **Tăng độ phân giải:** Sử dụng bản Fruits-360 gốc (ảnh ~500×500 hoặc 1000×1000) để cải thiện độ chính xác, đặc biệt với các class khó.
> - **Thử nghiệm kiến trúc khác:** EfficientNetB0/B1 có thể cho kết quả tốt hơn MobileNetV2 trong khi vẫn nhẹ.
> - **Background augmentation:** Thêm phép biến đổi thay đổi nền ảnh để model không phụ thuộc vào nền trắng.
> - **Triển khai thực tế:** Thu thập thêm ảnh từ môi trường thực tế, fine-tune lại model.
> - **Thêm tính năng:** Phát hiện độ chín của quả, nhận diện nhiều quả trong cùng một ảnh (object detection).
> - **Thêm cơ chế từ chối:** Thiết lập ngưỡng confidence (vd: 70%) — nếu confidence dưới ngưỡng, trả về "Không xác định".

---

## 13. Tóm tắt mô hình trong 10 câu

Dưới đây là 10 câu tóm tắt để trả lời nhanh khi thuyết trình:

1. **Bài toán:** Phân loại ảnh thuộc 1 trong 15 loại trái cây, sử dụng dataset Fruits-360 bản 100×100 pixels.

2. **Kiến trúc:** MobileNetV2 pre-trained trên ImageNet (không có top) + GlobalAveragePooling2D + Dense(128) + Dropout(0.4) + Softmax(15).

3. **Tiền xử lý:** Ảnh được resize về 160×160 pixels và chuẩn hóa về khoảng [-1, 1] bằng hàm `preprocess_input` của MobileNetV2.

4. **Augmentation:** Áp dụng 7 phép biến đổi cho tập train (xoay ±30°, zoom 0.7-1.3x, dịch ±15%, lật ngang, độ sáng 0.7-1.3x, kéo nghiêng 10°) để chống overfitting.

5. **Huấn luyện 2 pha:** Pha 1 huấn luyện classification head với base đóng băng (lr=0.001, tối đa 25 epoch). Pha 2 fine-tune 25% lớp cuối của MobileNetV2 (lr=0.0001, tối đa 7 epoch).

6. **Callback:** EarlyStopping (dừng nếu val_loss không giảm 5 epoch), ReduceLROnPlateau (giảm lr khi plateau), ModelCheckpoint (lưu model tốt nhất).

7. **Dữ liệu:** Train 70%, validation 30% từ Training gốc (~4200 ảnh). Test từ thư mục Test gốc (5034 ảnh), hoàn toàn tách biệt — không data leakage.

8. **Kết quả:** 68.28% accuracy trên test set. Macro F1 = 0.73. Lớp dễ nhất: Kiwi (F1=1.00). Lớp khó nhất: Peach (F1=0.35), Apple (F1=0.38).

9. **Nguyên nhân chính của lỗi:** Ảnh 100×100 không đủ độ phân giải để phân biệt chi tiết giữa các quả tròn cùng màu (táo-đào-cherry) và dataset nền trắng đồng nhất khiến model không tổng quát tốt ra ảnh thực tế.

10. **So với baseline:** Transfer Learning MobileNetV2 (68.28%) cải thiện +15% accuracy so với CNN tự xây dựng (53.30%), chứng tỏ hiệu quả của việc tận dụng pre-trained weights.

---

*File này được tạo từ việc phân tích source code thực tế của project, không dựa trên giả định. Mọi thông tin đều có thể kiểm chứng bằng cách đọc code tương ứng.*
