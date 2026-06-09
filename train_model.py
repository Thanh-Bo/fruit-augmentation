"""
train_model.py - Huấn luyện mô hình Transfer Learning MobileNetV2 phân loại trái cây.

Quy trình:
1. Load dữ liệu từ TRAIN_DIR và VALIDATION_DIR.
2. Áp dụng data augmentation cho tập train.
3. Xây dựng mô hình Transfer Learning với MobileNetV2 pre-trained.
4. Phase 1: Huấn luyện classification head (base đóng băng).
5. Phase 2: Fine-tune một phần base model.
6. Lưu model và class_indices.
7. Vẽ biểu đồ accuracy/loss.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    TRAIN_DIR, VALIDATION_DIR,
    IMG_SIZE, BATCH_SIZE, EPOCHS, FINE_TUNE_EPOCHS, NUM_CLASSES,
    MODEL_DIR, MODEL_PATH, CLASS_INDICES_PATH,
    RESULTS_DIR,
    AUGMENTATION_CONFIG
)


def create_data_generators():
    """
    Tạo ImageDataGenerator cho tập train (có augmentation) và validation.

    Sử dụng MobileNetV2 preprocess_input (scale về [-1, 1]).

    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation cho tập train
    # Giảm nhẹ augmentation so với custom CNN vì MobileNetV2 đã học đặc trưng tốt
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=AUGMENTATION_CONFIG["rotation_range"],
        zoom_range=AUGMENTATION_CONFIG["zoom_range"],
        width_shift_range=AUGMENTATION_CONFIG["width_shift_range"],
        height_shift_range=AUGMENTATION_CONFIG["height_shift_range"],
        horizontal_flip=AUGMENTATION_CONFIG["horizontal_flip"],
        brightness_range=AUGMENTATION_CONFIG["brightness_range"],
        shear_range=AUGMENTATION_CONFIG["shear_range"],
        fill_mode=AUGMENTATION_CONFIG["fill_mode"]
    )

    # Validation: chỉ preprocess, không augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Load dữ liệu từ thư mục
    print("\nĐang load dữ liệu training...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )

    print("\nĐang load dữ liệu validation...")
    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    # In thông tin class
    print(f"\nClass indices: {train_generator.class_indices}")

    return train_generator, validation_generator


def build_mobilenetv2_model():
    """
    Xây dựng mô hình Transfer Learning với MobileNetV2 pre-trained.

    Kiến trúc:
    - Base: MobileNetV2 (pre-trained trên ImageNet, 1.4M ảnh).
    - Head: GlobalAveragePooling + Dense(128) + Dropout(0.4) + Softmax(15).

    Base được đóng băng ban đầu, sau đó fine-tune 30% lớp cuối.

    Returns:
        tf.keras.Model: Mô hình đã compile.
    """
    print("\nĐang tải MobileNetV2 pre-trained (ImageNet)...")

    # Load MobileNetV2 không có top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=1.0  # Standard MobileNetV2
    )

    # Đóng băng base model ban đầu
    base_model.trainable = False

    print(f"  Base model: {base_model.name}")
    print(f"  Số lớp trong base: {len(base_model.layers)}")
    print(f"  Base trainable: {base_model.trainable}")

    # Xây dựng classification head
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(128, activation='relu', name='dense_head')(x)
    x = Dropout(0.4, name='dropout_head')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='FruitMobileNetV2')

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n  Tổng số lớp: {len(model.layers)}")
    print(f"  Trainable params (phase 1): "
          f"{model.trainable_weights}")
    model.summary()

    return model, base_model


def train_phase1_head(model, train_gen, val_gen):
    """
    Phase 1: Huấn luyện classification head (base đóng băng).

    Returns:
        History: Lịch sử huấn luyện phase 1.
    """
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    print(f"\nsteps_per_epoch: {steps_per_epoch}")
    print(f"validation_steps: {validation_steps}")
    print(f"Train samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    print("\n" + "=" * 60)
    print("PHASE 1: HUẤN LUYỆN CLASSIFICATION HEAD (Base đóng băng)")
    print("=" * 60)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    print(f"\n[HOÀN THÀNH PHASE 1] Số epoch đã train: "
          f"{len(history.history['loss'])}")
    print(f"  Val accuracy cuối: "
          f"{history.history['val_accuracy'][-1]:.4f}")

    return history


def train_phase2_finetune(model, base_model, train_gen, val_gen):
    """
    Phase 2: Fine-tune 30% lớp cuối của MobileNetV2.

    Args:
        model: Model đã train phase 1.
        base_model: MobileNetV2 base model.
        train_gen: Training generator.
        val_gen: Validation generator.

    Returns:
        History: Lịch sử phase 2 (nối tiếp phase 1).
    """
    # Mở khóa base model
    base_model.trainable = True

    # Chỉ fine-tune 25% lớp cuối của base (cân bằng)
    fine_tune_at = int(len(base_model.layers) * 0.75)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNE BASE MODEL")
    print("=" * 60)
    print(f"  Tổng lớp base: {len(base_model.layers)}")
    print(f"  Fine-tune từ lớp: {fine_tune_at}")
    print(f"  Số lớp trainable trong base: "
          f"{sum(1 for l in base_model.layers if l.trainable)}")

    # Compile lại với learning rate thấp hơn
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print(f"\n[HOÀN THÀNH PHASE 2] Số epoch fine-tune: "
          f"{len(history.history['loss'])}")
    print(f"  Val accuracy cuối: "
          f"{history.history['val_accuracy'][-1]:.4f}")

    return history


def plot_training_history(history1, history2=None):
    """
    Vẽ biểu đồ accuracy và loss. Nếu có phase 2, nối tiếp biểu đồ.

    Args:
        history1: History của phase 1.
        history2: History của phase 2 (None nếu không có).
    """
    # Gộp history nếu có phase 2
    if history2 is not None:
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = (history1.history['val_accuracy']
                   + history2.history['val_accuracy'])
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = (history1.history['val_loss']
                    + history2.history['val_loss'])
    else:
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Biểu đồ Accuracy
    ax1.plot(epochs_range, acc, label='Train Accuracy',
             color='#2563eb', marker='o', markersize=4, linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy',
             color='#16a34a', marker='s', markersize=4, linewidth=2)

    # Đánh dấu điểm bắt đầu phase 2
    if history2 is not None:
        phase1_end = len(history1.history['accuracy'])
        ax1.axvline(x=phase1_end, color='#f59e0b', linestyle='--',
                    linewidth=1.5, alpha=0.8, label='Bắt đầu Fine-tune')

    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Biểu đồ Loss
    ax2.plot(epochs_range, loss, label='Train Loss',
             color='#2563eb', marker='o', markersize=4, linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss',
             color='#16a34a', marker='s', markersize=4, linewidth=2)

    if history2 is not None:
        phase1_end = len(history1.history['loss'])
        ax2.axvline(x=phase1_end, color='#f59e0b', linestyle='--',
                    linewidth=1.5, alpha=0.8, label='Bắt đầu Fine-tune')

    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'accuracy_loss.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Đã lưu biểu đồ accuracy/loss tại: {output_path}")


def save_class_indices(train_gen):
    """
    Luu anh xa class_index -> class_name vao file JSON.

    Args:
        train_gen: Training generator (co thuoc tinh class_indices).
    """
    class_indices = train_gen.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}

    with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_to_class, f, ensure_ascii=False, indent=2)

    print(f"[OK] Đã lưu class_indices tại: {CLASS_INDICES_PATH}")
    print(f"  Nội dung: {index_to_class}")


def main():
    """
    Hàm chính - thực hiện toàn bộ quy trình huấn luyện Transfer Learning.
    """
    print("\n" + "#" * 60)
    print("#  HUẤN LUYỆN TRANSFER LEARNING MOBILENETV2")
    print("#  PHÂN LOẠI TRÁI CÂY")
    print("#" * 60)

    # Tạo thư mục model và results nếu chưa tồn tại
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Kiểm tra dữ liệu train/validation
    if not os.path.exists(TRAIN_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục train: {TRAIN_DIR}")
        print("Hãy chạy prepare_dataset.py trước.")
        return

    if not os.path.exists(VALIDATION_DIR):
        print(f"\n[LỖI] Không tìm thấy thư mục validation: {VALIDATION_DIR}")
        print("Hãy chạy prepare_dataset.py trước.")
        return

    # Bước 1: Tạo data generators
    train_generator, validation_generator = create_data_generators()

    # Bước 2: Xây dựng model Transfer Learning
    print("\n" + "=" * 60)
    print("KIẾN TRÚC MÔ HÌNH TRANSFER LEARNING")
    print("=" * 60)
    model, base_model = build_mobilenetv2_model()

    # Bước 3: Phase 1 - Huấn luyện classification head
    history1 = train_phase1_head(
        model, train_generator, validation_generator
    )

    # Bước 4: Phase 2 - Fine-tune base model
    print("\n" + "-" * 60)
    print("Bắt đầu Phase 2: Fine-tune...")
    print("-" * 60)
    history2 = train_phase2_finetune(
        model, base_model, train_generator, validation_generator
    )

    # Bước 5: Vẽ biểu đồ
    plot_training_history(history1, history2)

    # Bước 6: Lưu class indices
    save_class_indices(train_generator)

    # Tổng kết
    total_epochs = (len(history1.history['loss'])
                    + len(history2.history['loss']))
    final_val_acc = history2.history['val_accuracy'][-1]
    print("\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Tổng số epoch: {total_epochs}")
    print(f"  Phase 1: {len(history1.history['loss'])} epochs")
    print(f"  Phase 2: {len(history2.history['loss'])} epochs")
    print(f"  Val accuracy cuối cùng: {final_val_acc:.4f} "
          f"({final_val_acc*100:.2f}%)")
    print(f"\n  Model đã lưu tại: {MODEL_PATH}")
    print(f"  Class indices tại: {CLASS_INDICES_PATH}")
    print(f"  Biểu đồ tại: {RESULTS_DIR}/accuracy_loss.png")
    print("\n[HOÀN THÀNH] Chạy evaluate_model.py để đánh giá mô hình.")


if __name__ == "__main__":
    main()
