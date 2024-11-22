import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Paths
TRAIN_DIR = 'Brain MRI Images/Train'
VALID_DIR = 'Brain MRI Images/Validation'
MODEL_SAVE_DIR = 'models/'
PLOT_SAVE_DIR = 'static/plots/'

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Preprocessing function
def preprocess_data(data_dir, img_size):
    images, labels = [], []
    for label in ['Normal', 'Tumor']:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (img_size, img_size))
                images.append(img_resized.flatten() if img_size == 64 else img_resized)
                labels.append(0 if label == 'Normal' else 1)
    return np.array(images), np.array(labels)

# Load CNN data
X_train_cnn, y_train_cnn = preprocess_data(TRAIN_DIR, 256)
X_val_cnn, y_val_cnn = preprocess_data(VALID_DIR, 256)
X_train_cnn = X_train_cnn.reshape(-1, 256, 256, 1) / 255.0
X_val_cnn = X_val_cnn.reshape(-1, 256, 256, 1) / 255.0

# Train CNN Model
def train_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_cnn, y_train_cnn,
        validation_data=(X_val_cnn, y_val_cnn),
        epochs=15,  # Limited to 15 epochs
        batch_size=32,
        callbacks=[early_stopping]
    )
    model.save(os.path.join(MODEL_SAVE_DIR, 'cnn_model.keras'))
    plot_training_history(history, 'cnn_training')
    print("CNN model trained and saved!")

# Plot training history
def plot_training_history(history, plot_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{plot_name}_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{plot_name}_accuracy.png'))
    plt.close()

# Load RF data
X_train_rf, y_train_rf = preprocess_data(TRAIN_DIR, 64)
X_val_rf, y_val_rf = preprocess_data(VALID_DIR, 64)
scaler = StandardScaler()
X_train_rf = scaler.fit_transform(X_train_rf)
X_val_rf = scaler.transform(X_val_rf)

# Train Random Forest Model
def train_rf_model():
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_rf, y_train_rf)
    y_val_pred = rf_model.predict(X_val_rf)
    accuracy = accuracy_score(y_val_rf, y_val_pred)
    print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
    joblib.dump(rf_model, os.path.join(MODEL_SAVE_DIR, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'rf_scaler.pkl'))
    print("Random Forest model trained and saved!")

# Main function to train both models
if __name__ == '__main__':
    print("Training CNN model...")
    train_cnn_model()
    print("Training Random Forest model...")
    train_rf_model()
