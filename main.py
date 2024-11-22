import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 1. Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    return img_resized / 255.0

# 2. Load Data from 'Brain_MRI_Images/' folder
def load_data(data_dir):
    images, labels = [], []
    for label in ["Normal", "Tumor"]:  # Class folders based on new structure
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(1 if label == "Tumor" else 0)  # Assign 1 for Tumor, 0 for Normal
    return np.array(images), np.array(labels)

# Load training and validation data
train_data, train_labels = load_data("Brain MRI Images/Train/")
val_data, val_labels = load_data("Brain MRI Images/Validation/")

# 3. Create CNN Model
def build_model():
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
    return model

# 4. Train and Evaluate Model
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(-1, 256, 256, 1)
X_test = X_test.reshape(-1, 256, 256, 1)

model = build_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# 6. Save Model
model.save('models/mri_error_detector.keras')
print("Model training complete and saved.")

# 7. Make Predictions (example)
def make_predictions(model, new_images):
    predictions = model.predict(new_images)
    return predictions

# Example usage for predictions
# Assuming you have new images loaded and preprocessed
# new_images = [preprocess_image(img_path) for img_path in new_image_paths]
# new_images = np.array(new_images).reshape(-1, 256, 256, 1)  # Reshape as needed
# predictions = make_predictions(model, new_images)

# 8. Visualize Predictions (example)
def visualize_predictions(images, predictions):
    for i in range(len(images)):
        plt.imshow(images[i].reshape(256, 256), cmap='gray')
        plt.title('Predicted: ' + ('Tumor' if predictions[i] > 0.5 else 'Normal'))
        plt.axis('off')
        plt.show()

# Assuming new_images is prepared and predictions made
# visualize_predictions(new_images, predictions)

# Load saved model (if needed later)
# model = load_model('models/mri_error_detector.keras')