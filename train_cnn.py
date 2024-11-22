import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Define paths
DATA_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Train'
MODEL_PATH = "models/mri_error_detector.keras"

# Load 2D MRI data and labels
def load_data(data_dir):
    X, y = [], []
    for label, class_dir in enumerate(["Normal", "Tumor"]):
        class_path = os.path.join(data_dir, class_dir)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            
            # Read the image using OpenCV
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error loading image: {file_path}")
                continue
            
            # Resize the image to the required size (e.g., 64x64 for the model)
            img_resized = cv2.resize(img, (64, 64))
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_resized / 255.0
            
            # Add the image to the dataset
            X.append(img_normalized)
            y.append(label)
    
    # Convert lists to numpy arrays
    return np.array(X), np.array(y)

# Load and preprocess data
X, y = load_data(DATA_DIR)
X = X[..., np.newaxis]  # Add channel dimension (for grayscale image)
y = to_categorical(y, num_classes=2)  # Convert labels to one-hot encoding

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 2D CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Save the trained model after training
model.save("mri_error_detector_final.keras")
print("Training completed! Model saved at: mri_error_detector_final.keras")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Check predictions on a few validation samples
samples = X_val[:5]
true_labels = np.argmax(y_val[:5], axis=1)  # Convert one-hot labels back to integers

# Make predictions
predictions = model.predict(samples)

# Plot images with true vs predicted labels
for i in range(5):
    plt.imshow(samples[i].squeeze(), cmap='gray')  # Remove the channel dimension for display
    plt.title(f"True: {'Normal' if true_labels[i] == 0 else 'Tumor'}, Predicted: {'Normal' if np.argmax(predictions[i]) == 0 else 'Tumor'}")
    plt.show()

# Model Inference on New Image
img_path = "path_to_new_image.jpg"  # Replace with the actual path to the image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (64, 64))  # Resize to the same size as training images
img_normalized = img_resized / 255.0  # Normalize the image
img_input = img_normalized[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

# Make prediction
prediction = model.predict(img_input)
predicted_class = "Normal" if np.argmax(prediction) == 0 else "Tumor"
print(f"Prediction: {predicted_class}")
