import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Update the path here
TRAIN_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Train'

# Ensure the 'models' directory exists for saving the model
if not os.path.exists('models'):
    os.makedirs('models')

# Load and preprocess data
def load_data(data_dir):
    images, labels = [], []
    for label in ['Normal', 'Tumor']:  # Ensure the directories 'Normal' and 'Tumor' exist in the 'Train' folder
        class_dir = os.path.join(data_dir, label)
        print(f"Looking for class directory: {class_dir}")  # Debug print
        label_value = 0 if label == 'Normal' else 1  # Label 0 for Normal, 1 for Tumor
        if not os.path.exists(class_dir):
            print(f"Error: Directory {class_dir} does not exist!")  # Error if directory is missing
            continue  # Skip to the next label if the directory does not exist
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            print(f"Found file: {file_path}")  # Debug print
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))  # Resize image to 64x64
                images.append(img_resized.flatten())  # Flatten the image
                labels.append(label_value)
            else:
                print(f"Warning: Unable to read image {file_name}. Skipping...")
    return np.array(images), np.array(labels)

# Train and save the Random Forest model
def train_random_forest_model(data_dir, save_path='models/rf_model.pkl', scaler_path='models/scaler.pkl'):
    # Load dataset
    X, y = load_data(data_dir)

    # If no data loaded or only one class found, stop the training
    if X.size == 0 or y.size == 0:
        print("No data loaded. Check your dataset paths and try again.")
        return

    if len(np.unique(y)) == 1:
        print("Error: Only one class found. The dataset must contain both Normal and Tumor classes.")
        return

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize features to [0, 1]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Train the Random Forest model
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.4f}")

    # Save the trained model
    joblib.dump(rf_model, save_path)
    print(f"Random Forest model saved at {save_path}")

    # Save the scaler for future use
    joblib.dump(scaler, scaler_path)
    print(f"Scaler model saved at {scaler_path}")

# Run training if this file is executed directly
if __name__ == '__main__':
    train_random_forest_model(TRAIN_DIR)
