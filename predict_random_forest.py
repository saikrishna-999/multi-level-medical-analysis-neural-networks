import os
import cv2
import numpy as np
import joblib

# Path to the trained model and scaler
MODEL_PATH = 'models/rf_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Load the trained model and scaler
rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the function to load and preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    img_resized = cv2.resize(img, (64, 64))  # Resize image to match the model's input shape
    return img_resized.flatten()  # Flatten the image to match the training data shape

# Define the prediction function
def predict(image_path):
    # Preprocess the image
    image_data = preprocess_image(image_path)
    if image_data is None:
        return "Error: Invalid image file"
    
    # Scale the image data using the same scaler used in training
    image_data_scaled = scaler.transform([image_data])  # Transform the image to match the training data
    
    # Make prediction using the Random Forest model
    prediction = rf_model.predict(image_data_scaled)
    
    # Return the prediction result
    if prediction == 0:
        return "Normal"
    else:
        return "Tumor"

# Main block to process images from subfolders (Normal and Tumor)
if __name__ == '__main__':
    # Root directory containing the subfolders 'Normal' and 'Tumor'
    test_image_dir = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Train'  # Replace with your folder path
    
    # Loop through the subfolders (Normal and Tumor)
    for folder_name in ['Normal', 'Tumor']:
        folder_path = os.path.join(test_image_dir, folder_name)
        
        # Check if the subfolder exists
        if not os.path.exists(folder_path):
            print(f"Error: Subfolder {folder_name} does not exist!")
            continue
        
        print(f"Processing images in the {folder_name} folder...")
        
        # Loop through all images in the folder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # Only process files with valid image extensions (e.g., .jpg, .jpeg, .png)
            if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Processing image: {filename}")
                result = predict(image_path)
                print(f"Prediction for {filename}: {result}")
            else:
                print(f"Skipping non-image file: {filename}")
