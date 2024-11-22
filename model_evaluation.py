import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
)
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
TEST_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Train'
MODEL_SAVE_DIR = 'models/'
PLOT_SAVE_DIR = 'static/plots/'

# Ensure plot directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Preprocessing function for EfficientNet (224x224 RGB)
def preprocess_data_efficientnet(data_dir, img_size):
    images, labels = [], []
    for label in ['Normal', 'Tumor']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Directory not found: {label_dir}")
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (img_size, img_size))
                img_normalized = img_resized / 255.0  # Normalize image for EfficientNet
                images.append(np.stack([img_normalized] * 3, axis=-1))  # Convert grayscale to RGB
                labels.append(0 if label == 'Normal' else 1)
            else:
                print(f"Warning: Could not read image {file_path}. Skipping...")
    return np.array(images), np.array(labels)

# Load test data for EfficientNet
X_test_efficientnet, y_test = preprocess_data_efficientnet(TEST_DIR, 224)

# Load EfficientNet model
efficientnet_model = load_model(os.path.join(MODEL_SAVE_DIR, 'efficientnet_b0_model.h5'))

# Evaluate EfficientNet model
def evaluate_efficientnet_model():
    predictions = efficientnet_model.predict(X_test_efficientnet)
    y_pred = (predictions > 0.5).astype(int).flatten()
    evaluate_and_plot(y_test, y_pred, predictions.flatten(), 'efficientnet')

# Evaluation and plotting function
def evaluate_and_plot(y_true, y_pred, y_prob, model_name):
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"{model_name.upper()} Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")

    # Save metrics to a text file
    with open(os.path.join(PLOT_SAVE_DIR, f'{model_name}_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title(f'{model_name.upper()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f'{model_name.upper()} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{model_name}_roc_curve.png'))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green')
    plt.title(f'{model_name.upper()} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{model_name}_pr_curve.png'))
    plt.close()

# Main function
if __name__ == '__main__':
    print("Evaluating EfficientNet model...")
    evaluate_efficientnet_model()
    print("Evaluation complete. Metrics and plots saved!")
