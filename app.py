from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import joblib  # For Random Forest model and scaler

app = Flask(__name__)

# Load the trained 3D CNN model globally
model = load_model('models/mri_error_detector.keras')

# Load the Random Forest model and scaler globally
rf_model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Path to the folder containing baseline images
BASELINE_IMAGE_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Validation\Normal'

# Load lobe disadvantages from JSON file
with open('brain_lobes_info.json') as json_file:
    lobe_info = json.load(json_file)

# Preprocessing function for the CNN model
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        return img_normalized
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Preprocessing function for the Random Forest model
def preprocess_image_rf(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        img_resized = cv2.resize(img, (64, 64))  # Resize for RF input
        return img_resized.flatten()  # Flatten for RF input
    except Exception as e:
        print(f"Error during RF image preprocessing: {e}")
        return None

# Function to highlight lesion areas based on pixel intensity
def highlight_differences(original_image_path, is_abnormal):
    original_image = cv2.imread(original_image_path)
    dot_centers = []

    if is_abnormal:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(original_image, (cx, cy), 10, (0, 0, 255), -1)
                dot_centers.append((cx, cy))

    return original_image, dot_centers

# Function to convert image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# Function to map detected dots to brain regions
def get_affected_regions(dot_centers):
    region_mapping = {
        1: 'Frontal Lobe',
        2: 'Parietal Lobe',
        3: 'Occipital Lobe'
    }

    affected_regions = set()
    for (cx, cy) in dot_centers:
        if cx < 85:  # Left side
            affected_regions.add(region_mapping[1])  # Frontal Lobe
        elif cx < 170:  # Middle
            affected_regions.add(region_mapping[2])  # Parietal Lobe
        else:  # Right side
            affected_regions.add(region_mapping[3])  # Occipital Lobe

    return list(affected_regions)

# Function to predict with the Random Forest model
def predict_with_rf(image_path):
    img = preprocess_image_rf(image_path)
    if img is None:
        return None, 'Error: Image preprocessing failed'
    img_scaled = scaler.transform([img])
    prediction = rf_model.predict(img_scaled)
    prediction_value = rf_model.predict_proba(img_scaled)[0][1]  # Probability of being "Abnormal"
    result = 'Normal' if prediction[0] == 0 else 'Abnormal'
    return result, prediction_value

# Prediction function for CNN and generate side-by-side graph for CNN and Random Forest
def predict_and_generate_graph(file_path):
    try:
        # CNN Prediction
        img = preprocess_image(file_path)
        if img is None:
            return None, 'Error: Image preprocessing failed'

        img = img.reshape(1, 256, 256, 1)
        cnn_prediction = model.predict(img)
        cnn_confidence = cnn_prediction[0][0]
        cnn_result = 'Normal' if cnn_confidence < 0.5 else 'Abnormal'

        # RF Prediction
        rf_result, rf_confidence = predict_with_rf(file_path)

        # Create a figure with two subplots (side-by-side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  # Two plots side by side

        # CNN Bar Chart
        categories = ['Normal', 'Abnormal']
        cnn_probabilities = [1 - cnn_confidence, cnn_confidence]
        ax1.bar(categories, cnn_probabilities, color=['green', 'red'])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Probability')
        ax1.set_title('CNN Prediction')
        ax1.legend(['CNN'], loc='upper left')

        # RF Bar Chart
        rf_probabilities = [1 - rf_confidence, rf_confidence]
        ax2.bar(categories, rf_probabilities, color=['blue', 'orange'])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Probability')
        ax2.set_title('Random Forest Prediction')
        ax2.legend(['Random Forest'], loc='upper left')

        # Adjust the layout so that the graphs fit without overlapping
        plt.tight_layout()

        # Save the graph to a BytesIO object
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        graph_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close(fig)

        return graph_base64, cnn_result, cnn_confidence, rf_result, rf_confidence

    except Exception as e:
        print(f"Error during prediction or graph generation: {e}")
        return None, 'Error: Prediction failed', 0, 'Error', 0

# Route for the upload form
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle the file upload and display results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    file_ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(upload_dir, unique_filename)

    try:
        file.save(file_path)

        # Prediction and graph generation
        graph_base64, cnn_result, cnn_confidence, rf_result, rf_confidence = predict_and_generate_graph(file_path)

        if graph_base64 is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Highlight differences based on CNN prediction
        is_abnormal_cnn = cnn_result == 'Abnormal'
        highlighted_image, dot_centers = highlight_differences(file_path, is_abnormal_cnn)
        highlighted_image_base64 = image_to_base64(highlighted_image)

        # Determine affected regions based on the dot centers
        affected_region_names = get_affected_regions(dot_centers)

        os.remove(file_path)

        return render_template(
            'result.html',
            cnn_prediction=cnn_result,
            rf_prediction=rf_result,
            cnn_confidence=cnn_confidence,
            rf_confidence=rf_confidence,
            graph=graph_base64,
            highlighted_image=highlighted_image_base64,
            affected_region_names=affected_region_names
        )

    except Exception as e:
        print(f"Error during file upload or prediction: {e}")
        return jsonify({'error': 'File upload or prediction failed'}), 500

# Route for detailed report based on affected regions
@app.route('/report', methods=['GET'])
def report():
    affected_regions = request.args.getlist('regions')

    report_data = {}
    for region in affected_regions:
        region_data = lobe_info.get(region)
        if region_data:
            report_data[region] = region_data
        else:
            report_data[region] = {
                "disadvantages": "No detailed information available"
            }

    return render_template('report.html', report_data=report_data)

if __name__ == '__main__':
    app.run(debug=True)