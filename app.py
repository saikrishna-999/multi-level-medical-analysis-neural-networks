from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json  # Import JSON module

app = Flask(__name__)

# Load the trained 3D CNN model globally
model = load_model('models/mri_error_detector.keras')

# Path to the folder containing baseline images
BASELINE_IMAGE_DIR = r'C:\Users\saikr\OneDrive\Desktop\MRI_Error_Detection_Project\Brain MRI Images\Validation\Normal'

# Load lobe disadvantages from JSON file
with open('brain_lobes_info.json') as json_file:
    lobe_info = json.load(json_file)

# Preprocessing function for the image
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

# Function to highlight lesion areas based on pixel intensity
def highlight_differences(original_image_path, is_abnormal):
    original_image = cv2.imread(original_image_path)
    dot_centers = []

    if is_abnormal:
        # Convert to grayscale and apply threshold to detect bright areas
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours (possible lesion areas)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and select up to 3 largest regions
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        # Draw red dots on the largest lesion areas
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(original_image, (cx, cy), 10, (0, 0, 255), -1)  # Red dot
                dot_centers.append((cx, cy))  # Store the center of each dot

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
        # Assuming the image width and height are 256x256
        # Mapping dots to regions based on their x-coordinates
        if cx < 85:  # Left side
            affected_regions.add(region_mapping[1])  # Frontal Lobe
        elif cx < 170:  # Middle
            affected_regions.add(region_mapping[2])  # Parietal Lobe
        else:  # Right side
            affected_regions.add(region_mapping[3])  # Occipital Lobe

    return list(affected_regions)

# Prediction function
def predict_and_generate_graph(file_path):
    try:
        img = preprocess_image(file_path)
        if img is None:
            return None, 'Error: Image preprocessing failed'

        img = img.reshape(1, 256, 256, 1)  # Adjust shape for your model
        prediction = model.predict(img)
        prediction_value = prediction[0][0]

        # Define threshold and result
        threshold = 0.5
        result = 'Normal' if prediction_value < threshold else 'Abnormal'

        # Generate the graph
        fig, ax = plt.subplots()
        categories = ['Normal', 'Abnormal']
        probabilities = [1 - prediction_value, prediction_value]

        ax.bar(categories, probabilities, color=['green', 'red'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability')
        ax.set_title('MRI Error Detection')

        # Save the graph to a BytesIO object
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)

        # Encode the graph as base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close(fig)

        return img_base64, result

    except Exception as e:
        print(f"Error during prediction or graph generation: {e}")
        return None, 'Error: Prediction failed'

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

    # Ensure 'uploads' directory exists
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    # Generate a unique filename
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(upload_dir, unique_filename)

    try:
        # Save the uploaded file temporarily
        file.save(file_path)

        # Make prediction and generate the graph
        img_base64, result = predict_and_generate_graph(file_path)

        if img_base64 is None:
            return jsonify({'error': result}), 500

        # Highlight differences on the uploaded image
        is_abnormal = result == 'Abnormal'
        highlighted_image, dot_centers = highlight_differences(file_path, is_abnormal)
        highlighted_image_base64 = image_to_base64(highlighted_image)

        # Determine affected regions based on the dot centers
        affected_region_names = get_affected_regions(dot_centers)

        # Clean up the file after prediction
        os.remove(file_path)

        # Render the result page with the prediction result, graph, highlighted image, and affected regions
        return render_template('result.html', prediction=result, graph=img_base64, highlighted_image=highlighted_image_base64, affected_region_names=affected_region_names)

    except Exception as e:
        print(f"Error during file upload or prediction: {e}")
        return jsonify({'error': 'File upload or prediction failed'}), 500

# Route for detailed report based on affected regions
@app.route('/report', methods=['GET'])
def report():
    affected_regions = request.args.getlist('regions')

    report_data = {}
    for region in affected_regions:
        # Fetch lobe info from the JSON data
        region_data = lobe_info.get(region)
        if region_data:
            report_data[region] = region_data
        else:
            report_data[region] = {
                "disadvantages": "No data available",
                "functionalities": "No data available",
                "doctor_tips": "No data available"
            }
    
    print("Report Data:", report_data)  # Add this line to check the output
    return render_template('report.html', report_data=report_data)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
