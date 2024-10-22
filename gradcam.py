import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def generate_gradcam(image_path, model, layer_name='conv2d'):
    try:
        # Load and preprocess the input image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=0) / 255.0

        # Ensure the model is initialized by calling it once with a dummy input
        _ = model.predict(np.zeros((1, 256, 256, 1)))

        # Create a model that outputs both the target layer and predictions
        target_layer = model.get_layer(layer_name).output
        grad_model = Model(inputs=model.input, outputs=[target_layer, model.output])

        # Compute gradients of the predicted class with respect to the feature map
        with K.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]  # Assuming binary classification

        grads = tape.gradient(loss, conv_outputs)[0]

        # Perform guided backpropagation: keep only positive gradients
        guided_grads = np.maximum(grads, 0)

        # Generate weights and apply them to the feature map
        weights = np.mean(guided_grads, axis=(0, 1))
        cam = np.dot(conv_outputs[0], weights)

        # Normalize the heatmap
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (256, 256))

        # Convert to heatmap
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on the original image
        original_img = cv2.imread(image_path)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Save the heatmap image
        heatmap_path = f"{image_path.rsplit('.', 1)[0]}_heatmap.png"
        cv2.imwrite(heatmap_path, overlay)

        return heatmap_path, None

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None, str(e)
