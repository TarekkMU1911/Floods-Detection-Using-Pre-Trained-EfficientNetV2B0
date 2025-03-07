import numpy as np
import tifffile as tiff
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import os
import matplotlib.pyplot as plt

# Initialize Flask App
app = Flask(__name__, static_folder="static")

# Load Pretrained Model
MODEL_PATH = "Model/EfficientNetV2B0 Pretrained Model - Floods Detection.h5"
if os.path.exists(MODEL_PATH):
    PretrainedUNet = load_model(MODEL_PATH)
else:
    PretrainedUNet = None
    print("Error: Model file not found!")

# Image folder
IMAGE_FOLDER = "static/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Normalize image function
def normalize_image(image):
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    return (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

# Ensure input has exactly 12 channels
def preprocess_image(image, target_size=(128, 128, 12)):
    h, w, c = image.shape
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = normalize_image(image)
    
    # Resize image to match model input
    image = cv2.resize(image, (target_size[0], target_size[1]))
    
    # If the image has fewer than 12 bands, pad with zeros
    if c < 12:
        padded_image = np.zeros((target_size[0], target_size[1], 12), dtype=np.uint8)
        padded_image[:, :, :c] = image  # Copy available bands
        image = padded_image
    
    # Normalize for model input
    image = image / 255.0
    
    return image  # No batch dimension added yet

# Post Processing: Convert Probability to Binary Mask
def post_process_mask(mask, threshold=0.3655):
    return (mask > threshold).astype(np.uint8)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if PretrainedUNet is None:
        return render_template("index.html", error="Model not loaded. Check the model path.")
    
    file = request.files['file']
    if not file:
        return render_template("index.html", error="No file uploaded.")
    
    # Save uploaded file
    filepath = os.path.join(IMAGE_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Read the TIFF image
        image = tiff.imread(filepath).astype(np.float32)  # Ensure float32 for processing

        # Ensure correct shape
        if len(image.shape) == 2:  # Grayscale image (expand dims to create a single-channel "band")
            image = np.expand_dims(image, axis=-1)

        # Convert to uint8 for visualization
        vis_image = normalize_image(image)
        
        # Ensure enough bands exist for RGB visualization
        if vis_image.shape[-1] >= 4:
            rgb_image = np.stack([vis_image[:, :, 1], vis_image[:, :, 2], vis_image[:, :, 3]], axis=-1)
        else:
            rgb_image = np.stack([vis_image[:, :, 0]] * 3, axis=-1)  # Convert grayscale to RGB

        # Preprocess image for model
        input_image = preprocess_image(image)

        # Predict mask
        predicted_mask = PretrainedUNet.predict(np.expand_dims(input_image, axis=0))[0, :, :, 0]

        # Apply post-processing
        binary_mask = post_process_mask(predicted_mask)

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot RGB Image
        axes[0].imshow(rgb_image)
        axes[0].set_title("RGB Image (Bands 2,3,4)")
        axes[0].axis("off")
        
        # Plot Predicted Mask
        axes[1].imshow(binary_mask, cmap='copper')
        axes[1].set_title("Post-processed Mask")
        axes[1].axis("off")
        
        output_path = os.path.join(IMAGE_FOLDER, "visualization.png")
        plt.savefig(output_path)
        plt.close()
        
        # Save predicted mask separately for download
        mask_filename = "predicted_mask.png"
        mask_path = os.path.join(IMAGE_FOLDER, mask_filename)
        plt.imsave(mask_path, binary_mask, cmap='copper')

        return render_template("index.html", mask_image=mask_filename, visualization_image="visualization.png")

    except Exception as e:
        return render_template("index.html", error=f"Error processing image: {str(e)}")

@app.route("/download")
def download():
    mask_path = os.path.join(IMAGE_FOLDER, "predicted_mask.png")
    return send_file(mask_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
