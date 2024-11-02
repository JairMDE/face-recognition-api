import joblib
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from deepface import DeepFace  # Import DeepFace for embedding extraction
import io
import os

app = Flask(__name__)

# Direct paths to the model and encoder
MODEL_PATH = "./models/svm_model.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

# Load the model and encoder directly from the filesystem
try:
    # Load the SVM model using joblib
    svm_model = joblib.load(MODEL_PATH)
    print("SVM model loaded successfully from local path.")

    # Load the label encoder
    label_encoder = joblib.load(ENCODER_PATH)
    print("Label encoder loaded successfully from local path.")
except Exception as e:
    print(f"Error loading model or encoder: {str(e)}")

# Define the embedding extraction function
def generate_embedding(image):
    # Convert the PIL image to a format DeepFace can process directly
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    
    # Generate the embedding
    embedding = DeepFace.represent(img_path=img_byte_arr, model_name="Facenet", 
                                   detector_backend="mtcnn", enforce_detection=False)[0]['embedding']
    return np.array(embedding)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract the features from the input data
    features = data['features']

    # Make the prediction
    prediction = svm_model.predict([features])
    # Transform the predicted label back to the original class
    label = label_encoder.inverse_transform(prediction)

    return jsonify({'prediction': label[0]})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream)

    # Generate the embedding using DeepFace
    embedding = generate_embedding(image)

    # Make the prediction using the embedding
    prediction = svm_model.predict([embedding])
    label = label_encoder.inverse_transform(prediction)

    return jsonify({'prediction': label[0]})

if __name__ == '__main__':
    # Use Render's dynamically assigned port
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
