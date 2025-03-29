from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from model import TensorFlowImageClassifier
from preprocessing import preprocess_image, apply_image_enhancement
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)
# Configuration - using absolute paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'artifacts', 'VGG_net_kaggle.keras')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'artifacts', 'labels.json')

# Initialize classifier
try:
    classifier = TensorFlowImageClassifier(MODEL_PATH, LABELS_PATH)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[X] Error loading model: {e}")
    classifier = None

# Serve frontend files
@app.route('/')
def serve_index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/classify', methods=['POST'])
def classify_image():
    # Debugging: Print received files and form data
    print("Received files:", request.files)
    print("Received form:", request.form)
    
    # Check if file was properly uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    image_file = request.files['image']
    
    # Check if file has a name
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    if not ('.' in image_file.filename and 
            image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'Allowed file types are: png, jpg, jpeg, webp'}), 400

    try:
        # Get enhancement flag (default to False if not provided)
        enhance = request.form.get('enhance', 'false').lower() == 'true'
        
        # Process image
        img_array = preprocess_image(image_file)
        if enhance:
            img_array = apply_image_enhancement(img_array)
            
        # Get predictions
        class_id, probabilities = classifier.predict(img_array)
        
        # Ensure probabilities are valid
        if isinstance(probabilities, list):
            probabilities = [float(p) for p in probabilities]
        else:
            return jsonify({'error': 'Invalid prediction format'}), 500
            
        # Get indices of top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Create a properly formatted response
        response_data = {
            'class_id': int(class_id),
            'class_name': classifier.labels[str(class_id)] if isinstance(classifier.labels, dict) else classifier.labels[class_id],
            'confidence': float(probabilities[class_id]),
            'top_predictions': [
                {
                    'class': classifier.labels[str(i)] if isinstance(classifier.labels, dict) else classifier.labels[i],
                    'probability': float(probabilities[i])
                }
                for i in top_indices
            ]
        }
        
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    host = '0.0.0.0'  # Accessible from all network interfaces
    port = 5000
    print(f"\n * Server running at: http://{socket.gethostbyname(socket.gethostname())}:{port}\n")
    app.run(host=host, port=port, debug=True)