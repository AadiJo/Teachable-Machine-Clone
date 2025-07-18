import os
import base64
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import json

app = Flask(__name__, static_folder='.')
CORS(app)

# Directory for storing training images and models
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

# Create base folders if they don't exist
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    
# Create temp folder for predictions
os.makedirs(os.path.join(UPLOAD_FOLDER, 'temp'), exist_ok=True)

# Global variables to store models
feature_extractor = None
models = {
    'knn': None,
    'svm': None,
    'neural_network': None
}

def load_feature_extractor():
    """Load MobileNetV2 for feature extraction"""
    global feature_extractor
    if feature_extractor is None:
        # Load MobileNetV2 without top layer and use it as a feature extractor
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        feature_extractor = Model(inputs=base_model.input, outputs=x)
        print("Feature extractor loaded")
    return feature_extractor

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/capture', methods=['POST'])
def capture():
    """Save captured image from webcam"""
    data = request.get_json()
    
    # Get class ID and image data
    class_id = data.get('classId')
    image_data = data.get('imageData')
    
    if not image_data or class_id is None:
        return jsonify({'error': 'Missing data'}), 400
    
    # Process base64 image
    image_data = image_data.replace('data:image/png;base64,', '')
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    
    # Create class directory if it doesn't exist
    class_dir = os.path.join(UPLOAD_FOLDER, f'class_{class_id}')
    os.makedirs(class_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(class_dir, filename)
    
    # Save image
    with open(file_path, 'wb') as f:
        f.write(image_bytes)
    
    return jsonify({'success': True, 'filePath': file_path})

def extract_features(image_path):
    """Extract features from an image using MobileNetV2"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    extractor = load_feature_extractor()
    features = extractor.predict(img)
    
    return features.flatten()

@app.route('/api/train', methods=['POST'])
def train():
    """Train models using captured images"""
    data = request.get_json()
    model_type = data.get('modelType', 'all')  # 'knn', 'svm', 'neural_network', or 'all'
    
    # Get all images from each class
    X = []  # Features
    y = []  # Labels
    
    # Collect features for each class
    class_names = {}
    for class_dir in os.listdir(UPLOAD_FOLDER):
        if class_dir.startswith('class_'):
            class_id = int(class_dir.split('_')[1])
            class_name = data.get(f'className_{class_id}', f'Class {class_id+1}')
            class_names[class_id] = class_name
            
            class_path = os.path.join(UPLOAD_FOLDER, class_dir)
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    features = extract_features(img_path)
                    X.append(features)
                    y.append(class_id)
    
    if len(X) == 0:
        return jsonify({'error': 'No training data available'}), 400
    
    X = np.array(X)
    y = np.array(y)
    
    results = {}
    
    # Train KNN classifier
    if model_type in ['knn', 'all']:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        models['knn'] = knn
        joblib.dump(knn, os.path.join(MODEL_FOLDER, 'knn_model.pkl'))
        results['knn'] = 'KNN model trained successfully'
    
    # Train SVM classifier
    if model_type in ['svm', 'all']:
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(X, y)
        models['svm'] = svm
        joblib.dump(svm, os.path.join(MODEL_FOLDER, 'svm_model.pkl'))
        results['svm'] = 'SVM model trained successfully'
    
    # Train simple neural network
    if model_type in ['neural_network', 'all']:
        num_classes = len(np.unique(y))
        
        # Simple neural network classifier
        input_dim = X.shape[1]
        nn_model = tf.keras.Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        nn_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        nn_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        models['neural_network'] = nn_model
        nn_model.save(os.path.join(MODEL_FOLDER, 'nn_model.h5'))
        results['neural_network'] = 'Neural Network model trained successfully'
    
    # Save class names
    with open(os.path.join(MODEL_FOLDER, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    
    return jsonify({
        'success': True,
        'results': results,
        'classes': class_names
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using trained models"""
    data = request.get_json()
    
    # Get image data and model type
    image_data = data.get('imageData')
    model_type = data.get('modelType', 'knn')  # Default to KNN
    
    if not image_data:
        return jsonify({'error': 'Missing image data'}), 400
    
    # Process base64 image
    image_data = image_data.replace('data:image/png;base64,', '')
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(UPLOAD_FOLDER, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save image temporarily
    temp_path = os.path.join(temp_dir, 'temp_predict.jpg')
    with open(temp_path, 'wb') as f:
        f.write(image_bytes)
    
    # Extract features
    features = extract_features(temp_path)
    features = features.reshape(1, -1)
    
    # Load class names
    class_names = {}
    try:
        with open(os.path.join(MODEL_FOLDER, 'class_names.json'), 'r') as f:
            class_names = json.load(f)
    except:
        # Use default class names if file doesn't exist
        for i in range(10):
            class_names[str(i)] = f'Class {i+1}'
    
    # Make prediction based on model type
    result = {'model': model_type, 'predictions': []}
    
    try:
        if model_type == 'knn':
            if models['knn'] is None:
                models['knn'] = joblib.load(os.path.join(MODEL_FOLDER, 'knn_model.pkl'))
            
            pred = models['knn'].predict_proba(features)[0]
            class_ids = models['knn'].classes_
            
        elif model_type == 'svm':
            if models['svm'] is None:
                models['svm'] = joblib.load(os.path.join(MODEL_FOLDER, 'svm_model.pkl'))
                
            pred = models['svm'].predict_proba(features)[0]
            class_ids = models['svm'].classes_
            
        elif model_type == 'neural_network':
            if models['neural_network'] is None:
                models['neural_network'] = load_model(os.path.join(MODEL_FOLDER, 'nn_model.h5'))
                
            pred = models['neural_network'].predict(features)[0]
            class_ids = range(len(pred))
            
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Format predictions
        for i, class_id in enumerate(class_ids):
            class_id_str = str(int(class_id))
            result['predictions'].append({
                'class_id': int(class_id),
                'class_name': class_names.get(class_id_str, f'Class {int(class_id)+1}'),
                'probability': float(pred[i])
            })
            
        # Sort by probability
        result['predictions'] = sorted(result['predictions'], 
                                     key=lambda x: x['probability'], 
                                     reverse=True)
            
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'modelAvailable': False
        }), 500

# API endpoint to get available models
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available trained models"""
    available_models = []
    
    if os.path.exists(os.path.join(MODEL_FOLDER, 'knn_model.pkl')):
        available_models.append('knn')
    
    if os.path.exists(os.path.join(MODEL_FOLDER, 'svm_model.pkl')):
        available_models.append('svm')
    
    if os.path.exists(os.path.join(MODEL_FOLDER, 'nn_model.h5')):
        available_models.append('neural_network')
    
    return jsonify({
        'success': True,
        'availableModels': available_models
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of available classes"""
    try:
        with open(os.path.join(MODEL_FOLDER, 'class_names.json'), 'r') as f:
            class_names = json.load(f)
    except:
        # Count directories in UPLOAD_FOLDER
        class_names = {}
        for class_dir in os.listdir(UPLOAD_FOLDER):
            if class_dir.startswith('class_'):
                class_id = class_dir.split('_')[1]
                class_names[class_id] = f'Class {int(class_id)+1}'
    
    return jsonify({
        'success': True,
        'classes': class_names
    })

if __name__ == '__main__':
    load_feature_extractor()  # Pre-load the feature extractor
    app.run(debug=True, port=5000)
