# Teachable Machine Clone

A clone of Google's Teachable Machine that uses webcam input to train various machine learning models. This project combines a web frontend with a Python backend for machine learning.

## Features

- Capture webcam images for training data
- Train multiple types of machine learning models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- Real-time predictions with confidence scores
- Visual confidence indicators for each class

## Requirements

### Frontend

- Node.js
- npm

### Backend

- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - flask
  - flask-cors
  - numpy
  - opencv-python
  - tensorflow
  - scikit-learn
  - joblib

## Installation

1. Clone this repository
2. Install Node.js dependencies:
   ```
   npm install
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Start both servers separately

1. Start the Python backend server:

   ```
   python app.py
   ```

   This will run the Flask server on port 5000.

2. In a separate terminal, start the Node.js frontend server:

   ```
   npm start
   ```

   This will run the Express server on port 3000.

3. Open your browser and go to: http://localhost:3000

### Option 2: Access the Python server directly

1. Start the Python backend server:

   ```
   python app.py
   ```

2. Open your browser and go to: http://localhost:5000

## How to Use

1. Allow webcam access when prompted
2. Collect samples for each class by clicking the "Add Sample" buttons
3. Click "Train & Predict" to train the models
4. Once trained, you can select different model types to compare their performance
5. Point your webcam at objects to see real-time predictions

## Project Structure

- `app.py` - Python backend for model training and prediction
- `server.js` - Node.js Express server for serving the frontend
- `app.js` - Frontend JavaScript for webcam access and UI interactions
- `index.html` - Main webpage
- `requirements.txt` - Python dependencies
- `package.json` - Node.js dependencies

### Directories

- `/uploads` - Stores the training images captured from webcam
  - `/uploads/class_0` through `/uploads/class_9` - Images for each class
  - `/uploads/temp` - Temporary files for prediction
- `/models` - Stores trained ML models
  - `/models/class_0` through `/models/class_9` - Model files for each class

## Models

### K-Nearest Neighbors (KNN)

A simple and effective algorithm that classifies based on feature similarity.

### Support Vector Machine (SVM)

A powerful classifier that finds optimal boundaries between classes.

### Neural Network

A deep learning model that can capture complex patterns in data.

## Todo
- Pose model
- Sound model


