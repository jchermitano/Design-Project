from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model.pkl')

# Function to process audio data and generate feature vector
def process_audio_data(audio_data):
    # Here you would convert the audio data to a suitable format for analysis
    # For this example, let's just return a placeholder feature vector
    feature_vector = np.random.randn(1, 2)
    return feature_vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Get the audio data from the form
    audio_data = request.form['audio_data']

    # Process the audio data to generate feature vector
    feature_vector = process_audio_data(audio_data)

    # Make prediction using the loaded model
    prediction = model.predict(feature_vector)

    # Assuming prediction is binary (-1 for anomaly, 1 for normal), prepare result message
    if prediction == -1:
        result = "Anomaly detected!"
    else:
        result = "No anomaly detected."

    # Return result as JSON
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
