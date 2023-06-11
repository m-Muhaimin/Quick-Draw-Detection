import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the pre-trained models
cnn_model = keras.models.load_model('models/CNN_model.h5')
lstm_model = keras.models.load_model('models/LSTM_model.h5')

# List of category names
categories = ['ambulance', 'apple', 'bear', 'bicycle', 'bird', 'bus', 'cat', 'foot', 'owl', 'pig']

# Maximum sequence length for LSTM model
max_sequence_length = 100


# Define routes
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/classify', methods=['POST'])
def classify():
    # Get the selected model from the form
    model_type = request.form.get('model')

    # Get the uploaded image file
    image = request.files['image']

    # Preprocess the image
    img_array = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0

    # Reshape the image according to the selected model
    if model_type == 'cnn':
        img = np.reshape(img, (1, 28, 28, 1))
        prediction = cnn_model.predict(img)
    elif model_type == 'lstm':
        img = pad_sequences([img], maxlen=max_sequence_length, padding='post', truncating='post')
        img = np.reshape(img, (img.shape[0], max_sequence_length, 1))
        prediction = lstm_model.predict(img)
    else:
        error_message = 'Invalid model selection.'
        return render_template('error.html', error_message=error_message)

    predicted_label = categories[np.argmax(prediction)]

    return render_template('result.html', predicted_label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
