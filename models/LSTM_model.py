import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Directory containing the dataset files
data_directory = 'data/cnn_dataset2'

# List of category names
categories = ['ambulance', 'apple', 'bear', 'bicycle', 'bird', 'bus', 'cat', 'foot', 'owl', 'pig']

# Load the dataset for each category
dataset = []
labels = []
for i, category in enumerate(categories):
    category_file = os.path.join(data_directory, f"{category}.npy")
    category_data = np.load(category_file, allow_pickle=True)
    dataset.extend(category_data)
    labels.extend([i] * len(category_data))  # Assign label for the category

# Pad or truncate the sequences to a fixed length
max_sequence_length = 100  # Choose an appropriate value
dataset = pad_sequences(dataset, maxlen=max_sequence_length, padding='post', truncating='post')

# Convert the dataset and labels to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Normalize the pixel values of images between 0 and 1
dataset = dataset / 255.0

# Split the dataset into training and testing sets
split_index = int(0.9 * len(dataset))  # Split at 90% for training
train_images = dataset[:split_index]
train_labels = labels[:split_index]
test_images = dataset[split_index:]
test_labels = labels[split_index:]

# Convert the dataset to TensorFlow tensors
train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)

# Define the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(max_sequence_length, 1)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)

# Save the model as an h5 file
model.save('LSTM_model.h5')