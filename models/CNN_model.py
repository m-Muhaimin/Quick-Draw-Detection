import os
import numpy as np
from tensorflow import keras

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

# Reshape the images for compatibility with Conv2D layer
train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)

# Select random images from each category for further evaluation
num_images_per_category = 5
category_indices = np.unique(test_labels)
random_indices = []

for category_index in category_indices:
    indices = np.where(test_labels == category_index)[0]
    random_indices.extend(np.random.choice(indices, size=num_images_per_category, replace=False))

evaluation_images = test_images[random_indices]
evaluation_labels = test_labels[random_indices]

# Predict labels for evaluation images
predictions = model.predict(evaluation_images)
predicted_labels = np.argmax(predictions, axis=1)

# Print the true and predicted labels for evaluation images
print("Evaluation Results:")
for i in range(len(evaluation_images)):
    true_label = categories[evaluation_labels[i]]
    predicted_label = categories[predicted_labels[i]]
    print(f"Image {i+1}: True Label = {true_label}, Predicted Label = {predicted_label}")



