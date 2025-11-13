import os
import cv2
import numpy as np

# Load images function with memory optimizations
def load_images_from_combined_folder(folder_path, img_size=(256, 256)):
    satellite_images = []
    mask_images = []
    files = os.listdir(folder_path)
    sat_files = [f for f in files if '_sat' in f]

    for sat_file in sat_files:
        # Extract base name like "20876"
        base_name = sat_file.replace('_sat.jpg', '')
        mask_file = f"{base_name}_mask.png"
        sat_path = os.path.join(folder_path, sat_file)
        mask_path = os.path.join(folder_path, mask_file)

        if os.path.exists(mask_path):
            # Load and resize satellite image
            sat_img = cv2.imread(sat_path)
            sat_img = cv2.resize(sat_img, img_size)
            satellite_images.append(sat_img)

            # Load and resize mask image
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, img_size)
            mask_images.append(mask_img)
        else:
            print(f"Mask not found for: {sat_file}")

    # Convert to numpy array and normalize
    X = (np.array(satellite_images, dtype=np.float32)) / 255.0
    y = (np.array(mask_images, dtype=np.float32)) / 255.0
    y = np.round(y).astype(np.uint8)

    return X, y

# Paths for Sentinel and PALSAR
sentinel_folder = r"C:\Users\PADMAPRIYA\Downloads\train-20250415T115310Z-001\train\sentinel"
palsar_folder = r"C:\Users\PADMAPRIYA\Downloads\train-20250415T115310Z-001\train\palsar"

# Load images for both Sentinel and PALSAR
X_sentinel, y_sentinel = load_images_from_combined_folder(sentinel_folder)
X_palsar, y_palsar = load_images_from_combined_folder(palsar_folder)

# Combine Sentinel and PALSAR data
X_combined = np.concatenate([X_sentinel, X_palsar], axis=0)
y_combined = np.concatenate([y_sentinel, y_palsar], axis=0)

# Shuffle the data
perm = np.random.permutation(len(X_combined))
X_combined = X_combined[perm]
y_combined = y_combined[perm]

# Output shapes
print(f"Total combined satellite images: {X_combined.shape}")
print(f"Total combined masks: {y_combined.shape}")
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming X_combined, y_combined are already prepared

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Model definition with upsampling layers to ensure output matches input size
model = models.Sequential([
    layers.InputLayer(input_shape=(256, 256, 3)),  # Input layer for satellite images
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Upsampling to increase the spatial dimensions
    layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'),

    # Output layer: pixel-wise binary classification
    layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')  # Output layer with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary to check architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save("oil_spill_model.h5")

# Optionally, you can also plot training & validation accuracy/loss graphs here

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
