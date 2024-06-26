import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input, concatenate
from skimage.feature import daisy
from skimage.color import rgb2gray

# Set the numpy pseudo-random generator at a fixed value for repeatability
np.random.seed(1000)

# Ensure the correct paths for both NG and OK images
train_ng_image_directory = r'C:\python\deep\datasets\train\NG'
train_ok_image_directory = r'C:\python\deep\datasets\train\OK'
test_ng_image_directory = r'C:\python\deep\datasets\test\NG'
test_ok_image_directory = r'C:\python\deep\datasets\test\OK'

# Initialize dataset and labels
train_dataset = []
train_daisy_features = []
train_label = []
test_dataset = []
test_daisy_features = []
test_label = []

# Supported image formats
supported_formats = ('.jpg', '.jpeg', '.png')

def load_images_from_directory(directory, label_value):
    dataset = []
    daisy_features_list = []
    label = []
    images = os.listdir(directory)
    for image_name in images:
        if image_name.lower().endswith(supported_formats):
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((64, 64))  # Resize to 64x64
                dataset.append(np.array(image))
                # Convert image to grayscale and extract DAISY features
                gray_image = rgb2gray(np.array(image))
                daisy_features = daisy(gray_image, step=4, radius=15, rings=3, histograms=8, orientations=8, visualize=False)
                daisy_features_flattened = daisy_features.flatten()
                daisy_features_list.append(daisy_features_flattened)
                label.append(label_value)
            else:
                print(f"Warning: {image_path} could not be read and was skipped.")
        else:
            print(f"Warning: {image_name} is not a supported image format and was skipped.")
    return dataset, daisy_features_list, label

# Load training NG (defective) images
ng_train_dataset, ng_train_daisy, ng_train_label = load_images_from_directory(train_ng_image_directory, 0)
train_dataset.extend(ng_train_dataset)
train_daisy_features.extend(ng_train_daisy)
train_label.extend(ng_train_label)

# Load training OK (non-defective) images
ok_train_dataset, ok_train_daisy, ok_train_label = load_images_from_directory(train_ok_image_directory, 1)
train_dataset.extend(ok_train_dataset)
train_daisy_features.extend(ok_train_daisy)
train_label.extend(ok_train_label)

# Load test NG (defective) images
ng_test_dataset, ng_test_daisy, ng_test_label = load_images_from_directory(test_ng_image_directory, 0)
test_dataset.extend(ng_test_dataset)
test_daisy_features.extend(ng_test_daisy)
test_label.extend(ng_test_label)

# Load test OK (non-defective) images
ok_test_dataset, ok_test_daisy, ok_test_label = load_images_from_directory(test_ok_image_directory, 1)
test_dataset.extend(ok_test_dataset)
test_daisy_features.extend(ok_test_daisy)
test_label.extend(ok_test_label)

# Convert to numpy arrays
train_dataset = np.array(train_dataset)
train_daisy_features = np.array(train_daisy_features)
train_label = np.array(train_label)
test_dataset = np.array(test_dataset)
test_daisy_features = np.array(test_daisy_features)
test_label = np.array(test_label)

# Normalize the image data
train_dataset = train_dataset / 255.0
test_dataset = test_dataset / 255.0

# Check shapes and types
print(f"train_dataset type: {type(train_dataset)}, shape: {train_dataset.shape}, dtype: {train_dataset.dtype}")
print(f"train_daisy_features type: {type(train_daisy_features)}, shape: {train_daisy_features.shape}, dtype: {train_daisy_features.dtype}")
print(f"train_label type: {type(train_label)}, shape: {train_label.shape}, dtype: {train_label.dtype}")
print(f"test_dataset type: {type(test_dataset)}, shape: {test_dataset.shape}, dtype: {test_dataset.dtype}")
print(f"test_daisy_features type: {type(test_daisy_features)}, shape: {test_daisy_features.shape}, dtype: {test_daisy_features.dtype}")
print(f"test_label type: {type(test_label)}, shape: {test_label.shape}, dtype: {test_label.dtype}")

# Check for NaN values
print(f"NaNs in train_dataset: {np.sum(np.isnan(train_dataset))}")
print(f"NaNs in train_daisy_features: {np.sum(np.isnan(train_daisy_features))}")
print(f"NaNs in train_label: {np.sum(np.isnan(train_label))}")
print(f"NaNs in test_dataset: {np.sum(np.isnan(test_dataset))}")
print(f"NaNs in test_daisy_features: {np.sum(np.isnan(test_daisy_features))}")
print(f"NaNs in test_label: {np.sum(np.isnan(test_label))}")

# Ensure no None values (not applicable here but for completeness)
print(f"None values in train_dataset: {np.sum(train_dataset == None)}")
print(f"None values in train_daisy_features: {np.sum(train_daisy_features == None)}")
print(f"None values in train_label: {np.sum(train_label == None)}")
print(f"None values in test_dataset: {np.sum(test_dataset == None)}")
print(f"None values in test_daisy_features: {np.sum(test_daisy_features == None)}")
print(f"None values in test_label: {np.sum(test_label == None)}")

# Build the CNN model for image data
INPUT_SHAPE = (64, 64, 3)
image_input = Input(shape=INPUT_SHAPE)

conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(image_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = BatchNormalization(axis=-1)(pool1)
drop1 = Dropout(rate=0.2)(norm1)
conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(drop1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = BatchNormalization(axis=-1)(pool2)
drop2 = Dropout(rate=0.2)(norm2)

flat = Flatten()(drop2)

hidden1 = Dense(512, activation='relu')(flat)
norm3 = BatchNormalization(axis=-1)(hidden1)
drop3 = Dropout(rate=0.2)(norm3)
hidden2 = Dense(256, activation='relu')(drop3)
norm4 = BatchNormalization(axis=-1)(hidden2)
drop4 = Dropout(rate=0.2)(norm4)

# DAISY feature input
daisy_input = Input(shape=(train_daisy_features.shape[1],))
daisy_dense = Dense(128, activation='relu')(daisy_input)
daisy_norm = BatchNormalization()(daisy_dense)
daisy_drop = Dropout(rate=0.2)(daisy_norm)

# Concatenate CNN and DAISY features
combined = concatenate([drop4, daisy_drop])

combined_hidden = Dense(128, activation='relu')(combined)
combined_norm = BatchNormalization()(combined_hidden)
combined_drop = Dropout(rate=0.2)(combined_norm)

out = Dense(1, activation='sigmoid')(combined_drop)

# Multi-input model
model = Model(inputs=[image_input, daisy_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the model
history = model.fit([train_dataset, train_daisy_features], train_label, batch_size=64, epochs=1000, validation_split=0.1, shuffle=True)

# Calculate accuracy on the test data
print("Test Accuracy: {:.2f}%".format(model.evaluate([test_dataset, test_daisy_features], test_label)[1]*100))

# Plot the performance
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy']) + 1
epoch_list = list(range(1, max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

plt.show()

# Save the model
model.save('binary_classification_cnn_daisy_64x64.h5')
