# ===== IMPORTING NECESSARY LIBRARIES =====
import numpy as np     
import pandas as pd     
import librosa    
import librosa.display    
import sklearn     
import tensorflow as tf    
import matplotlib.pyplot as plt    
import seaborn as sns    
import soundfile as sf    
import os     
from tqdm import tqdm  
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from utils.preprocess import extract_mel_spectrogram, pad_or_truncate_spec, augment_spectrogram
from utils.preprocess import encode_labels, one_hot_encode

# === FEATURE EXTRACTION ===
USE_AUGMENTATION = True

# 1. Load metadata
try:
    df = pd.read_csv('metadata/extracted_features.csv')
    print(f"Loaded metadata: {df.shape[0]} files")
except Exception as e:
    print(f"Failed to load metadata: {e}")
    exit()

# 2. Feature Extraction Loop
features = []
labels = []
lengths = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
    path = row['filepath']
    label = row['emotion']

    mel_spec = extract_mel_spectrogram(path)
    if mel_spec is None:
        continue

    if USE_AUGMENTATION and np.random.rand() < 0.3:
        mel_spec = augment_spectrogram(mel_spec)

    lengths.append(mel_spec.shape[1])
    features.append(mel_spec)
    labels.append(label)

# 3. Padding (after max length computed)
MAX_PAD_LENGTH = max(lengths)
X = np.array([pad_or_truncate_spec(m, MAX_PAD_LENGTH) for m in features])
y = np.array(labels)

print(f"\n Features extracted: {X.shape}, Labels: {y.shape}")

# --- Optional: Visualize a Mel-spectrogram ---
if len(X) > 0:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X[0], sr=22050, x_axis='time', y_axis='mel',
                             fmax=22050/2, hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-spectrogram for "{y[0]}" emotion')
    plt.tight_layout()
    plt.show()

# Save the Extracted Features and Labels (Optional)
np.save('saved_data/X_features.npy', X)
np.save('saved_data/y_labels.npy', y)

print("\nFeatures (X_features.npy) and labels (y_labels.npy) saved successfully.")

# === Data Preprocessing ===

# 1. Reshape features for CNN input
X = np.expand_dims(X, axis=-1)
print(f"Reshaped features: {X.shape}")

# 2. Encode string labels to integers
y_int, label_encoder = encode_labels(y)
print(f"Encoded string labels to integers. Classes: {label_encoder.classes_}")

# 3. Convert to one-hot
y_onehot, _ = one_hot_encode(y_int)
print(f"One-hot encoded labels: {y_onehot.shape}")

# Save classes for inference (optional)
np.save("model/label_encoder_classes.npy", label_encoder.classes_)

# === TRAINING AND TESTING SPLIT ===

# 1. First Split: Training + Temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)


# Second Split: Validation + Test (split temp equally)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print Shapes 
print(f"\n Data split completed:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# Optional: Save Split Files (for later use/reuse)
np.save('saved_data/X_train.npy', X_train)
np.save('saved_data/y_train.npy', y_train)
np.save('saved_data/X_val.npy', X_val)
np.save('saved_data/y_val.npy', y_val)
np.save('saved_data/X_test.npy', X_test)
np.save('saved_data/y_test.npy', y_test)

print("\nProcessed data (X_train, y_train, X_val, y_val, X_test, y_test) saved as .npy files.")
print("Label encoder classes saved as 'label_encoder_classes.npy'.")

# === AUGMENTATION ON SPLIT DATA ===

def apply_augmentation_to_batch(X_batch, probability=0.3):
    augmented = []
    for i in range(len(X_batch)):
        mel = X_batch[i, :, :, 0]  # Remove channel
        if np.random.rand() < probability:
            mel = augment_spectrogram(mel)
        mel = np.expand_dims(mel, axis=-1)  # Restore channel
        augmented.append(mel)
    return np.array(augmented)

if USE_AUGMENTATION:
    print("Applying augmentation to training data...")
    X_train = apply_augmentation_to_batch(X_train)
    print("Augmentation applied successfully.")

# === BUILDING MODEL ARCHITECTURE ===

# We can load all the saved preprocessed data from the saved_data directory. 

# 1. Get the input shape and number of classes from the preprocessed data
input_shape = X_train.shape[1:] # (n_mels, max_pad_length, 1)
num_classes = y_train.shape[1]   # Number of one-hot encoded classes

print(f"\nModel Input Shape: {input_shape}")
print(f"Number of Output Classes: {num_classes}")

# 2. Build the CNN Model
def build_cnn_model(input_shape, num_classes):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Third Convolutional Block (Optional, depending on complexity)
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Flattening and Dense Layers
    model.add(Flatten()) # Flattens the 2D feature maps into a 1D vector

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

model = build_cnn_model(input_shape, num_classes)

# Print the model summary to see the layers and number of parameters
print("\n---CNN Model Summary---")
model.summary()

# === COMPILING AND TRAINING THE MODEL ===

# 1. Compile the model created
optimizer = Adam(learning_rate=0.0003)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("\nModel compiled with Adam optimizer, Categorical Crossentropy loss, and Accuracy metric.")

# 2. Callbacks for Efficient Training 

# Early Stopping: Stop training if validation loss doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', # Monitor validation loss
                               patience=20,        # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity.
                               verbose=1)

# Reduce Learning Rate on Plateau: Reduce learning rate when validation loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', # Monitor validation loss
                              factor=0.5,         # Factor by which the learning rate will be reduced (e.g., new_lr = lr * 0.5)
                              patience=10,        # Number of epochs with no improvement after which learning rate will be reduced
                              min_lr=0.00001,     # Lower bound on the learning rate
                              verbose=1)

# Model Checkpoint: Save the best model based on validation accuracy
model_checkpoint = ModelCheckpoint(filepath='model/best_emotion_cnn_model.keras', 
                                   monitor='val_accuracy', # Monitor validation accuracy
                                   save_best_only=True,    # Only save the best model
                                   mode='max',             # Maximize validation accuracy
                                   verbose=1)

# List of all callbacks to use
callbacks = [early_stopping, reduce_lr, model_checkpoint]
print("\nCallbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) configured.")

# 3. Training the model
EPOCHS = 75
BATCH_SIZE = 32

print(f"\nStarting model training for up to {EPOCHS} epochs with batch size {BATCH_SIZE}...")
print("Training progress will be displayed below:")

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=1)

print("\nModel training complete!")

# 4. Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# 5. Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# 6. Saving the trained model (Best model from checkpoint)
try:
    # Attempt to load the best model saved by ModelCheckpoint
    best_model = tf.keras.models.load_model('best_emotion_cnn_model.keras')
    print("\nBest trained model loaded from 'best_emotion_cnn_model.keras'.")
except Exception as e:
    print(f"\nCould not load the best model. Ensure 'best_emotion_cnn_model.keras' was saved correctly. Error: {e}")
    best_model = model # Fallback to the last trained model if checkpoint fails

# === MODEL EVALUATION ===

# 1. Load the best model
model_path = 'model/best_emotion_cnn_model.keras'
try:
    model = load_model(model_path)
    print(f"\nSuccessfully loaded the best model from '{model_path}'.")
except Exception as e:
    print(f"Error: Could not load the model from '{model_path}'.")
    print(f"Please ensure the model was saved correctly in Step 6. Error details: {e}")
    exit()

# Evaluating Performance on the Test Set 
print("\n--- Evaluating Model Performance on Test Set ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

