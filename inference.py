import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# === MAKING PREDICTION AND INTERPRETING RESULTS ===

# 1. Load the preprocessed data and labels 
try:
    X_train = np.load('saved_data/X_train.npy') # X_train and y_train needed for input_shape etc.
    y_train = np.load('saved_data/y_train.npy')
    X_test = np.load('saved_data/X_test.npy')
    y_test = np.load('saved_data/y_test.npy')
    label_encoder_classes = np.load('model/label_encoder_classes.npy', allow_pickle=True)

    print("All preprocessed data and label encoder classes loaded successfully.")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Emotion classes: {label_encoder_classes}")

except FileNotFoundError:
    print("Error: One or more .npy files not found.")
    exit()

# 2. Load the best model
model_path = 'best_emotion_cnn_model.keras'
try:
    model = load_model(model_path)
    print(f"\nSuccessfully loaded the best model from '{model_path}'.")
except Exception as e:
    print(f"Error: Could not load the model from '{model_path}'.")
    print(f"Error details: {e}")
    exit()

# 3. Making predictions

# Get predictions (probabilities for each class)
y_pred_probs = model.predict(X_test, verbose=1)

# Convert probabilities to class labels (index of the highest probability)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded true labels back to class labels
y_true_classes = np.argmax(y_test, axis=1)

# Convert numerical class labels back to original emotion names for readability
y_pred_emotion_names = label_encoder_classes[y_pred_classes]
y_true_emotion_names = label_encoder_classes[y_true_classes]

print("\n--- Classification Report ---")
# Generate and print the classification report
# target_names are the actual labels for our classes
print(classification_report(y_true_emotion_names, y_pred_emotion_names, target_names=label_encoder_classes))

print("\n--- Confusion Matrix ---")
# Generate the confusion matrix
cm = confusion_matrix(y_true_emotion_names, y_pred_emotion_names, labels=label_encoder_classes)
print(cm)

# 4. Visualize the Confusion Matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder_classes, yticklabels=label_encoder_classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nModel evaluation complete. Review the report and confusion matrix for detailed performance insights.")
