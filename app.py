import gradio as gr
import numpy as np
import librosa 
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model and encoder
model = load_model("model/best_emotion_cnn_model.keras")
label_encoder_classes = np.load("model/label_encoder_classes.npy", allow_pickle=True)

# Constants
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
MAX_PAD_LENGTH = 228

# Prediction function
def predict_emotion_from_audio(audio):
    if audio is None:
        return "No audio received", None, None
    
    try:
        audio_data, sr = librosa.load(audio, sr=SR)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        padded = librosa.util.fix_length(mel_spec_db, size=MAX_PAD_LENGTH, axis=1)
        input_data = np.expand_dims(np.expand_dims(padded, axis=0), axis=-1)

        predictions = model.predict(input_data)[0]
        pred_index = np.argmax(predictions)
        emotion = label_encoder_classes[pred_index]
        confidence = float(predictions[pred_index])

        # --- Spectrogram Image ---
        fig, ax = plt.subplots(figsize=(8,3))
        librosa.display.specshow(padded, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.title("Mel-Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()

        return emotion, confidence, fig
    except Exception as e:
        return f"Error: {e}", None, None
    
# Gradio Interface
interface = gr.Interface(
    fn=predict_emotion_from_audio,
    inputs=gr.Audio(type='filepath', label='Upload Audio (.wav)'),
    outputs=[
        gr.Text(label="Predicted Emotion"),
        gr.Number(label="Confidence"),
        gr.Plot(label="Mel-Spectrogram")
    ],
    title="Speech Emotion Recognizer",
    description="Upload a voice sample to detect emotion from speech. Powered by CNN + Spectrograms.",
    theme='soft'
)

if __name__ == "__main__":
    interface.launch()