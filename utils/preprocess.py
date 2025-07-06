import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Mel Spectogram Extractor Function
def extract_mel_spectrogram(file_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, 
                                                          n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
# Feature Padding Function
def pad_or_truncate_spec(spec, max_len, constant_val=-80.0):
    return librosa.util.fix_length(spec, size=max_len, axis=1, constant_values=constant_val)

# Function to Encode Labels
def encode_labels(label_array):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(label_array)
    return y_encoded, encoder

def one_hot_encode(y_encoded):
    y_encoded = y_encoded.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(y_encoded), encoder

# Augmentation Function
def augment_spectrogram(mel_spec_db, apply_noise=True, apply_mask=True):
    augmented = mel_spec_db.copy()

    if apply_noise:
        noise_amp = 0.01*np.random.rand()
        augmented += noise_amp * np.random.normal(size=augmented.shape)

    if apply_mask:
        # Time Masking
        num_frames = augmented.shape[1]
        t_start = np.random.randint(0, num_frames // 2)
        t_end = t_start + np.random.randint(num_frames // 10)
        augmented[:, t_start:t_end] = np.min(augmented)

        # Frequency Masking
        num_mels = augmented.shape[0]
        f_start = np.random.randint(0, num_mels // 2)
        f_end = f_start + np.random.randint(num_mels // 10)
        augmented[f_start:f_end, :] = np.min(augmented)

    return augmented


