# 🎤 Speech Emotion Recognition using CNN & Mel-Spectrograms

> A deep learning project to classify human emotions (like happy, sad, angry, fearful) from voice signals using Mel-Spectrograms and Convolutional Neural Networks.

This project combines audio feature extraction, CNN-based training, data augmentation, and optional web demo using Gradio.

---

## 📁 Datasets Used

Combined from three open-source emotion speech datasets:

- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)

Metadata extracted using a custom script `data_metadata_extractor.py`, resulting in a clean CSV with:
- File path
- Emotion label
- Gender
- Actor ID
- Dataset origin

---

## 🧠 Model Architecture

- **Input**: Mel-Spectrogram (grayscale image)
- **Model**: Convolutional Neural Network (CNN)

```text
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
      ↓
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
      ↓
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
      ↓
Flatten → Dense(256) → Dropout → Dense(Num_Classes, softmax)
````

---

## 🛠️ Tools & Libraries

* Python, NumPy, Pandas
* TensorFlow / Keras
* Librosa (for Mel-spectrograms)
* Scikit-learn (label encoding, metrics)
* Matplotlib, Seaborn (evaluation plots)
* Gradio (for optional UI)

---

## 📊 Evaluation

**Test Accuracy**: \~Varied per run
**Note**: Model shows strong training accuracy but overfitting tendencies due to limited compute during training. Validation accuracy can be improved with more balanced data and augmentation.

> ⚠️ Work in progress to improve generalization and deploy real-time demo.

---

## 📦 Project Structure

```
├── metadata/                ← Parsed dataset metadata (CSV)
├── saved_data/              ← Processed features/labels (.npy)
├── model/                   ← Saved model & encoder classes
├── utils/                   ← Preprocessing & helper scripts
├── train.py                 ← Model training pipeline
├── inference.py             ← Load & predict from audio files
├── app.py                   ← Gradio-based prediction UI (optional)
└── README.md
```

---

## ✅ Features Implemented

* ✅ Multi-dataset integration (RAVDESS, TESS, CREMA-D)
* ✅ Mel-spectrogram-based feature extraction
* ✅ Audio preprocessing & padding
* ✅ Label encoding + one-hot transformation
* ✅ CNN architecture with dropout, batchnorm & callbacks
* ✅ Real-time prediction module & evaluation

---

## 🚀 Future Enhancements

* [ ] Add pitch/time-based audio augmentation
* [ ] Improve class balance for edge cases
* [ ] Refactor inference for real-time mic input
* [ ] Deploy Gradio app for public usage
* [ ] Publish on HuggingFace Spaces

---


---

## 🙋‍♂️ About Me

I'm **Neeraj**, a student from India passionate about machine learning and computer vision.

📧 Email: [www.asneeraj@gmail.com](mailto:asneeraj@gmail.com)
📌 This project is a part of my AI/ML journey — feedback and forks are welcome!

---

## ⭐ Star This Repo!

If you found this project helpful or inspiring, a ⭐ would mean a lot and help me grow.
Thanks for visiting! 😊

```

