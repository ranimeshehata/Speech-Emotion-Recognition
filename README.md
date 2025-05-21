# Speech Emotion Recognition

This project implements a Speech Emotion Recognition (SER) system using deep learning. It extracts audio features from the CREMA-D dataset and trains both 1D and 2D Convolutional Neural Networks (CNNs) to classify emotions from speech.

## Features

- **1D Feature Space:** Zero Crossing Rate (ZCR) and Energy
- **2D Feature Space:** Mel Spectrogram
- **Data Augmentation:** Applied only to 1D features (ZCR, Energy)
- **Model Architectures:** 1D CNN for 1D features, 2D CNN for Mel Spectrograms
- **Evaluation:** Accuracy, F1 Score, and Confusion Matrix

## Project Structure

```
Speech-Emotion-Recognition/
│
├── Speech_Emotion_Recognition.ipynb   # Main Jupyter Notebook
├── model_1d.h5                        # Saved 1D CNN model (after training)
├── model_2d.h5                        # Saved 2D CNN model (after training)
└── README.md                          # This file
```

## How to Run

1. **Install Requirements**

   Make sure you have Python 3.8+ and install the required packages:
   ```bash
   pip install numpy pandas librosa matplotlib scikit-learn tensorflow keras tqdm seaborn
   ```

2. **Dataset**

   - Download the [CREMA-D dataset](https://zenodo.org/record/3816443) and place the `.wav` files in the `Crema` folder inside your project directory.

3. **Run the Notebook**

   - Open `Speech_Emotion_Recognition.ipynb` in Jupyter Notebook or VS Code.
   - Run all cells to extract features, train models, and evaluate performance.

## Evaluation Accuracies
 
- The **1D CNN** test accuracy `Test accuracy 1D: 56.05%`
- The **2D CNN** test accuracy `Test accuracy 2D: 87.66%`


## Key Sections in the Notebook

- **Feature Extraction:** Extracts ZCR, Energy, and Mel Spectrogram from each audio file.
- **Data Augmentation:** Augments only the 1D features (not the Mel Spectrogram).
- **Data Preparation:** Pads and stacks features for model input.
- **Model Training:** Trains both 1D and 2D CNNs.
- **Evaluation:** Plots accuracy/loss curves, confusion matrices, and prints F1 scores.

## Results

- **Confusion Matrices:** Visualizes model performance and highlights the most confusing emotion pairs.
- **Most Confusing Classes:** Automatically identified and printed after evaluation.

## Notes

- Data augmentation is **not** applied to the 2D Mel Spectrogram features.
- The notebook is designed to run efficiently, but processing the full dataset may require significant memory.
