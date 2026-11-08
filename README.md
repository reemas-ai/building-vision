# Building Vision — Surface Crack Detection

## Overview

Building Vision is a computer vision project that uses deep learning to classify images of concrete surfaces into two categories:

- **Crack:** A visible surface crack is detected.
- **No Crack:** No surface crack is detected by the model.

The project demonstrates transfer learning, image preprocessing, dataset validation, model evaluation, and image classification using PyTorch.

**Note:** This is an educational machine learning project, not a certified structural safety inspection system.

## Technologies

- Python
- PyTorch and Torchvision
- ResNet50
- OpenCV and Pillow
- Scikit-learn
- Matplotlib
- KaggleHub

## Dataset

Dataset: [Surface Crack Detection — Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

The project uses images from two categories: Positive (Crack) and Negative (No Crack).

An initial subset of 4,000 images was selected, containing 2,000 images per class.

Exact duplicate files were identified using SHA-256 hashes and removed before splitting the data.

| Dataset stage | Images |
|---|---:|
| Initial subset | 4,000 |
| After duplicate removal | 3,991 |
| Training | 3,192 |
| Validation | 399 |
| Testing | 400 |

A SHA-256 comparison found no byte-identical files shared between the training, validation, and testing splits.

This does not rule out visually similar images or other forms of data leakage.

## Model Architecture

The project uses ResNet50 with pretrained ImageNet weights.

The original backbone parameters are frozen, and the final fully connected layer is replaced with a two-class classification layer.

**Training configuration:**

| Parameter | Value |
|---|---|
| Architecture | ResNet50 |
| Pretrained weights | ImageNet |
| Output classes | 2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 10 |
| Image size | 224 × 224 |

Training augmentation includes random horizontal flips, vertical flips, and rotations. Validation and testing use deterministic image preprocessing.

## Model Evaluation

After training for 10 epochs, the following results were obtained:

| Metric | Result |
|---|---:|
| Training Loss | 0.0076 |
| Validation Loss | 0.0085 |
| Validation Accuracy | 99.75% |
| Test Accuracy | 99.75% |
| Crack Recall (Test) | 100% |

### Test Confusion Matrix

| Actual / Predicted | No Crack | Crack |
|---|---:|---:|
| No Crack | 199 | 1 |
| Crack | 0 | 200 |

The model correctly classified 399 out of 400 test images.

The test set contained one false positive and zero false negatives.

These results describe performance on the selected dataset and should not be interpreted as proof of equivalent performance in real-world building inspections.

## Image Prediction

The project includes a `predict_image()` function that processes an input image and returns one of the following predictions:

- `Crack Detected`
- `No Crack Detected`

The model uses the same normalization and resizing procedure during inference as during testing.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/reemas-ai/building-vision.git
cd building-vision
```

Install the required Python dependencies and run:

```bash
python crack_detection.py
```

The script downloads the dataset through KaggleHub, prepares the data, trains the model, evaluates it, and saves model weights as `crack_model.pth`.

**Current implementation note:** The script also attempts to predict several images referenced by relative file paths. Those images must be present in the working directory for the final inference calls to succeed.

## Limitations

- The evaluation uses a subset of one public dataset.
- High test accuracy does not guarantee performance on photographs from different environments.
- Exact duplicate removal does not detect every near-duplicate image.
- Real-world image quality, lighting, camera angle, and surface texture may affect predictions.
- The current implementation does not establish calibrated confidence estimates.
- Predictions must not replace professional structural inspections.

## Future Improvements

Potential improvements include:

- Testing against independent image sources.
- Evaluating robustness under changing environmental conditions.
- Adding systematic error analysis.
- Packaging inference separately from training.
- Improving reproducibility through automated tests and dependency management.

## Author

Developed as an individual AI and Data Science portfolio project.

GitHub: [reemas-ai](https://github.com/reemas-ai)