# X-Ray Image Classification

Deep learning project for classifying chest X-ray images into 20 classes using a pretrained DenseNet121 model with PyTorch.

## Competition

This project was developed for the following Kaggle competition:

**26T1 DL Gen AI NPPE 1**

[Kaggle Competition](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1)

## Overview

The project uses a transfer-learning approach for chest X-ray image classification.

The main workflow is:

1. Load the training and test datasets.
2. Apply image preprocessing and augmentation.
3. Split the training data into training and validation sets.
4. Fine-tune a pretrained DenseNet121 model.
5. Evaluate the model on the validation set.
6. Generate predictions for the test images.
7. Create the submission file in the required format.

## Model Architecture

The project uses **DenseNet121 pretrained on ImageNet**.

The original classifier is replaced with the following classification head:

```text
DenseNet121
    ↓
Linear
    ↓
ReLU
    ↓
Dropout (0.5)
    ↓
Linear
    ↓
20 Classes
```

## Image Preprocessing

Images are processed at a resolution of **224 × 224**.

### Training Augmentation

- Resize to 224 × 224
- Random horizontal flip
- Random rotation up to 10 degrees
- Convert to tensor
- ImageNet normalization

### Validation/Test

- Resize to 224 × 224
- Convert to tensor
- ImageNet normalization

## Training Configuration

| Parameter | Value |
|---|---|
| Model | DenseNet121 |
| Pretrained weights | ImageNet |
| Number of classes | 20 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | CrossEntropyLoss |
| Initial epochs | 5 |
| Train/Validation split | 80/20 |
| Random state | 2021 |
| Dropout | 0.5 |
| Device | CUDA if available, otherwise CPU |

## Prediction

For test-time inference, the trained model produces probabilities for the 20 classes. The notebook applies the prediction adjustment used in the project before selecting the final class.

The final predictions are converted into one-hot encoded class labels for submission.

## Installation

Clone the repository:

```bash
git clone https://github.com/Tanishq-lalani/X-Ray-Image-classification.git
cd X-Ray-Image-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The main Python dependencies are:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- Scikit-learn

## Running the Project

Open the notebook using Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Run the notebook cells in order to:

- prepare the dataset
- train the model
- validate the model
- generate test predictions
- create `submission.csv`

## Dataset

The dataset is provided through the Kaggle competition.

You can access it here:

[Kaggle Competition Dataset](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1)

The notebook expects the competition dataset to contain training and test CSV files along with the corresponding X-ray images.

## Output

The notebook generates:

```text
submission.csv
```

This file contains the predictions in the format required for the Kaggle competition.

## Disclaimer

This project is developed for educational and competition purposes. The model predictions should not be used as a substitute for professional medical diagnosis.

## Author

**Tanishq Lalani**

GitHub: [Tanishq-lalani](https://github.com/Tanishq-lalani)
