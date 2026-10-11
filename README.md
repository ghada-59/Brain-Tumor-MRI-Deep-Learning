# Brain Tumor MRI Classification — Deep Learning Practice Project

An educational biomedical-imaging portfolio project for four-class brain MRI image classification using TensorFlow/Keras and MobileNetV2 transfer learning. It includes a stratified data split, training/evaluation scripts, saved aggregate metrics, plots, CLI inference, tests, and a Streamlit demonstration.

> **Safety and scope:** this is a learning project, not a medical device. It must not be used to diagnose or exclude a brain tumour or guide treatment. The model is not clinically validated, and its scores are not calibrated clinical confidence.

## Task and classes

The dataset classes are `glioma`, `meningioma`, `notumor` (no tumor), and `pituitary`. This is image-level classification only; it does not segment, outline, or localize a tumour and does not process DICOM series or 3D volumes.

## Pipeline

![Project pipeline](docs/pipeline_overview.png)

1. Read image folders from `data/Training/<class>/` and `data/Testing/<class>/`.
2. Split each training class independently into training and validation subsets using a fixed seed, ensuring each class is represented in both.
3. Resize images and train a MobileNetV2 classifier with ImageNet initialization by default.
4. Select checkpoints using validation loss.
5. Evaluate on the separate `Testing/` folder and generate aggregate metrics and plots.
6. Use the CLI or local Streamlit interface for an educational inference demonstration.

## Repository layout

- `app.py` — Streamlit demonstration
- `train.py` — training and held-out evaluation
- `predict.py` — command-line inference
- `check_dataset.py` — dataset folder check
- `src/brain_tumor_dl/data.py` — deterministic class-stratified split and dataset loading
- `src/brain_tumor_dl/model.py` — MobileNetV2 transfer-learning model
- `src/brain_tumor_dl/evaluation.py` — metrics and visualization generation
- `notebooks/brain_tumor_mri_deep_learning.ipynb` — guided notebook
- `tests/` — unit tests
- `artifacts/` — aggregate JSON metrics/configuration are tracked; trained model files and checkpoints are excluded
- `reports/` — aggregate plots and CSV reports are tracked; example images drawn from the test dataset are not published

The MRI dataset and trained model weights are intentionally excluded from this repository. The latest recorded local run used MobileNetV2 with ImageNet initialization, 224 × 224 input images, batch size 32, seed 42, and a 15% class-stratified validation split. On the separate `Testing/` folder (1,599 images), it achieved **81.93% test accuracy**, **0.8123 macro F1**, and **0.8123 weighted F1**.

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| Glioma | 0.927 | 0.664 | 0.774 | 399 |
| Meningioma | 0.739 | 0.643 | 0.687 | 400 |
| No tumor | 0.875 | 0.978 | 0.923 | 400 |
| Pituitary | 0.766 | 0.993 | 0.865 | 400 |

The main weakness is recall for glioma (66.4%) and meningioma (64.3%), so accuracy alone is not sufficient. These are dataset-specific results, not clinical performance claims. The repository tracks aggregate reports, but does not include the trained model or dataset; GitHub Actions tests code only and does not retrain the model.

## 1. Prepare the dataset

Download the [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) yourself and check its current terms. Do not commit the dataset, patient images, or trained weights.

Arrange folders as follows, with class names matching exactly:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 2. Install dependencies

Use Python 3.11 in Anaconda Prompt or PowerShell from the repository root:

```powershell
conda create --prefix D:\anaconda_envs\brain-tumor-dl python=3.11 -y
conda activate D:\anaconda_envs\brain-tumor-dl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional notebook dependencies:

```powershell
python -m pip install -r requirements-notebook.txt
jupyter lab
```

## 3. Check and train

```powershell
python check_dataset.py
python -m pytest -q
python train.py --epochs 10 --batch-size 32
```

Training may take substantial time and may download ImageNet weights. If memory is limited, reduce the batch size, for example to `--batch-size 16`. Training writes local model files under `artifacts/`, aggregate JSON metrics/configuration, and plots/CSV reports under `reports/`.

After retraining, inspect the class-wise precision, recall, F1 scores and confusion matrix. Replace the old published aggregate reports with the newly generated results only after confirming the training completed successfully. Report the actual metrics, including weaknesses; do not tune repeatedly against the held-out test set.

## 4. Predict one image and run the demo

```powershell
python predict.py "D:\path\to\an\MRI-image.jpg"
python -m streamlit run app.py
```

The Streamlit demo expects local trained model files in `artifacts/`. Its output is a model prediction for an input image, not a clinical conclusion.

## Evaluation and limitations

- Training/validation splitting is performed independently per class with a fixed random seed.
- The separate `Testing/` folder is reserved for held-out evaluation.
- Accuracy alone is insufficient; inspect macro F1, per-class recall, and the confusion matrix.
- Performance on this dataset does not establish generalization to other hospitals, scanners, protocols, or patient populations.
- Dataset provenance, label quality, patient-level separation, and licensing must be checked before making stronger claims.
- These results do not establish generalization to other hospitals, scanners, protocols, or patient populations.
- Dataset provenance, label quality, possible duplicate images, patient-level separation, and licensing should be checked before stronger claims.
- This project does not include external validation, calibration, clinical review, DICOM/3D processing, or tumour segmentation.
