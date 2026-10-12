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

The MRI dataset and trained model weights are intentionally excluded from this repository. The tracked aggregate reports describe a previous run until you retrain using the corrected split; do not treat those older scores as the current validated result.

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

## Results — Corrected Dataset Split

Two experiments were evaluated on the same held-out test set of 1,578 images from the deduplicated dataset. Exact duplicate-image auditing was performed, but patient-level independence could not be established.

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Test accuracy | 80.67% | **85.93%** |
| Macro-F1 | 79.61% | **85.40%** |
| Weighted-F1 | 79.66% | **85.45%** |

The fine-tuned model improved test accuracy by 5.26 percentage points and macro-F1 by 5.79 percentage points compared with the baseline.

### Fine-tuned model — Per-class performance

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| Glioma | 95.62% | 74.35% | 83.65% | 382 |
| Meningioma | 83.23% | 70.20% | 76.16% | 396 |
| No tumor | 84.47% | 99.25% | 91.26% | 400 |
| Pituitary | 83.23% | 99.25% | 90.54% | 400 |

### Evaluation protocol and limitations

- The training data was deduplicated using exact-image checks. This does not guarantee that images from the same patient are separated across training and test sets.
- The test set contains 1,578 images and is used for held-out evaluation. Repeatedly tuning the model against test results can lead to optimistic estimates.
- Performance is uneven across classes. In particular, the fine-tuned model's recall for meningioma is 70.20%, so some meningioma images are missed.
- Results on this dataset do not establish generalization to other hospitals, scanners, imaging protocols, or patient populations.
- Dataset provenance, label quality, licensing, and patient-level separation require further verification.
- These results are for an educational portfolio project. The model is not clinically validated, is not a medical device, and must not be used to diagnose or exclude tumors or guide treatment.
- The separate 40-image prediction grid is illustrative only and is not a substitute for the complete test-set evaluation.
