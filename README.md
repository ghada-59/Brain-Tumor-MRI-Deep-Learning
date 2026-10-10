# Brain Tumor MRI Classification — Deep Learning Practice Project

A reproducible **medical-imaging learning project** that trains a four-class MRI image classifier with **TensorFlow/Keras, CNNs, and MobileNetV2 transfer learning**. It includes a dataset checker, training/evaluation pipeline, saved metrics and plots, a command-line prediction script, and a small Streamlit demo.

> **Important:** this is an educational portfolio project, not a medical device. It must not be used to diagnose, exclude, or guide treatment for a brain tumor. The dataset consists of labelled images and does not represent a clinically validated workflow.

## What the model predicts

The intended dataset classes are:

- `glioma`
- `meningioma`
- `notumor` (no tumor)
- `pituitary`

The model outputs one class per image. **This repository implements image classification, not tumor segmentation, tumor outlining, or localization.** The softmax score is a model score; it is not calibrated clinical confidence.

## Pipeline

![Project pipeline](docs/pipeline_overview.png)

1. Read the labelled `Training/` and held-out `Testing/` image folders.
2. Split `Training/` into training and validation subsets.
3. Resize images to 224 × 224 and apply modest data augmentation during training.
4. Use a MobileNetV2 feature extractor pretrained on ImageNet (the default) with a new four-class classification head.
5. Train with early stopping and learning-rate reduction callbacks.
6. Evaluate once on the separate `Testing/` folder and generate metrics and visualizations.
7. Run a local Streamlit demo or a command-line prediction on a single image.

## Repository layout

```text
Brain-Tumor-MRI-Deep-Learning/
├── app.py                         # Streamlit image-upload demo
├── train.py                       # Training + held-out evaluation
├── predict.py                     # CLI prediction for one image
├── check_dataset.py                # Dataset folder sanity check
├── requirements.txt
├── requirements-notebook.txt        # Optional Jupyter environment
├── data/                           # Local dataset only; not committed
│   ├── Training/<class>/...
│   └── Testing/<class>/...
├── notebooks/brain_tumor_mri_deep_learning.ipynb  # Guided notebook
├── src/brain_tumor_dl/
│   ├── data.py                     # Dataset loading and folder checks
│   ├── model.py                    # MobileNetV2 transfer-learning model
│   └── evaluation.py               # Metrics and plot generation
├── artifacts/                      # Local model + metric JSON; not committed
├── reports/                        # Plots and CSV reports from your own run
├── docs/pipeline_overview.png
└── tests/
```

## 1. Get the dataset

This project is designed for the public [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Download the ZIP from Kaggle yourself and check the dataset's current terms before using or redistributing it. **Do not add the dataset, patient images, or trained model weights to this GitHub repository.**

Arrange the extracted files so the folders are directly under `data/`:

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

Folder names must match exactly between `Training/` and `Testing/`. The archive may extract into an extra parent directory; move `Training/` and `Testing/` into `data/` if necessary.

## 2. Create an environment and install dependencies

Python 3.11 is a good starting point for this TensorFlow setup. From a terminal opened in the repository root:

### Windows PowerShell / Anaconda Prompt

```powershell
conda create --prefix D:\anaconda_envs\brain-tumor-dl python=3.11 -y
conda activate D:\anaconda_envs\brain-tumor-dl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not use Conda, create and activate a virtual environment with Python 3.11, then run the final two commands. The first `pip install` may take a while because TensorFlow is a large dependency.

### Optional: use the guided notebook

To run the notebook in JupyterLab, install its optional dependencies in the same environment:

```powershell
python -m pip install -r requirements-notebook.txt
jupyter lab
```

Open `notebooks/brain_tumor_mri_deep_learning.ipynb` and run the cells from top to bottom. It follows the same model/data/evaluation modules as the scripts.

## 3. Check the dataset

```powershell
python check_dataset.py
```

The check reports the number of recognized image files in every class folder. It checks folder structure, not image correctness, label quality, or dataset provenance.

## 4. Train and evaluate

From the repository root:

```powershell
python train.py --epochs 10 --batch-size 32
```

The default configuration uses ImageNet-pretrained MobileNetV2 weights, which may need to download once. If you intentionally want to train from random initialization instead, use:

```powershell
python train.py --epochs 15 --batch-size 16 --weights none
```

The `none` option avoids downloading pretrained weights but is not expected to give the same results. Training speed depends on your computer and available hardware. To reduce memory use, try `--batch-size 16` or `--batch-size 8`.

### What is generated after training?

- `artifacts/brain_tumor_mobilenetv2.keras` — the trained Keras model
- `artifacts/best_model.keras` — best checkpoint selected by validation loss
- `artifacts/class_names.json` — class order used by the model
- `artifacts/test_metrics.json` — held-out test accuracy and F1 scores
- `artifacts/classification_report.json` — precision, recall, F1 and support
- `artifacts/training_config.json` — settings from the run
- `reports/training_curves.png` — training/validation accuracy and loss
- `reports/confusion_matrix.png` — held-out test confusion matrix
- `reports/class_metrics.png` — per-class precision, recall and F1
- `reports/prediction_examples.png` — test-image prediction examples
- `reports/classification_report.csv` and `reports/training_history.csv` — table outputs

**No performance numbers are claimed in this README before a local run has been completed.** Report the scores produced by your own test run, including weaknesses or classes that the model confuses. Do not select hyperparameters repeatedly against the test set; use validation data for model choices.

## 5. Predict one image

Once training finishes:

```powershell
python predict.py "D:\path\to\an\MRI-image.jpg"
```

The output lists the predicted dataset class and the top class scores. This is a demo of model inference, not a clinical conclusion.

## 6. Launch the demo interface

```powershell
streamlit run app.py
```

A local browser window should open. Upload a JPG or PNG image and click **Run model prediction**. The model and class-name file must exist in `artifacts/` first.

## 7. Run the lightweight helper tests

```powershell
python -m pytest -q
```

The tests cover dataset-folder image counting; they are not a clinical validation suite and do not prove model accuracy.

## Evaluation notes

- `Training/` is split into training and validation subsets using a fixed seed.
- `Testing/` remains separate for the final held-out evaluation.
- Accuracy is reported alongside macro F1 and weighted F1 so class-level failures are easier to notice.
- The class-wise report and confusion matrix should be checked before interpreting the overall score.
- These files are ordinary labelled image files, not DICOM series or 3D volumes. This implementation does not process DICOM metadata or perform tumor segmentation.
- A strong score on this dataset does not establish generalization to other hospitals, scanners, acquisition protocols, or patient populations.
