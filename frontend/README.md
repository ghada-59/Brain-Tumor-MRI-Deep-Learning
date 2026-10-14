# NeuroVision React interface

This optional React/Vite UI uses the existing shared preprocessing function and the trained MobileNetV2 model via a local FastAPI adapter.

## Setup

Install the optional API dependencies from the repository root:

~~~bash
python -m pip install -r requirements-react-api.txt
~~~

Start the API in terminal 1:

~~~bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
~~~

Start the frontend in terminal 2:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite, normally http://127.0.0.1:5173/.

## Model requirement

The trained model weights are intentionally not tracked in this repository. The API expects artifacts/brain_tumor_mobilenetv2.keras and artifacts/class_names.json. Train the model locally using the documented repository workflow before attempting inference. If the weights are missing, the UI reports the limitation instead of fabricating scores.

## Scope

This is image-level educational classification only. It does not segment or localize a tumour, is not clinically validated, and must not be used to diagnose or exclude a tumour or guide treatment. Model scores are not calibrated clinical confidence.
