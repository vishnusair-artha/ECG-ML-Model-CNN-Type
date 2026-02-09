# ECG ML Model (CNN)

This project builds and deploys a 1D CNN model trained on open-source ECG datasets from PhysioNet, including:

- MIT-BIH Arrhythmia Database (`mitdb`)
- Chapman-Shaoxing and Ningbo ECG Dataset (`chapman_shaoxing`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download + Preprocess

Use the helper script to download a small sample of records from both datasets and create preprocessed datasets.

```bash
python scripts/download_datasets.py \
  --mit-records 100 101 102 \
  --chapman-records A0001 A0002
```

The script outputs compressed datasets in `data/processed/`.

> Note: PhysioNet datasets may require credentials depending on your environment. Configure `WFDB` access
> as needed if downloads fail.

## Train the Model

```bash
python scripts/train_model.py \
  --datasets data/processed/mit_bih.npz data/processed/chapman.npz \
  --epochs 10 \
  --batch-size 64 \
  --model-name ecg_cnn
```

Model artifacts are saved to `models/`.

## Deploy (FastAPI)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Front-end MVP

Open `http://localhost:8000` to use the investor demo interface. Upload CSV, TXT, or PDF ECG files to
receive a prediction, confidence score, and interpretation.

> If the model artifacts are not present in `models/`, the service falls back to demo predictions so the
> UI can still be showcased.

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, 0.05, 0.0]}'
```

## Evaluate Accuracy

Use the evaluation script to compute accuracy and a classification report on your processed datasets.

```bash
python scripts/evaluate_model.py \
  --datasets data/processed/mit_bih.npz data/processed/chapman.npz \
  --model-name ecg_cnn
```

## Project Structure

```
app/              # FastAPI deployment
scripts/          # Download + training scripts
src/ecg_ml/       # Data + modeling code
```
