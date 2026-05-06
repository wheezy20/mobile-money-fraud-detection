# Fraud Detection API

REST endpoint for the tuned Random Forest fraud detection model.

## Run locally

From the project root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health` — readiness check
- `POST /predict` — submit a transaction, get a fraud probability
- `GET /docs` — interactive API documentation (Swagger UI)
- `GET /redoc` — alternative API documentation

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 5,
    "type": "TRANSFER",
    "amount": 181000.0,
    "oldbalanceOrg": 181000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'
```

## Model

- Source: `models/tuned_rf.pkl`
- Version: `tuned_rf_v1`
- See Section D of the project report for training methodology.