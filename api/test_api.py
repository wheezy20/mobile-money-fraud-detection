"""
Smoke tests for the fraud detection API.

Run while uvicorn is serving:
    python api/test_api.py

This is not a formal test suite — it's a reproducible demonstration
that the API correctly classifies a known-fraud and known-legitimate
transaction, and rejects malformed inputs.
"""

import json
import requests

BASE_URL = "http://localhost:8000"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# Health check
section("Test 1: Health check")
r = requests.get(f"{BASE_URL}/health")
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# Known fraud signature
section("Test 2: Textbook fraud signature")
fraud_tx = {
    "step": 5,
    "type": "TRANSFER",
    "amount": 181000.0,
    "oldbalanceOrg": 181000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}
print(f"Request: {json.dumps(fraud_tx, indent=2)}")
r = requests.post(f"{BASE_URL}/predict", json=fraud_tx)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
assert r.json()["is_fraud"] == True, "Expected fraud classification"

# Known legitimate
section("Test 3: Normal legitimate transaction")
legit_tx = {
    "step": 100,
    "type": "CASH_OUT",
    "amount": 5000.0,
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 45000.0,
    "oldbalanceDest": 200000.0,
    "newbalanceDest": 205000.0,
}
print(f"Request: {json.dumps(legit_tx, indent=2)}")
r = requests.post(f"{BASE_URL}/predict", json=legit_tx)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
assert r.json()["is_fraud"] == False, "Expected legitimate classification"

# Validation rejection
section("Test 4: Invalid type rejected by Pydantic")
bad_tx = {
    "step": 5,
    "type": "PAYMENT",  # not in the Literal[CASH_OUT, TRANSFER]
    "amount": 100.0,
    "oldbalanceOrg": 0.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
}
print(f"Request: {json.dumps(bad_tx, indent=2)}")
r = requests.post(f"{BASE_URL}/predict", json=bad_tx)
print(f"Status: {r.status_code} (expected 422)")
print(f"Response: {r.json()}")
assert r.status_code == 422, "Expected validation rejection"

print(f"\n{'=' * 60}")
print("All API tests passed.")
print(f"{'=' * 60}")