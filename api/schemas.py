"""
Request and response schemas for the mobile money fraud detection API.

Pydantic models defined here are used by FastAPI for automatic validation
of incoming JSON requests. If a request is missing a required field, has
the wrong type, or fails any defined constraint, FastAPI rejects it
before reaching the prediction code.
"""

from typing import Literal
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    A single mobile money transaction submitted for fraud scoring.
    
    The fields mirror the raw PaySim columns. The API performs all
    feature engineering internally so callers can submit transactions
    in their natural form rather than the engineered feature space.
    """
    
    step: int = Field(
        ..., 
        ge=1, 
        le=10000,
        description="Hour of the simulation (1-744 for the original 30-day window)",
        examples=[5],
    )
    type: Literal["CASH_OUT", "TRANSFER"] = Field(
        ...,
        description="Transaction type. Only CASH_OUT and TRANSFER are scored "
                    "for fraud, as established by the EDA in Section B.",
        examples=["TRANSFER"],
    )
    amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount",
        examples=[181000.0],
    )
    oldbalanceOrg: float = Field(
        ...,
        ge=0,
        description="Sender's account balance before the transaction",
        examples=[181000.0],
    )
    newbalanceOrig: float = Field(
        ...,
        ge=0,
        description="Sender's account balance after the transaction",
        examples=[0.0],
    )
    oldbalanceDest: float = Field(
        ...,
        ge=0,
        description="Recipient's account balance before the transaction",
        examples=[0.0],
    )
    newbalanceDest: float = Field(
        ...,
        ge=0,
        description="Recipient's account balance after the transaction",
        examples=[0.0],
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "step": 5,
                    "type": "TRANSFER",
                    "amount": 181000.0,
                    "oldbalanceOrg": 181000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """The fraud probability and classification for one transaction."""
    
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability that the transaction is fraudulent",
    )
    is_fraud: bool = Field(
        ...,
        description="Binary classification at the default threshold of 0.5",
    )
    threshold: float = Field(
        default=0.5,
        description="Decision threshold used for the classification",
    )
    model_version: str = Field(
        ...,
        description="Identifier of the model that produced this prediction",
    )


class HealthResponse(BaseModel):
    """Service health status."""
    
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    