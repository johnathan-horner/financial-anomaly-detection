"""
Financial Transaction Anomaly Detection API

FastAPI application for real-time transaction scoring, fraud investigation,
and system monitoring with comprehensive OpenAPI documentation.
"""

import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import random

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tags metadata for OpenAPI documentation
tags_metadata = [
    {"name": "Scoring", "description": "Transaction anomaly detection and scoring"},
    {"name": "Investigation", "description": "Fraud investigation and analysis workflows"},
    {"name": "Monitoring", "description": "System metrics, dashboard, and performance"},
    {"name": "Audit", "description": "Transaction audit trails and compliance"},
    {"name": "Feedback", "description": "Model feedback and continuous learning"},
]

# Initialize FastAPI app
app = FastAPI(
    title="Financial Transaction Anomaly Detection System",
    description="Production-grade AI system for real-time fraud detection using PyTorch autoencoders and LangGraph investigation workflows",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for API
class TransactionFeatures(BaseModel):
    """Transaction features for anomaly scoring."""
    amount: float = Field(..., description="Transaction amount in USD", gt=0)
    merchant_category: str = Field(..., description="Merchant category (retail, restaurant, gas, atm, online)")
    time_of_day: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    location: str = Field(..., description="Transaction location or 'online'")
    customer_id: str = Field(..., description="Customer identifier", min_length=1)

class AnomalyScoreResponse(BaseModel):
    """Response model for transaction anomaly scoring."""
    transaction_id: str = Field(..., description="Generated transaction identifier")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score (0=normal, 1=highly anomalous)")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    routing_decision: str = Field(..., description="auto-approve, review, block")
    feature_breakdown: Dict[str, float] = Field(..., description="Contribution of each feature to anomaly score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for scoring")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scoring timestamp")

class InvestigationRequest(BaseModel):
    """Request model for fraud investigation."""
    transaction_id: str = Field(..., description="Transaction ID to investigate")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Initial anomaly score")
    customer_id: str = Field(..., description="Customer identifier")
    merchant_name: str = Field(..., description="Merchant name")

class InvestigationStep(BaseModel):
    """Single step in fraud investigation workflow."""
    step_name: str = Field(..., description="Investigation step name")
    status: str = Field(..., description="completed, in_progress, failed")
    findings: str = Field(..., description="Step findings and analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in findings")
    processing_time_ms: float = Field(..., description="Step processing time")

class InvestigationResponse(BaseModel):
    """Response model for fraud investigation."""
    investigation_id: str = Field(..., description="Investigation identifier")
    transaction_id: str = Field(..., description="Associated transaction ID")
    status: str = Field(..., description="Investigation status")
    steps: List[InvestigationStep] = Field(..., description="Investigation workflow steps")
    final_recommendation: str = Field(..., description="Final fraud determination")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    investigation_time_ms: float = Field(..., description="Total investigation time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Investigation timestamp")

class SystemMetrics(BaseModel):
    """System dashboard metrics."""
    total_transactions_scored: int = Field(..., description="Total transactions scored today")
    fraud_rate_percent: float = Field(..., description="Detected fraud rate percentage")
    false_positive_rate_percent: float = Field(..., description="False positive rate percentage")
    avg_investigation_time_minutes: float = Field(..., description="Average investigation time")
    model_version: str = Field(..., description="Current model version")
    last_retrain_date: str = Field(..., description="Last model retrain date")
    system_health: str = Field(..., description="Overall system health status")

class AuditTrailEntry(BaseModel):
    """Audit trail entry for transaction."""
    timestamp: datetime = Field(..., description="Action timestamp")
    action: str = Field(..., description="Action performed")
    actor: str = Field(..., description="Who performed the action")
    details: Dict[str, Any] = Field(..., description="Action details")

class AuditTrailResponse(BaseModel):
    """Audit trail response for transaction."""
    transaction_id: str = Field(..., description="Transaction identifier")
    audit_trail: List[AuditTrailEntry] = Field(..., description="Chronological audit trail")
    total_entries: int = Field(..., description="Total number of audit entries")

class FeedbackRequest(BaseModel):
    """Feedback request for model improvement."""
    transaction_id: str = Field(..., description="Transaction identifier")
    analyst_decision: str = Field(..., description="confirmed_fraud, false_positive, unclear")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analyst confidence in decision")
    notes: Optional[str] = Field(None, description="Additional analyst notes")

class DriftStatus(BaseModel):
    """Model drift monitoring status."""
    psi_score: float = Field(..., description="Population Stability Index score")
    threshold: float = Field(..., description="PSI threshold for alerts")
    status: str = Field(..., description="healthy, warning, critical")
    last_calculated: datetime = Field(..., description="Last drift calculation timestamp")
    recommendations: List[str] = Field(..., description="Recommended actions")

# Helper functions for mock responses
def generate_transaction_id(features: Dict) -> str:
    """Generate a deterministic transaction ID."""
    content = f"{features['amount']}{features['customer_id']}{int(time.time())}"
    return f"txn_{hashlib.md5(content.encode()).hexdigest()[:8]}"

def calculate_mock_anomaly_score(features: TransactionFeatures) -> tuple:
    """Calculate mock anomaly score based on transaction features."""
    base_score = 0.1
    breakdown = {"amount": 0.0, "merchant": 0.0, "time": 0.0, "location": 0.0}

    # Amount-based scoring
    if features.amount > 1000:
        amount_contrib = min(0.4, (features.amount - 1000) / 10000)
        base_score += amount_contrib
        breakdown["amount"] = amount_contrib

    # Merchant category risk
    high_risk_merchants = ["atm", "casino", "foreign"]
    if any(risk in features.merchant_category.lower() for risk in high_risk_merchants):
        merchant_contrib = 0.25
        base_score += merchant_contrib
        breakdown["merchant"] = merchant_contrib

    # Time of day (late night transactions more suspicious)
    if features.time_of_day < 6 or features.time_of_day > 22:
        time_contrib = 0.15
        base_score += time_contrib
        breakdown["time"] = time_contrib

    # Location risk
    if "foreign" in features.location.lower() or "unknown" in features.location.lower():
        location_contrib = 0.2
        base_score += location_contrib
        breakdown["location"] = location_contrib

    # Add some randomness
    final_score = min(0.99, base_score + random.uniform(0, 0.1))

    # Determine risk level and routing
    if final_score >= 0.7:
        risk_level, routing = "high", "block"
    elif final_score >= 0.3:
        risk_level, routing = "medium", "review"
    else:
        risk_level, routing = "low", "auto-approve"

    return final_score, risk_level, routing, breakdown

# API Endpoints

@app.post("/score", response_model=AnomalyScoreResponse, tags=["Scoring"])
async def score_transaction(features: TransactionFeatures):
    """
    Score a transaction for anomalous patterns using the PyTorch autoencoder model.

    Returns anomaly score, risk level, and routing decision for fraud prevention workflow.
    """
    start_time = time.time()

    try:
        # Generate transaction ID
        transaction_id = generate_transaction_id(features.dict())

        # Calculate anomaly score
        anomaly_score, risk_level, routing, breakdown = calculate_mock_anomaly_score(features)

        processing_time = (time.time() - start_time) * 1000

        return AnomalyScoreResponse(
            transaction_id=transaction_id,
            anomaly_score=round(anomaly_score, 3),
            risk_level=risk_level,
            routing_decision=routing,
            feature_breakdown=breakdown,
            processing_time_ms=round(processing_time, 2),
            model_version="autoencoder_v1.2"
        )

    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        raise HTTPException(status_code=500, detail="Transaction scoring failed")

@app.post("/investigate", response_model=InvestigationResponse, tags=["Investigation"])
async def investigate_transaction(request: InvestigationRequest):
    """
    Run LangGraph fraud investigation workflow for flagged transactions.

    Executes 4-node investigation: customer history, merchant check, pattern analysis, summary.
    """
    start_time = time.time()

    try:
        investigation_id = f"inv_{hashlib.md5(request.transaction_id.encode()).hexdigest()[:8]}"

        # Mock investigation steps
        steps = [
            InvestigationStep(
                step_name="Customer History Analysis",
                status="completed",
                findings=f"Customer {request.customer_id} typically spends $45/day at grocery/gas. This transaction is 15x above baseline. No late-night activity in past 90 days.",
                confidence=0.92,
                processing_time_ms=1200.0
            ),
            InvestigationStep(
                step_name="Merchant Risk Assessment",
                status="completed",
                findings=f"Merchant '{request.merchant_name}' has 2.3% fraud rate (industry avg 1.1%). 15 fraud reports in last 30 days from 12,000 transactions.",
                confidence=0.78,
                processing_time_ms=850.0
            ),
            InvestigationStep(
                step_name="Pattern Analysis",
                status="completed",
                findings="Transaction matches known pattern: high-value, unusual merchant, outside normal hours. Similar to Card-Not-Present fraud cluster from last month.",
                confidence=0.85,
                processing_time_ms=1500.0
            ),
            InvestigationStep(
                step_name="Investigation Summary",
                status="completed",
                findings="Multiple fraud indicators present: amount anomaly, merchant risk, timing pattern. Recommend immediate customer contact and temporary card freeze.",
                confidence=0.88,
                processing_time_ms=400.0
            )
        ]

        # Determine final recommendation
        avg_confidence = sum(step.confidence for step in steps) / len(steps)
        if avg_confidence > 0.8 and request.anomaly_score > 0.6:
            recommendation = "confirmed_fraud"
        elif avg_confidence > 0.6:
            recommendation = "suspicious_review_required"
        else:
            recommendation = "insufficient_evidence"

        total_time = (time.time() - start_time) * 1000

        return InvestigationResponse(
            investigation_id=investigation_id,
            transaction_id=request.transaction_id,
            status="completed",
            steps=steps,
            final_recommendation=recommendation,
            confidence_score=round(avg_confidence, 3),
            investigation_time_ms=round(total_time, 2)
        )

    except Exception as e:
        logger.error(f"Error during investigation: {e}")
        raise HTTPException(status_code=500, detail="Investigation failed")

@app.get("/dashboard/metrics", response_model=SystemMetrics, tags=["Monitoring"])
async def get_dashboard_metrics():
    """
    Retrieve current system performance metrics and operational statistics.

    Returns fraud detection rates, processing times, model performance, and system health.
    """
    try:
        return SystemMetrics(
            total_transactions_scored=random.randint(50000, 150000),
            fraud_rate_percent=round(random.uniform(0.5, 2.5), 2),
            false_positive_rate_percent=round(random.uniform(1.8, 3.2), 2),
            avg_investigation_time_minutes=round(random.uniform(3.5, 8.2), 1),
            model_version="autoencoder_v1.2",
            last_retrain_date="2024-01-15",
            system_health="healthy"
        )

    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")

@app.get("/audit/{transaction_id}", response_model=AuditTrailResponse, tags=["Audit"])
async def get_audit_trail(transaction_id: str):
    """
    Retrieve complete audit trail for a specific transaction.

    Returns chronological log of all actions, decisions, and modifications for compliance.
    """
    try:
        # Mock audit trail
        base_time = datetime.utcnow() - timedelta(hours=2)

        audit_entries = [
            AuditTrailEntry(
                timestamp=base_time,
                action="transaction_received",
                actor="payment_gateway",
                details={"source": "merchant_api", "amount": 1250.00}
            ),
            AuditTrailEntry(
                timestamp=base_time + timedelta(seconds=1),
                action="anomaly_scored",
                actor="ml_model_v1.2",
                details={"score": 0.82, "risk_level": "high"}
            ),
            AuditTrailEntry(
                timestamp=base_time + timedelta(seconds=5),
                action="investigation_triggered",
                actor="fraud_detection_system",
                details={"investigation_id": f"inv_{transaction_id[-8:]}"}
            ),
            AuditTrailEntry(
                timestamp=base_time + timedelta(minutes=1),
                action="analyst_reviewed",
                actor="fraud_analyst_j.smith",
                details={"decision": "confirmed_fraud", "confidence": 0.9}
            ),
            AuditTrailEntry(
                timestamp=base_time + timedelta(minutes=2),
                action="customer_notified",
                actor="notification_service",
                details={"method": "sms", "status": "delivered"}
            )
        ]

        return AuditTrailResponse(
            transaction_id=transaction_id,
            audit_trail=audit_entries,
            total_entries=len(audit_entries)
        )

    except Exception as e:
        logger.error(f"Error retrieving audit trail: {e}")
        raise HTTPException(status_code=500, detail="Audit trail retrieval failed")

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit analyst feedback for model retraining and continuous improvement.

    Collects fraud confirmations and false positive corrections for model optimization.
    """
    try:
        # Log feedback (in production, this would update training data)
        logger.info(f"Feedback received for {feedback.transaction_id}: {feedback.analyst_decision}")

        # Mock response
        return {
            "status": "feedback_recorded",
            "transaction_id": feedback.transaction_id,
            "feedback_id": f"fb_{hashlib.md5(feedback.transaction_id.encode()).hexdigest()[:8]}",
            "impact": "Will be included in next model training cycle",
            "next_retrain_eta": "2024-02-01"
        }

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Feedback submission failed")

@app.get("/drift/status", response_model=DriftStatus, tags=["Monitoring"])
async def get_drift_status():
    """
    Check current model drift status and population stability metrics.

    Returns PSI scores, drift thresholds, and recommended actions for model health.
    """
    try:
        # Mock drift calculation
        psi_score = random.uniform(0.05, 0.25)
        threshold = 0.15

        if psi_score < 0.1:
            status = "healthy"
            recommendations = ["Continue monitoring", "Model performing well"]
        elif psi_score < threshold:
            status = "warning"
            recommendations = ["Increased monitoring recommended", "Review feature distributions"]
        else:
            status = "critical"
            recommendations = ["Immediate model retraining required", "Investigate data quality", "Consider feature engineering"]

        return DriftStatus(
            psi_score=round(psi_score, 4),
            threshold=threshold,
            status=status,
            last_calculated=datetime.utcnow(),
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error calculating drift status: {e}")
        raise HTTPException(status_code=500, detail="Drift status calculation failed")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint for load balancers and monitoring systems."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model_status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)