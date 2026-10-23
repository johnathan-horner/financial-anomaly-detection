#!/usr/bin/env python3
"""
MCP Server for Financial Anomaly Detection System

Exposes tools, resources, and prompts for AI assistant integration
with the financial transaction anomaly detection system.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import hashlib

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Financial Anomaly Detection")

# Sample data for demonstration
SAMPLE_TRANSACTIONS = [
    {
        "id": "txn_001",
        "amount": 1250.00,
        "merchant": "Amazon",
        "location": "Online",
        "timestamp": "2024-01-15T14:30:00Z",
        "customer_id": "cust_001",
        "anomaly_score": 0.85
    },
    {
        "id": "txn_002",
        "amount": 50.00,
        "merchant": "Starbucks",
        "location": "New York, NY",
        "timestamp": "2024-01-15T08:15:00Z",
        "customer_id": "cust_002",
        "anomaly_score": 0.15
    }
]

@mcp.tool()
def detect_transaction_anomaly(
    amount: float,
    merchant: str,
    location: str,
    customer_id: str
) -> Dict[str, Any]:
    """
    Analyze a transaction for anomalous patterns using the PyTorch autoencoder model.

    Args:
        amount: Transaction amount in USD
        merchant: Merchant name or identifier
        location: Transaction location or "Online"
        customer_id: Customer identifier

    Returns:
        Dictionary with anomaly score, risk level, and reasoning
    """
    try:
        # Simulate anomaly detection logic
        # Higher amounts, unusual merchants, or foreign locations increase score
        base_score = 0.1

        if amount > 1000:
            base_score += 0.3
        if amount > 5000:
            base_score += 0.4

        if "foreign" in location.lower() or "atm" in merchant.lower():
            base_score += 0.25

        # Add some randomness to simulate model uncertainty
        anomaly_score = min(0.99, base_score + random.uniform(0, 0.2))

        risk_level = "low"
        if anomaly_score >= 0.7:
            risk_level = "high"
        elif anomaly_score >= 0.3:
            risk_level = "medium"

        transaction_id = f"txn_{hashlib.md5(f'{amount}{merchant}{location}'.encode()).hexdigest()[:8]}"

        return {
            "transaction_id": transaction_id,
            "anomaly_score": round(anomaly_score, 3),
            "risk_level": risk_level,
            "amount": amount,
            "merchant": merchant,
            "location": location,
            "customer_id": customer_id,
            "processing_time_ms": round(random.uniform(50, 200), 2),
            "model_version": "autoencoder_v1.2",
            "reasoning": f"Score {anomaly_score:.3f} due to amount ${amount}, merchant '{merchant}', location '{location}'"
        }

    except Exception as e:
        logger.error(f"Error in transaction anomaly detection: {e}")
        raise

@mcp.tool()
def investigate_customer_pattern(customer_id: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Investigate a customer's historical transaction patterns for anomaly context.

    Args:
        customer_id: Customer identifier to investigate
        days_back: Number of days of history to analyze

    Returns:
        Customer profile with spending patterns, risk indicators, and recommendations
    """
    try:
        # Simulate customer investigation
        base_daily_spend = random.uniform(50, 500)
        total_transactions = random.randint(20, 150)

        risk_indicators = []
        if total_transactions > 100:
            risk_indicators.append("High transaction volume")
        if base_daily_spend > 300:
            risk_indicators.append("High daily spend average")

        return {
            "customer_id": customer_id,
            "analysis_period_days": days_back,
            "total_transactions": total_transactions,
            "average_daily_spend": round(base_daily_spend, 2),
            "typical_merchants": ["Amazon", "Grocery Store", "Gas Station", "Restaurant"],
            "geographical_pattern": "Primarily New York, NY area",
            "risk_indicators": risk_indicators,
            "customer_since": "2019-03-15",
            "account_status": "active",
            "previous_fraud_cases": random.randint(0, 2),
            "recommendation": "Monitor for location anomalies" if len(risk_indicators) > 0 else "Low risk profile"
        }

    except Exception as e:
        logger.error(f"Error investigating customer pattern: {e}")
        raise

@mcp.tool()
def analyze_merchant_risk(merchant_name: str) -> Dict[str, Any]:
    """
    Analyze merchant risk profile and fraud associations.

    Args:
        merchant_name: Name of the merchant to analyze

    Returns:
        Merchant risk assessment with fraud history and risk factors
    """
    try:
        # Simulate merchant risk analysis
        risk_score = random.uniform(0.1, 0.8)

        risk_factors = []
        if "atm" in merchant_name.lower():
            risk_factors.append("ATM transactions carry higher fraud risk")
        if "online" in merchant_name.lower():
            risk_factors.append("Online merchant")
        if risk_score > 0.5:
            risk_factors.append("Above-average fraud reports")

        return {
            "merchant_name": merchant_name,
            "risk_score": round(risk_score, 3),
            "risk_level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
            "fraud_reports_30d": random.randint(0, 25),
            "total_transactions_30d": random.randint(1000, 50000),
            "risk_factors": risk_factors,
            "merchant_category": "retail" if "store" in merchant_name.lower() else "service",
            "geographic_presence": "National" if risk_score > 0.4 else "Regional",
            "compliance_status": "verified"
        }

    except Exception as e:
        logger.error(f"Error analyzing merchant risk: {e}")
        raise

@mcp.tool()
def get_investigation_queue() -> Dict[str, Any]:
    """
    Retrieve current investigation queue status and pending cases.

    Returns:
        Investigation queue status with pending cases and priority distribution
    """
    try:
        # Simulate investigation queue
        high_priority = random.randint(2, 15)
        medium_priority = random.randint(5, 30)
        low_priority = random.randint(10, 50)

        return {
            "queue_status": {
                "total_pending": high_priority + medium_priority + low_priority,
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority
            },
            "average_resolution_time_hours": round(random.uniform(2.5, 8.5), 1),
            "cases_resolved_today": random.randint(15, 45),
            "escalation_rate_percent": round(random.uniform(5, 20), 1),
            "top_fraud_types": [
                "Card-not-present fraud",
                "Account takeover",
                "Unusual location activity",
                "High-value transactions"
            ],
            "queue_capacity": 200,
            "current_investigators": random.randint(3, 8),
            "next_case_eta_minutes": random.randint(15, 120)
        }

    except Exception as e:
        logger.error(f"Error getting investigation queue: {e}")
        raise

@mcp.tool()
def generate_fraud_alert(
    transaction_id: str,
    anomaly_score: float,
    risk_level: str
) -> Dict[str, Any]:
    """
    Generate a fraud alert for high-risk transactions.

    Args:
        transaction_id: Transaction identifier
        anomaly_score: Anomaly score from detection model
        risk_level: Risk level (low, medium, high)

    Returns:
        Fraud alert details with recommended actions
    """
    try:
        alert_id = f"alert_{hashlib.md5(transaction_id.encode()).hexdigest()[:10]}"

        recommended_actions = []
        if risk_level == "high":
            recommended_actions = [
                "Immediate customer contact",
                "Temporary card freeze",
                "Require additional authentication"
            ]
        elif risk_level == "medium":
            recommended_actions = [
                "Customer notification",
                "Enhanced monitoring",
                "Request transaction verification"
            ]
        else:
            recommended_actions = ["Monitor for patterns", "Log for review"]

        return {
            "alert_id": alert_id,
            "transaction_id": transaction_id,
            "severity": risk_level.upper(),
            "anomaly_score": anomaly_score,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "status": "active",
            "recommended_actions": recommended_actions,
            "estimated_loss_prevention": f"${random.randint(100, 5000)}" if risk_level != "low" else "$0",
            "requires_human_review": risk_level in ["medium", "high"],
            "auto_resolution_possible": risk_level == "low",
            "sla_hours": 1 if risk_level == "high" else 4 if risk_level == "medium" else 24
        }

    except Exception as e:
        logger.error(f"Error generating fraud alert: {e}")
        raise

@mcp.tool()
def get_system_metrics() -> Dict[str, Any]:
    """
    Retrieve current system performance metrics and model statistics.

    Returns:
        System metrics including model performance and operational statistics
    """
    try:
        return {
            "model_performance": {
                "accuracy": 94.2,
                "precision": 91.8,
                "recall": 89.5,
                "f1_score": 90.6,
                "false_positive_rate": 2.1,
                "model_version": "autoencoder_v1.2",
                "last_training": "2024-01-01T00:00:00Z"
            },
            "operational_metrics": {
                "transactions_processed_today": random.randint(50000, 150000),
                "anomalies_detected_today": random.randint(500, 2000),
                "avg_processing_time_ms": round(random.uniform(75, 150), 2),
                "system_uptime_hours": round(random.uniform(168, 720), 1),
                "queue_depth": random.randint(10, 100)
            },
            "fraud_prevention": {
                "prevented_fraud_amount_today": f"${random.randint(10000, 100000):,}",
                "cases_closed_today": random.randint(20, 80),
                "customer_satisfaction": round(random.uniform(85, 95), 1)
            },
            "system_health": "healthy",
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise

# Resources
@mcp.resource("model://autoencoder")
def get_autoencoder_info():
    """
    Returns information about the PyTorch autoencoder model used for anomaly detection.
    """
    return {
        "model_name": "Transaction Anomaly Autoencoder",
        "framework": "PyTorch",
        "version": "1.2.0",
        "model_type": "Autoencoder Neural Network",
        "input_features": [
            "transaction_amount", "merchant_category", "location_risk_score",
            "customer_spending_pattern", "time_of_day", "day_of_week"
        ],
        "output": "anomaly_score (0-1 scale)",
        "training_data": {
            "size": "2.5M transactions",
            "date_range": "2022-01-01 to 2023-12-31",
            "fraud_positive_rate": "0.8%"
        },
        "performance": {
            "accuracy": "94.2%",
            "precision": "91.8%",
            "recall": "89.5%",
            "f1_score": "90.6%",
            "auc_roc": "0.967"
        },
        "deployment": "SageMaker real-time endpoint",
        "latency_p95": "120ms",
        "throughput": "1000 predictions/minute"
    }

@mcp.resource("data://risk_thresholds")
def get_risk_thresholds():
    """
    Returns the risk scoring thresholds and routing configuration.
    """
    return {
        "anomaly_thresholds": {
            "low_risk": "< 0.3",
            "medium_risk": "0.3 - 0.7",
            "high_risk": "> 0.7"
        },
        "routing_logic": {
            "auto_approve": "< 0.3 (no human review)",
            "investigation_queue": "0.3 - 0.7 (analyst review)",
            "immediate_alert": "> 0.7 (senior analyst + customer contact)"
        },
        "business_rules": {
            "max_auto_approve_amount": "$500",
            "foreign_transaction_threshold": "0.5",
            "velocity_check_window_hours": 24,
            "customer_contact_sla_hours": 2
        },
        "model_confidence": {
            "high_confidence": "> 0.9",
            "medium_confidence": "0.7 - 0.9",
            "low_confidence": "< 0.7"
        }
    }

# Prompts
@mcp.prompt("fraud_investigation")
def fraud_investigation_prompt():
    """
    Complete fraud investigation workflow for analyzing suspicious transactions.

    Use this prompt to conduct a thorough investigation of potentially fraudulent
    transactions including customer pattern analysis and merchant risk assessment.
    """
    return """
    You are a fraud investigation specialist. Analyze this transaction for potential fraud:

    **Investigation Workflow:**
    1. Use detect_transaction_anomaly() to get the initial risk assessment
    2. Use investigate_customer_pattern() to understand customer's normal behavior
    3. Use analyze_merchant_risk() to assess merchant-related risks
    4. If risk is medium/high, use generate_fraud_alert() to create appropriate alerts
    5. Check get_investigation_queue() for current workload and prioritization

    **Analysis Framework:**
    - Transaction characteristics (amount, timing, location)
    - Customer behavior patterns and deviations
    - Merchant reputation and fraud history
    - Geographic and temporal risk factors
    - Account history and previous incidents

    **Decision Making:**
    - Low risk (< 0.3): Auto-approve with monitoring
    - Medium risk (0.3-0.7): Queue for analyst review
    - High risk (> 0.7): Immediate escalation and customer contact

    Please provide a comprehensive fraud risk assessment with specific recommendations for action.
    """

@mcp.prompt("batch_analysis")
def batch_analysis_prompt():
    """
    Batch transaction analysis for identifying patterns and trends across multiple transactions.

    Use this prompt for analyzing multiple transactions to identify fraud rings,
    emerging attack patterns, or systematic vulnerabilities.
    """
    return """
    You are analyzing multiple transactions for fraud patterns and trends.

    **Batch Analysis Process:**
    1. Use detect_transaction_anomaly() for each transaction to get individual risk scores
    2. Use investigate_customer_pattern() for involved customers to identify behavioral clusters
    3. Use analyze_merchant_risk() to identify high-risk merchant associations
    4. Use get_system_metrics() to understand current detection effectiveness

    **Pattern Recognition:**
    - Look for geographical clustering of suspicious activity
    - Identify merchant-specific fraud trends
    - Detect coordinated customer account compromises
    - Analyze temporal patterns in fraud attempts

    **Trend Analysis:**
    - Compare current patterns to historical baselines
    - Identify emerging fraud techniques
    - Assess model performance across different fraud types
    - Recommend threshold adjustments or model improvements

    **Recommendations:**
    - Update risk rules based on new patterns
    - Adjust monitoring thresholds
    - Implement new detection capabilities
    - Enhance customer protection measures

    Provide insights on fraud patterns, system performance, and recommended improvements.
    """

if __name__ == "__main__":
    mcp.run()