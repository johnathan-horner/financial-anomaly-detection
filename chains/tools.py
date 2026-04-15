"""
LangChain Tools for Database Queries and Model Inference

Implements tools for accessing customer history, merchant risk data,
and SageMaker model scoring within the LangGraph investigation workflow.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

import numpy as np

logger = logging.getLogger(__name__)


class CustomerHistoryTool:
    """Tool for retrieving customer transaction history from DynamoDB."""

    def __init__(self, dynamodb_client):
        self.dynamodb = dynamodb_client
        self.table_name = "customer-history"

    def get_customer_history(
        self,
        customer_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """Retrieve customer transaction history within date range."""

        try:
            table = self.dynamodb.Table(self.table_name)

            # Query with date range
            response = table.query(
                KeyConditionExpression="customer_id = :customer_id AND #ts BETWEEN :start_date AND :end_date",
                ExpressionAttributeNames={
                    "#ts": "timestamp"
                },
                ExpressionAttributeValues={
                    ":customer_id": customer_id,
                    ":start_date": start_date.isoformat(),
                    ":end_date": end_date.isoformat()
                },
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )

            transactions = []
            for item in response.get("Items", []):
                # Convert DynamoDB types to standard Python types
                transaction = self._convert_dynamodb_item(item)
                transactions.append(transaction)

            logger.info(f"Retrieved {len(transactions)} transactions for customer {customer_id}")
            return transactions

        except ClientError as e:
            logger.error(f"DynamoDB query failed for customer {customer_id}: {str(e)}")
            raise

    def _convert_dynamodb_item(self, item: Dict) -> Dict:
        """Convert DynamoDB item to standard Python types."""

        converted = {}
        for key, value in item.items():
            if isinstance(value, Decimal):
                # Convert Decimal to float
                converted[key] = float(value)
            else:
                converted[key] = value

        return converted


class MerchantRiskTool:
    """Tool for retrieving merchant risk data from DynamoDB."""

    def __init__(self, dynamodb_client):
        self.dynamodb = dynamodb_client
        self.table_name = "merchant-risk"

    def get_merchant_risk(self, merchant_id: str) -> Dict[str, Any]:
        """Retrieve merchant risk profile and fraud flags."""

        try:
            table = self.dynamodb.Table(self.table_name)

            response = table.get_item(
                Key={"merchant_id": merchant_id}
            )

            if "Item" in response:
                merchant_data = self._convert_dynamodb_item(response["Item"])
                logger.info(f"Retrieved merchant data for {merchant_id}")
                return merchant_data
            else:
                # Return default risk profile for unknown merchants
                logger.warning(f"No risk data found for merchant {merchant_id}")
                return {
                    "merchant_id": merchant_id,
                    "risk_level": "medium",
                    "fraud_flags": [],
                    "category_risk_score": 0.5,
                    "transaction_volume": 0,
                    "chargeback_rate": 0.0,
                    "last_updated": datetime.utcnow().isoformat()
                }

        except ClientError as e:
            logger.error(f"DynamoDB query failed for merchant {merchant_id}: {str(e)}")
            raise

    def _convert_dynamodb_item(self, item: Dict) -> Dict:
        """Convert DynamoDB item to standard Python types."""

        converted = {}
        for key, value in item.items():
            if isinstance(value, Decimal):
                converted[key] = float(value)
            elif isinstance(value, set):
                converted[key] = list(value)
            else:
                converted[key] = value

        return converted


class TransactionScorer:
    """Tool for scoring transactions using SageMaker endpoint."""

    def __init__(self, sagemaker_client):
        self.sagemaker = sagemaker_client
        self.endpoint_name = "financial-anomaly-detector"

    def score_transaction(self, transaction_features: Dict[str, Any]) -> Dict[str, float]:
        """Score a transaction using the deployed autoencoder model."""

        try:
            # Prepare features for model input
            feature_vector = self._prepare_features(transaction_features)

            # Convert to JSON payload
            payload = {
                "instances": [feature_vector]
            }

            # Invoke SageMaker endpoint
            response = self.sagemaker.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )

            # Parse response
            result = json.loads(response["Body"].read().decode())

            scores = {
                "anomaly_score": result.get("predictions", [0.0])[0],
                "reconstruction_error": result.get("reconstruction_errors", [0.0])[0],
                "feature_contributions": result.get("feature_contributions", {})
            }

            logger.info(f"Transaction scored: anomaly_score={scores['anomaly_score']:.4f}")
            return scores

        except Exception as e:
            logger.error(f"SageMaker scoring failed: {str(e)}")
            # Return default score
            return {
                "anomaly_score": 0.5,
                "reconstruction_error": 0.5,
                "feature_contributions": {}
            }

    def _prepare_features(self, transaction: Dict[str, Any]) -> List[float]:
        """Prepare transaction features for model input."""

        # This should match the feature preparation in the autoencoder model
        features = []

        # Amount features
        amount = float(transaction.get("amount", 0))
        features.append(np.log1p(amount))  # log_amount

        # Time features
        timestamp = datetime.fromisoformat(transaction.get("timestamp", datetime.utcnow().isoformat()))
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        features.extend([
            np.sin(2 * np.pi * hour / 24),  # hour_sin
            np.cos(2 * np.pi * hour / 24),  # hour_cos
            np.sin(2 * np.pi * day_of_week / 7),  # dow_sin
            np.cos(2 * np.pi * day_of_week / 7)   # dow_cos
        ])

        # Geographic features
        features.append(float(transaction.get("distance_from_home", 0)))

        # Categorical features
        features.append(float(transaction.get("is_preferred_category", 0)))
        features.append(float(transaction.get("is_weekend", 0)))

        # Statistical features
        features.extend([
            float(transaction.get("amount_zscore", 0)),
            float(transaction.get("transactions_last_hour", 0)),
            float(transaction.get("transactions_last_day", 0)),
            np.log1p(float(transaction.get("spend_last_hour", 0))),
            np.log1p(float(transaction.get("spend_last_day", 0)))
        ])

        # Encoded categorical features (simplified mapping)
        merchant_category = transaction.get("merchant_category", "unknown")
        category_mapping = {
            "grocery": 0, "gas_station": 1, "restaurant": 2, "retail": 3,
            "pharmacy": 4, "entertainment": 5, "travel": 6, "online": 7,
            "utility": 8, "insurance": 9, "healthcare": 10, "education": 11,
            "automotive": 12, "home_improvement": 13, "unknown": 14
        }
        features.append(float(category_mapping.get(merchant_category, 14)))

        risk_profile = transaction.get("risk_profile", "medium")
        risk_mapping = {"low_risk": 0, "medium_risk": 1, "high_risk": 2}
        features.append(float(risk_mapping.get(risk_profile, 1)))

        return features


class DatabaseSeeder:
    """Utility for seeding DynamoDB tables with initial data."""

    def __init__(self, dynamodb_client):
        self.dynamodb = dynamodb_client

    def seed_merchant_risk_data(self) -> None:
        """Seed merchant risk table with realistic data."""

        merchant_risk_table = self.dynamodb.Table("merchant-risk")

        # Sample merchant categories with risk profiles
        merchant_data = [
            {
                "merchant_id": "MERCH_grocery_0001",
                "merchant_name": "SuperMart",
                "category": "grocery",
                "risk_level": "low",
                "fraud_flags": [],
                "category_risk_score": 0.1,
                "transaction_volume": 50000,
                "chargeback_rate": 0.002,
                "avg_transaction_amount": 65.50,
                "last_updated": datetime.utcnow().isoformat()
            },
            {
                "merchant_id": "MERCH_online_0001",
                "merchant_name": "FastDelivery",
                "category": "online",
                "risk_level": "medium",
                "fraud_flags": ["high_velocity"],
                "category_risk_score": 0.4,
                "transaction_volume": 25000,
                "chargeback_rate": 0.008,
                "avg_transaction_amount": 120.00,
                "last_updated": datetime.utcnow().isoformat()
            },
            {
                "merchant_id": "MERCH_entertainment_0001",
                "merchant_name": "GamingHub",
                "category": "entertainment",
                "risk_level": "high",
                "fraud_flags": ["suspected_fraud", "unusual_patterns"],
                "category_risk_score": 0.8,
                "transaction_volume": 5000,
                "chargeback_rate": 0.15,
                "avg_transaction_amount": 299.99,
                "last_updated": datetime.utcnow().isoformat()
            }
        ]

        try:
            with merchant_risk_table.batch_writer() as batch:
                for merchant in merchant_data:
                    batch.put_item(Item=merchant)

            logger.info(f"Seeded {len(merchant_data)} merchant risk records")

        except ClientError as e:
            logger.error(f"Failed to seed merchant risk data: {str(e)}")
            raise

    def seed_customer_history_sample(self, customer_profiles: List[Dict]) -> None:
        """Seed customer history table with sample data."""

        customer_history_table = self.dynamodb.Table("customer-history")

        try:
            # Take first 10 customer profiles for seeding
            sample_customers = customer_profiles[:10]

            with customer_history_table.batch_writer() as batch:
                for customer in sample_customers:
                    # Generate 30 days of sample history per customer
                    for day_offset in range(30):
                        transaction_date = datetime.utcnow() - timedelta(days=day_offset)

                        # Generate 1-3 transactions per day
                        num_transactions = np.random.randint(1, 4)
                        for _ in range(num_transactions):
                            transaction_time = transaction_date.replace(
                                hour=np.random.randint(8, 22),
                                minute=np.random.randint(0, 59),
                                second=np.random.randint(0, 59)
                            )

                            # Random transaction based on customer profile
                            amount_params = customer['typical_amount_range']
                            amount = max(1.0, np.random.normal(
                                amount_params['amount_mean'],
                                amount_params['amount_std']
                            ))

                            category = np.random.choice(customer['preferred_categories'])

                            transaction = {
                                "customer_id": customer['customer_id'],
                                "timestamp": transaction_time.isoformat(),
                                "transaction_id": f"HIST_{customer['customer_id']}_{day_offset}_{_}",
                                "amount": Decimal(str(round(amount, 2))),
                                "merchant_category": category,
                                "merchant_id": f"MERCH_{category}_{np.random.randint(1, 100):04d}",
                                "is_anomaly": False
                            }

                            batch.put_item(Item=transaction)

            logger.info(f"Seeded customer history for {len(sample_customers)} customers")

        except ClientError as e:
            logger.error(f"Failed to seed customer history: {str(e)}")
            raise