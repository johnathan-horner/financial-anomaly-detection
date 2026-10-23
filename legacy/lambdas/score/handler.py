"""
AWS Lambda Function for Real-time Transaction Scoring

Consumes transactions from Kinesis Data Stream, scores them using SageMaker endpoint,
and routes based on anomaly score (auto-approve or send to investigation queue).
"""

import json
import base64
import logging
import os
from typing import Dict, List, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker = boto3.client('sagemaker-runtime')
sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
SAGEMAKER_ENDPOINT = os.environ.get('SAGEMAKER_ENDPOINT', 'financial-anomaly-detector')
INVESTIGATION_QUEUE_URL = os.environ.get('INVESTIGATION_QUEUE_URL')
TRANSACTIONS_TABLE = os.environ.get('TRANSACTIONS_TABLE', 'transactions')
AUTO_APPROVE_THRESHOLD = float(os.environ.get('AUTO_APPROVE_THRESHOLD', '0.3'))


def lambda_handler(event, context):
    """
    Main Lambda handler for processing Kinesis stream records.

    Expected Kinesis record format:
    {
        "transaction_id": "TXN_12345",
        "customer_id": "CUST_67890",
        "amount": 150.50,
        "merchant_id": "MERCH_retail_001",
        "merchant_category": "retail",
        "timestamp": "2024-01-01T12:00:00Z",
        "location_lat": 40.7128,
        "location_lon": -74.0060
    }
    """

    processed_count = 0
    auto_approved_count = 0
    flagged_count = 0
    errors = []

    try:
        # Process each record from Kinesis
        for record in event.get('Records', []):
            try:
                # Decode Kinesis record
                transaction = decode_kinesis_record(record)

                # Score transaction
                scoring_result = score_transaction(transaction)

                # Route based on score
                routing_result = route_transaction(transaction, scoring_result)

                # Update metrics
                processed_count += 1
                if routing_result['decision'] == 'auto_approve':
                    auto_approved_count += 1
                else:
                    flagged_count += 1

                logger.info(f"Processed transaction {transaction['transaction_id']}: "
                           f"score={scoring_result['anomaly_score']:.4f}, "
                           f"decision={routing_result['decision']}")

            except Exception as e:
                error_msg = f"Failed to process record: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Send metrics to CloudWatch
        send_metrics({
            'processed_count': processed_count,
            'auto_approved_count': auto_approved_count,
            'flagged_count': flagged_count,
            'error_count': len(errors)
        })

        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed': processed_count,
                'auto_approved': auto_approved_count,
                'flagged': flagged_count,
                'errors': len(errors)
            })
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def decode_kinesis_record(record: Dict) -> Dict:
    """Decode base64 Kinesis record data."""

    try:
        # Decode base64 data
        data = base64.b64decode(record['kinesis']['data']).decode('utf-8')
        transaction = json.loads(data)

        # Validate required fields
        required_fields = ['transaction_id', 'customer_id', 'amount', 'merchant_id', 'timestamp']
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")

        return transaction

    except Exception as e:
        raise ValueError(f"Failed to decode Kinesis record: {str(e)}")


def score_transaction(transaction: Dict) -> Dict:
    """Score transaction using SageMaker endpoint."""

    try:
        # Prepare features for model
        features = prepare_transaction_features(transaction)

        # Create payload for SageMaker
        payload = {
            "instances": [features],
            "configuration": {
                "return_feature_contributions": True
            }
        }

        # Invoke SageMaker endpoint
        response = sagemaker.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Parse response
        result = json.loads(response['Body'].read().decode())

        return {
            'anomaly_score': result.get('predictions', [0.5])[0],
            'reconstruction_error': result.get('reconstruction_errors', [0.0])[0],
            'feature_contributions': result.get('feature_contributions', {}),
            'model_version': result.get('model_version', 'unknown'),
            'scoring_timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"SageMaker scoring failed for {transaction['transaction_id']}: {str(e)}")
        # Return fallback score
        return {
            'anomaly_score': 0.5,
            'reconstruction_error': 0.5,
            'feature_contributions': {},
            'model_version': 'fallback',
            'scoring_timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }


def prepare_transaction_features(transaction: Dict) -> List[float]:
    """Prepare transaction features for model input."""

    # This should match the preprocessing in the autoencoder model
    import numpy as np

    features = []

    # Amount features
    amount = float(transaction.get('amount', 0))
    features.append(np.log1p(amount))

    # Time features
    try:
        timestamp = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
    except:
        # Fallback time features
        features.extend([0.0, 1.0, 0.0, 1.0])

    # Geographic features (simplified)
    features.append(float(transaction.get('distance_from_home', 10.0)))

    # Categorical features
    features.extend([
        float(transaction.get('is_preferred_category', 1)),
        float(timestamp.weekday() >= 5 if 'timestamp' in locals() else 0)  # is_weekend
    ])

    # Statistical features (defaults for real-time scoring)
    features.extend([
        0.0,  # amount_zscore (calculated in batch)
        float(transaction.get('transactions_last_hour', 0)),
        float(transaction.get('transactions_last_day', 0)),
        np.log1p(float(transaction.get('spend_last_hour', 0))),
        np.log1p(float(transaction.get('spend_last_day', 0)))
    ])

    # Encoded categorical features
    merchant_category = transaction.get('merchant_category', 'unknown')
    category_mapping = {
        'grocery': 0, 'gas_station': 1, 'restaurant': 2, 'retail': 3,
        'pharmacy': 4, 'entertainment': 5, 'travel': 6, 'online': 7,
        'utility': 8, 'insurance': 9, 'healthcare': 10, 'education': 11,
        'automotive': 12, 'home_improvement': 13, 'unknown': 14
    }
    features.append(float(category_mapping.get(merchant_category, 14)))

    # Risk profile (default to medium)
    features.append(1.0)  # medium_risk

    return features


def route_transaction(transaction: Dict, scoring_result: Dict) -> Dict:
    """Route transaction based on anomaly score."""

    anomaly_score = scoring_result['anomaly_score']
    decision_timestamp = datetime.utcnow().isoformat()

    try:
        if anomaly_score < AUTO_APPROVE_THRESHOLD:
            # Auto-approve and store in DynamoDB
            decision = 'auto_approve'
            store_transaction(transaction, scoring_result, decision)

        else:
            # Send to investigation queue
            decision = 'investigate'
            send_to_investigation_queue(transaction, scoring_result)
            store_transaction(transaction, scoring_result, decision)

        return {
            'decision': decision,
            'anomaly_score': anomaly_score,
            'timestamp': decision_timestamp
        }

    except Exception as e:
        logger.error(f"Routing failed for {transaction['transaction_id']}: {str(e)}")
        # Store with error status
        store_transaction(transaction, scoring_result, 'error', str(e))
        return {
            'decision': 'error',
            'anomaly_score': anomaly_score,
            'timestamp': decision_timestamp,
            'error': str(e)
        }


def store_transaction(
    transaction: Dict,
    scoring_result: Dict,
    decision: str,
    error: str = None
) -> None:
    """Store transaction in DynamoDB transactions table."""

    try:
        table = dynamodb.Table(TRANSACTIONS_TABLE)

        item = {
            'transaction_id': transaction['transaction_id'],
            'timestamp': transaction['timestamp'],
            'customer_id': transaction['customer_id'],
            'amount': str(transaction['amount']),  # Store as string to avoid Decimal issues
            'merchant_id': transaction['merchant_id'],
            'merchant_category': transaction.get('merchant_category', 'unknown'),
            'location_lat': str(transaction.get('location_lat', 0)),
            'location_lon': str(transaction.get('location_lon', 0)),
            'anomaly_score': str(scoring_result['anomaly_score']),
            'reconstruction_error': str(scoring_result['reconstruction_error']),
            'model_version': scoring_result.get('model_version', 'unknown'),
            'decision': decision,
            'processed_timestamp': datetime.utcnow().isoformat(),
            'ttl': int((datetime.utcnow().timestamp() + 90 * 24 * 3600))  # 90 days TTL
        }

        if error:
            item['error'] = error

        if scoring_result.get('feature_contributions'):
            item['feature_contributions'] = json.dumps(scoring_result['feature_contributions'])

        table.put_item(Item=item)

    except ClientError as e:
        logger.error(f"Failed to store transaction {transaction['transaction_id']}: {str(e)}")
        raise


def send_to_investigation_queue(transaction: Dict, scoring_result: Dict) -> None:
    """Send flagged transaction to SQS investigation queue."""

    try:
        message_body = {
            'transaction': transaction,
            'scoring_result': scoring_result,
            'flagged_timestamp': datetime.utcnow().isoformat(),
            'source': 'kinesis_scoring_lambda'
        }

        response = sqs.send_message(
            QueueUrl=INVESTIGATION_QUEUE_URL,
            MessageBody=json.dumps(message_body, default=str),
            MessageAttributes={
                'transaction_id': {
                    'StringValue': transaction['transaction_id'],
                    'DataType': 'String'
                },
                'customer_id': {
                    'StringValue': transaction['customer_id'],
                    'DataType': 'String'
                },
                'anomaly_score': {
                    'StringValue': str(scoring_result['anomaly_score']),
                    'DataType': 'Number'
                }
            }
        )

        logger.info(f"Sent transaction {transaction['transaction_id']} to investigation queue: "
                   f"MessageId={response['MessageId']}")

    except ClientError as e:
        logger.error(f"Failed to send to investigation queue: {str(e)}")
        raise


def send_metrics(metrics: Dict) -> None:
    """Send custom metrics to CloudWatch."""

    try:
        metric_data = []

        for metric_name, value in metrics.items():
            metric_data.append({
                'MetricName': metric_name,
                'Value': value,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            })

        cloudwatch.put_metric_data(
            Namespace='FinancialAnomalyDetection/Scoring',
            MetricData=metric_data
        )

    except ClientError as e:
        logger.error(f"Failed to send CloudWatch metrics: {str(e)}")
        # Don't raise - metrics failure shouldn't stop transaction processing