"""
AWS Lambda Function for Dashboard API Endpoints

Provides REST API endpoints for fraud detection dashboard metrics,
transaction queries, and analyst feedback collection.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
TRANSACTIONS_TABLE = os.environ.get('TRANSACTIONS_TABLE', 'transactions')
INVESTIGATIONS_TABLE = os.environ.get('INVESTIGATIONS_TABLE', 'investigations')
FEEDBACK_QUEUE_URL = os.environ.get('FEEDBACK_QUEUE_URL')


def lambda_handler(event, context):
    """Main Lambda handler for API Gateway requests."""

    try:
        # Parse request
        http_method = event.get('httpMethod')
        resource_path = event.get('resource')
        path_parameters = event.get('pathParameters') or {}
        query_parameters = event.get('queryStringParameters') or {}
        body = event.get('body')

        logger.info(f"API request: {http_method} {resource_path}")

        # Route to appropriate handler
        if resource_path == '/transactions/{id}' and http_method == 'GET':
            return get_transaction(path_parameters.get('id'))

        elif resource_path == '/dashboard/metrics' and http_method == 'GET':
            return get_dashboard_metrics(query_parameters)

        elif resource_path == '/dashboard/drift' and http_method == 'GET':
            return get_model_drift_metrics(query_parameters)

        elif resource_path == '/feedback' and http_method == 'POST':
            return submit_feedback(json.loads(body) if body else {})

        else:
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Endpoint not found'})
            }

    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Internal server error'})
        }


def get_transaction(transaction_id: str) -> Dict:
    """Get full transaction record with investigation details."""

    try:
        if not transaction_id:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing transaction_id'})
            }

        # Get transaction from DynamoDB
        transactions_table = dynamodb.Table(TRANSACTIONS_TABLE)

        response = transactions_table.get_item(
            Key={'transaction_id': transaction_id}
        )

        if 'Item' not in response:
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Transaction not found'})
            }

        transaction = convert_dynamodb_item(response['Item'])

        # Get investigation details if available
        investigation = None
        if transaction.get('decision') != 'auto_approve':
            investigation = get_investigation_details(transaction_id)

        result = {
            'transaction': transaction,
            'investigation': investigation
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(result, default=str)
        }

    except ClientError as e:
        logger.error(f"DynamoDB error getting transaction {transaction_id}: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Database error'})
        }


def get_investigation_details(transaction_id: str) -> Optional[Dict]:
    """Get investigation details for a transaction."""

    try:
        investigations_table = dynamodb.Table(INVESTIGATIONS_TABLE)

        response = investigations_table.get_item(
            Key={'transaction_id': transaction_id}
        )

        if 'Item' in response:
            return convert_dynamodb_item(response['Item'])

        return None

    except ClientError as e:
        logger.error(f"Error getting investigation for {transaction_id}: {str(e)}")
        return None


def get_dashboard_metrics(query_params: Dict) -> Dict:
    """Get fraud detection dashboard metrics."""

    try:
        # Parse time range
        time_range = query_params.get('timeRange', '24h')
        hours = parse_time_range(time_range)

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get transaction metrics from DynamoDB
        transaction_metrics = get_transaction_metrics(start_time, end_time)

        # Get CloudWatch metrics
        cloudwatch_metrics = get_cloudwatch_dashboard_metrics(start_time, end_time)

        # Combine metrics
        dashboard_data = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'transaction_metrics': transaction_metrics,
            'performance_metrics': cloudwatch_metrics,
            'generated_at': datetime.utcnow().isoformat()
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(dashboard_data, default=str)
        }

    except Exception as e:
        logger.error(f"Error generating dashboard metrics: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Failed to generate metrics'})
        }


def get_transaction_metrics(start_time: datetime, end_time: datetime) -> Dict:
    """Get transaction-level metrics from DynamoDB."""

    try:
        transactions_table = dynamodb.Table(TRANSACTIONS_TABLE)

        # Query transactions in time range
        response = transactions_table.scan(
            FilterExpression='processed_timestamp BETWEEN :start AND :end',
            ExpressionAttributeValues={
                ':start': start_time.isoformat(),
                ':end': end_time.isoformat()
            }
        )

        transactions = [convert_dynamodb_item(item) for item in response.get('Items', [])]

        # Calculate metrics
        total_transactions = len(transactions)
        auto_approved = len([t for t in transactions if t.get('decision') == 'auto_approve'])
        flagged = len([t for t in transactions if t.get('decision') == 'investigate'])
        blocked = len([t for t in transactions if t.get('decision') == 'block_and_alert'])

        # Anomaly score distribution
        scores = [float(t.get('anomaly_score', 0)) for t in transactions if t.get('anomaly_score')]
        score_distribution = {
            'mean': sum(scores) / len(scores) if scores else 0,
            'median': sorted(scores)[len(scores)//2] if scores else 0,
            'p95': sorted(scores)[int(0.95 * len(scores))] if scores else 0,
            'p99': sorted(scores)[int(0.99 * len(scores))] if scores else 0
        }

        return {
            'total_transactions': total_transactions,
            'auto_approved': auto_approved,
            'flagged_for_investigation': flagged,
            'blocked': blocked,
            'fraud_rate': flagged / total_transactions if total_transactions > 0 else 0,
            'auto_approval_rate': auto_approved / total_transactions if total_transactions > 0 else 0,
            'anomaly_score_distribution': score_distribution
        }

    except ClientError as e:
        logger.error(f"Error getting transaction metrics: {str(e)}")
        return {}


def get_cloudwatch_dashboard_metrics(start_time: datetime, end_time: datetime) -> Dict:
    """Get performance metrics from CloudWatch."""

    try:
        metrics = {}

        # Define metrics to retrieve
        metric_queries = [
            {
                'name': 'processed_count',
                'namespace': 'FinancialAnomalyDetection/Scoring',
                'metric_name': 'processed_count',
                'statistic': 'Sum'
            },
            {
                'name': 'flagged_count',
                'namespace': 'FinancialAnomalyDetection/Scoring',
                'metric_name': 'flagged_count',
                'statistic': 'Sum'
            },
            {
                'name': 'lambda_duration',
                'namespace': 'AWS/Lambda',
                'metric_name': 'Duration',
                'statistic': 'Average'
            },
            {
                'name': 'lambda_errors',
                'namespace': 'AWS/Lambda',
                'metric_name': 'Errors',
                'statistic': 'Sum'
            }
        ]

        for query in metric_queries:
            try:
                response = cloudwatch.get_metric_statistics(
                    Namespace=query['namespace'],
                    MetricName=query['metric_name'],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,  # 1 hour periods
                    Statistics=[query['statistic']]
                )

                datapoints = response.get('Datapoints', [])
                if datapoints:
                    # Get latest value
                    latest = max(datapoints, key=lambda x: x['Timestamp'])
                    metrics[query['name']] = latest[query['statistic']]
                else:
                    metrics[query['name']] = 0

            except Exception as e:
                logger.warning(f"Failed to get metric {query['name']}: {str(e)}")
                metrics[query['name']] = 0

        return metrics

    except Exception as e:
        logger.error(f"Error getting CloudWatch metrics: {str(e)}")
        return {}


def get_model_drift_metrics(query_params: Dict) -> Dict:
    """Get model drift and performance metrics."""

    try:
        # Parse time range
        time_range = query_params.get('timeRange', '7d')
        hours = parse_time_range(time_range)

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get drift metrics from CloudWatch
        drift_metrics = {}

        metric_queries = [
            {
                'name': 'avg_anomaly_score',
                'namespace': 'FinancialAnomalyDetection/ModelDrift',
                'metric_name': 'AverageAnomalyScore',
                'statistic': 'Average'
            },
            {
                'name': 'score_variance',
                'namespace': 'FinancialAnomalyDetection/ModelDrift',
                'metric_name': 'ScoreVariance',
                'statistic': 'Average'
            },
            {
                'name': 'feature_drift_score',
                'namespace': 'FinancialAnomalyDetection/ModelDrift',
                'metric_name': 'FeatureDriftScore',
                'statistic': 'Average'
            }
        ]

        for query in metric_queries:
            try:
                response = cloudwatch.get_metric_statistics(
                    Namespace=query['namespace'],
                    MetricName=query['metric_name'],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # Daily periods
                    Statistics=[query['statistic']]
                )

                datapoints = response.get('Datapoints', [])
                if datapoints:
                    # Get time series data
                    time_series = [
                        {
                            'timestamp': dp['Timestamp'].isoformat(),
                            'value': dp[query['statistic']]
                        }
                        for dp in sorted(datapoints, key=lambda x: x['Timestamp'])
                    ]
                    drift_metrics[query['name']] = time_series
                else:
                    drift_metrics[query['name']] = []

            except Exception as e:
                logger.warning(f"Failed to get drift metric {query['name']}: {str(e)}")
                drift_metrics[query['name']] = []

        result = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'drift_metrics': drift_metrics,
            'generated_at': datetime.utcnow().isoformat()
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(result, default=str)
        }

    except Exception as e:
        logger.error(f"Error getting drift metrics: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Failed to get drift metrics'})
        }


def submit_feedback(feedback_data: Dict) -> Dict:
    """Submit analyst feedback for model training."""

    try:
        # Validate feedback data
        required_fields = ['transaction_id', 'feedback_type', 'analyst_id']
        for field in required_fields:
            if field not in feedback_data:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': f'Missing required field: {field}'})
                }

        valid_feedback_types = ['confirmed_fraud', 'false_positive', 'uncertain']
        if feedback_data['feedback_type'] not in valid_feedback_types:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': f'Invalid feedback_type. Must be one of: {valid_feedback_types}'})
            }

        # Send feedback to SQS queue for processing
        sqs = boto3.client('sqs')

        feedback_message = {
            'transaction_id': feedback_data['transaction_id'],
            'feedback_type': feedback_data['feedback_type'],
            'analyst_id': feedback_data['analyst_id'],
            'confidence': feedback_data.get('confidence', 1.0),
            'notes': feedback_data.get('notes', ''),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'dashboard_api'
        }

        response = sqs.send_message(
            QueueUrl=FEEDBACK_QUEUE_URL,
            MessageBody=json.dumps(feedback_message, default=str)
        )

        logger.info(f"Feedback submitted for transaction {feedback_data['transaction_id']}: "
                   f"MessageId={response['MessageId']}")

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'message': 'Feedback submitted successfully',
                'message_id': response['MessageId']
            })
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Failed to submit feedback'})
        }


def parse_time_range(time_range: str) -> int:
    """Parse time range string to hours."""

    time_map = {
        '1h': 1,
        '6h': 6,
        '12h': 12,
        '24h': 24,
        '3d': 72,
        '7d': 168,
        '30d': 720
    }

    return time_map.get(time_range, 24)


def convert_dynamodb_item(item: Dict) -> Dict:
    """Convert DynamoDB item to standard Python types."""

    converted = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            # Convert Decimal to float
            converted[key] = float(value)
        elif isinstance(value, set):
            # Convert set to list
            converted[key] = list(value)
        else:
            converted[key] = value

    return converted