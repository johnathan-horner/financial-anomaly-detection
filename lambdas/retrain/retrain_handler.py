"""
AWS Lambda Function for Model Retraining Pipeline

Triggered by EventBridge schedule to initiate model retraining workflow
using updated feedback data and performance metrics.
"""

import json
import logging
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

# Environment variables
TRAINING_JOB_ROLE = os.environ.get('TRAINING_JOB_ROLE')
TRAINING_IMAGE_URI = os.environ.get('TRAINING_IMAGE_URI')
S3_BUCKET = os.environ.get('S3_BUCKET')
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'financial-anomaly-detector')
RETRAINING_TOPIC_ARN = os.environ.get('RETRAINING_TOPIC_ARN')
FEEDBACK_TABLE = os.environ.get('FEEDBACK_TABLE', 'feedback')
MIN_FEEDBACK_COUNT = int(os.environ.get('MIN_FEEDBACK_COUNT', '100'))


def lambda_handler(event, context):
    """Main Lambda handler for retraining workflow."""

    try:
        logger.info("Starting model retraining pipeline")

        # Step 1: Check if retraining is needed
        retraining_decision = should_retrain()

        if not retraining_decision['should_retrain']:
            logger.info(f"Retraining not needed: {retraining_decision['reason']}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Retraining not needed',
                    'reason': retraining_decision['reason']
                })
            }

        # Step 2: Prepare training data
        training_data_info = prepare_training_data()

        # Step 3: Start SageMaker training job
        training_job_result = start_training_job(training_data_info)

        # Step 4: Send notification
        send_notification({
            'event': 'retraining_started',
            'training_job_name': training_job_result['training_job_name'],
            'training_data': training_data_info,
            'decision_factors': retraining_decision
        })

        logger.info(f"Retraining pipeline started: {training_job_result['training_job_name']}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Retraining pipeline started',
                'training_job_name': training_job_result['training_job_name'],
                'estimated_completion': training_job_result['estimated_completion']
            })
        }

    except Exception as e:
        logger.error(f"Retraining pipeline failed: {str(e)}")

        # Send error notification
        send_notification({
            'event': 'retraining_failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def should_retrain() -> Dict[str, Any]:
    """Determine if model retraining is needed based on various factors."""

    try:
        # Check 1: Feedback volume
        feedback_count = get_recent_feedback_count()

        if feedback_count < MIN_FEEDBACK_COUNT:
            return {
                'should_retrain': False,
                'reason': f'Insufficient feedback data: {feedback_count} < {MIN_FEEDBACK_COUNT}'
            }

        # Check 2: Model performance degradation
        performance_metrics = get_model_performance_metrics()

        current_f1 = performance_metrics.get('f1_score', 0.8)
        baseline_f1 = 0.75  # Minimum acceptable F1 score

        if current_f1 < baseline_f1:
            return {
                'should_retrain': True,
                'reason': f'Performance degradation: F1={current_f1:.3f} < {baseline_f1}',
                'feedback_count': feedback_count,
                'performance': performance_metrics
            }

        # Check 3: Data drift
        drift_score = get_data_drift_score()

        if drift_score > 0.3:
            return {
                'should_retrain': True,
                'reason': f'Data drift detected: score={drift_score:.3f}',
                'feedback_count': feedback_count,
                'performance': performance_metrics,
                'drift_score': drift_score
            }

        # Check 4: Time-based retraining (weekly)
        last_training = get_last_training_time()
        days_since_training = (datetime.utcnow() - last_training).days

        if days_since_training >= 7:
            return {
                'should_retrain': True,
                'reason': f'Scheduled retraining: {days_since_training} days since last training',
                'feedback_count': feedback_count,
                'performance': performance_metrics
            }

        return {
            'should_retrain': False,
            'reason': 'All metrics within acceptable ranges',
            'feedback_count': feedback_count,
            'performance': performance_metrics
        }

    except Exception as e:
        logger.error(f"Error determining retraining need: {str(e)}")
        # Default to retraining on error to be safe
        return {
            'should_retrain': True,
            'reason': f'Error in evaluation, defaulting to retrain: {str(e)}'
        }


def get_recent_feedback_count() -> int:
    """Get count of feedback records from last 7 days."""

    try:
        feedback_table = dynamodb.Table(FEEDBACK_TABLE)

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        response = feedback_table.scan(
            FilterExpression='#ts BETWEEN :start_date AND :end_date',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={
                ':start_date': start_date.isoformat(),
                ':end_date': end_date.isoformat()
            },
            Select='COUNT'
        )

        return response.get('Count', 0)

    except ClientError as e:
        logger.error(f"Error getting feedback count: {str(e)}")
        return 0


def get_model_performance_metrics() -> Dict[str, float]:
    """Get recent model performance metrics."""

    try:
        # In a real implementation, this would query a metrics storage system
        # For now, return placeholder metrics
        return {
            'f1_score': 0.82,
            'precision': 0.85,
            'recall': 0.79,
            'false_positive_rate': 0.02,
            'auc_roc': 0.91
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {'f1_score': 0.5}  # Conservative fallback


def get_data_drift_score() -> float:
    """Calculate data drift score based on recent transaction patterns."""

    try:
        # In a real implementation, this would calculate drift using:
        # - Feature distribution changes
        # - Population stability index
        # - KL divergence between current and training distributions

        # Placeholder implementation
        return 0.15  # No significant drift

    except Exception as e:
        logger.error(f"Error calculating drift score: {str(e)}")
        return 0.5  # Conservative fallback indicating potential drift


def get_last_training_time() -> datetime:
    """Get timestamp of last successful training job."""

    try:
        # Query SageMaker for recent training jobs
        response = sagemaker.list_training_jobs(
            NameContains='financial-anomaly',
            StatusEquals='Completed',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        training_jobs = response.get('TrainingJobSummaries', [])

        if training_jobs:
            return training_jobs[0]['CreationTime']
        else:
            # No previous training jobs found, assume 30 days ago
            return datetime.utcnow() - timedelta(days=30)

    except Exception as e:
        logger.error(f"Error getting last training time: {str(e)}")
        return datetime.utcnow() - timedelta(days=30)


def prepare_training_data() -> Dict[str, Any]:
    """Prepare and upload training data to S3."""

    try:
        # Step 1: Collect feedback data
        feedback_data = collect_feedback_data()

        # Step 2: Merge with original training data
        updated_dataset = merge_with_original_data(feedback_data)

        # Step 3: Upload to S3
        training_data_uri = upload_training_data(updated_dataset)

        return {
            'training_data_uri': training_data_uri,
            'feedback_records': len(feedback_data),
            'total_records': len(updated_dataset),
            'data_version': datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        }

    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise


def collect_feedback_data() -> List[Dict]:
    """Collect all feedback data for retraining."""

    try:
        feedback_table = dynamodb.Table(FEEDBACK_TABLE)

        # Scan all feedback records
        response = feedback_table.scan()

        feedback_records = []
        for item in response.get('Items', []):
            feedback_records.append(convert_dynamodb_item(item))

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = feedback_table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            for item in response.get('Items', []):
                feedback_records.append(convert_dynamodb_item(item))

        logger.info(f"Collected {len(feedback_records)} feedback records")
        return feedback_records

    except ClientError as e:
        logger.error(f"Error collecting feedback data: {str(e)}")
        raise


def merge_with_original_data(feedback_data: List[Dict]) -> List[Dict]:
    """Merge feedback data with original training dataset."""

    try:
        # Download original training data from S3
        original_data_key = 'training-data/transactions.json'

        s3_response = s3.get_object(
            Bucket=S3_BUCKET,
            Key=original_data_key
        )

        original_data = json.loads(s3_response['Body'].read().decode())

        # Update labels based on feedback
        transaction_feedback = {
            fb['transaction_id']: fb['feedback_type']
            for fb in feedback_data
        }

        updated_data = []
        for record in original_data:
            transaction_id = record.get('transaction_id')

            if transaction_id in transaction_feedback:
                # Update label based on feedback
                feedback_type = transaction_feedback[transaction_id]

                if feedback_type == 'confirmed_fraud':
                    record['is_anomaly'] = True
                elif feedback_type == 'false_positive':
                    record['is_anomaly'] = False
                # 'uncertain' keeps original label

            updated_data.append(record)

        logger.info(f"Updated {len(updated_data)} training records with feedback")
        return updated_data

    except Exception as e:
        logger.error(f"Error merging training data: {str(e)}")
        raise


def upload_training_data(dataset: List[Dict]) -> str:
    """Upload training dataset to S3."""

    try:
        # Create training data file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        s3_key = f'retraining-data/{timestamp}/training_data.json'

        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(dataset, default=str),
            ContentType='application/json'
        )

        training_data_uri = f's3://{S3_BUCKET}/{s3_key}'
        logger.info(f"Training data uploaded to {training_data_uri}")

        return training_data_uri

    except Exception as e:
        logger.error(f"Error uploading training data: {str(e)}")
        raise


def start_training_job(training_data_info: Dict) -> Dict[str, Any]:
    """Start SageMaker training job."""

    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        job_name = f'financial-anomaly-retrain-{timestamp}'

        # Training job configuration
        training_config = {
            'TrainingJobName': job_name,
            'RoleArn': TRAINING_JOB_ROLE,
            'AlgorithmSpecification': {
                'TrainingImage': TRAINING_IMAGE_URI,
                'TrainingInputMode': 'File'
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': training_data_info['training_data_uri'],
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/json',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{S3_BUCKET}/model-artifacts/{timestamp}/'
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600  # 1 hour
            },
            'HyperParameters': {
                'epochs': '100',
                'learning_rate': '0.001',
                'batch_size': '256'
            },
            'Tags': [
                {'Key': 'Project', 'Value': 'FinancialAnomalyDetection'},
                {'Key': 'Purpose', 'Value': 'Retraining'},
                {'Key': 'DataVersion', 'Value': training_data_info['data_version']}
            ]
        }

        # Start training job
        response = sagemaker.create_training_job(**training_config)

        # Estimate completion time (1 hour from now)
        estimated_completion = datetime.utcnow() + timedelta(hours=1)

        return {
            'training_job_name': job_name,
            'training_job_arn': response['TrainingJobArn'],
            'estimated_completion': estimated_completion.isoformat()
        }

    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise


def send_notification(message: Dict) -> None:
    """Send notification about retraining status."""

    try:
        if not RETRAINING_TOPIC_ARN:
            logger.warning("No SNS topic configured for notifications")
            return

        sns.publish(
            TopicArn=RETRAINING_TOPIC_ARN,
            Subject=f"Fraud Detection Model Retraining - {message['event']}",
            Message=json.dumps(message, indent=2, default=str)
        )

        logger.info(f"Notification sent: {message['event']}")

    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        # Don't raise - notification failure shouldn't stop retraining


def convert_dynamodb_item(item: Dict) -> Dict:
    """Convert DynamoDB item to standard Python types."""

    from decimal import Decimal

    converted = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            converted[key] = float(value)
        elif isinstance(value, set):
            converted[key] = list(value)
        else:
            converted[key] = value

    return converted