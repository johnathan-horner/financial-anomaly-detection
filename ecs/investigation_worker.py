"""
ECS Fargate Investigation Worker

Long-running service that pulls flagged transactions from SQS and runs
the complete LangGraph investigation workflow using Amazon Bedrock.
"""

import json
import logging
import os
import signal
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from agents.investigation_graph import InvestigationGraphFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/investigation_worker.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
shutdown_requested = False


class InvestigationWorker:
    """ECS worker for processing investigation requests."""

    def __init__(self):
        self.setup_aws_clients()
        self.setup_investigation_graph()
        self.setup_environment()

    def setup_aws_clients(self):
        """Initialize AWS service clients."""

        try:
            self.sqs = boto3.client('sqs')
            self.dynamodb = boto3.resource('dynamodb')
            self.bedrock = boto3.client('bedrock-runtime')
            self.sagemaker = boto3.client('sagemaker-runtime')
            self.sns = boto3.client('sns')
            self.cloudwatch = boto3.client('cloudwatch')

            logger.info("AWS clients initialized successfully")

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise

    def setup_investigation_graph(self):
        """Initialize the LangGraph investigation workflow."""

        try:
            self.investigation_graph = InvestigationGraphFactory.create_graph(
                bedrock_client=self.bedrock,
                dynamodb_client=self.dynamodb,
                sagemaker_client=self.sagemaker
            )

            logger.info("Investigation graph initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize investigation graph: {str(e)}")
            raise

    def setup_environment(self):
        """Load environment variables and configuration."""

        self.queue_url = os.environ.get('INVESTIGATION_QUEUE_URL')
        if not self.queue_url:
            raise ValueError("INVESTIGATION_QUEUE_URL environment variable not set")

        self.investigations_table = os.environ.get('INVESTIGATIONS_TABLE', 'investigations')
        self.fraud_alert_topic = os.environ.get('FRAUD_ALERT_TOPIC_ARN')

        # Processing configuration
        self.max_messages = int(os.environ.get('MAX_MESSAGES_PER_BATCH', '10'))
        self.visibility_timeout = int(os.environ.get('VISIBILITY_TIMEOUT', '300'))
        self.wait_time = int(os.environ.get('WAIT_TIME_SECONDS', '20'))

        logger.info(f"Worker configured: queue={self.queue_url}, "
                   f"max_messages={self.max_messages}, "
                   f"visibility_timeout={self.visibility_timeout}")

    def run(self):
        """Main worker loop."""

        logger.info("Starting investigation worker")

        while not shutdown_requested:
            try:
                # Poll for messages
                messages = self.poll_messages()

                if not messages:
                    logger.debug("No messages received, continuing poll...")
                    continue

                logger.info(f"Received {len(messages)} messages for processing")

                # Process messages
                for message in messages:
                    try:
                        self.process_message(message)
                    except Exception as e:
                        logger.error(f"Failed to process message {message.get('MessageId', 'unknown')}: {str(e)}")
                        # Don't delete message on processing error - let it retry

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in main worker loop: {str(e)}")
                time.sleep(5)  # Brief pause before retrying

        logger.info("Investigation worker shutting down")

    def poll_messages(self) -> list:
        """Poll SQS queue for investigation requests."""

        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=self.max_messages,
                VisibilityTimeoutSeconds=self.visibility_timeout,
                WaitTimeSeconds=self.wait_time,
                MessageAttributeNames=['All']
            )

            return response.get('Messages', [])

        except ClientError as e:
            logger.error(f"Failed to poll SQS queue: {str(e)}")
            return []

    def process_message(self, message: Dict) -> None:
        """Process a single investigation message."""

        message_id = message.get('MessageId')
        receipt_handle = message.get('ReceiptHandle')

        try:
            # Parse message body
            message_body = json.loads(message['Body'])
            transaction_data = message_body.get('transaction', {})

            if not transaction_data.get('transaction_id'):
                raise ValueError("Invalid message: missing transaction_id")

            logger.info(f"Processing investigation for transaction {transaction_data['transaction_id']}")

            # Run investigation workflow
            start_time = time.time()
            investigation_result = self.investigation_graph.investigate(transaction_data)
            investigation_duration = time.time() - start_time

            # Store investigation results
            self.store_investigation_result(investigation_result, investigation_duration)

            # Route based on decision
            self.route_investigation_result(investigation_result)

            # Send metrics
            self.send_investigation_metrics(investigation_result, investigation_duration)

            # Delete message from queue (successful processing)
            self.delete_message(receipt_handle)

            logger.info(f"Investigation completed for {transaction_data['transaction_id']}: "
                       f"decision={investigation_result['decision']}, "
                       f"duration={investigation_duration:.2f}s")

        except Exception as e:
            logger.error(f"Investigation processing failed for message {message_id}: {str(e)}")
            # Don't delete message - let it retry or move to DLQ
            raise

    def store_investigation_result(
        self,
        investigation_result: Dict,
        duration: float
    ) -> None:
        """Store investigation results in DynamoDB."""

        try:
            investigations_table = self.dynamodb.Table(self.investigations_table)

            item = {
                'transaction_id': investigation_result['transaction_id'],
                'timestamp': investigation_result['timestamp'],
                'decision': investigation_result['decision'],
                'confidence': str(investigation_result['confidence']) if investigation_result['confidence'] else '0',
                'investigation_duration': str(duration),
                'customer_history_count': investigation_result['customer_history_count'],
                'merchant_risk_level': investigation_result['merchant_risk_level'] or 'unknown',
                'pattern_analysis': json.dumps(investigation_result.get('pattern_analysis', {})),
                'investigation_summary': json.dumps(investigation_result.get('investigation_summary', {})),
                'errors': json.dumps(investigation_result.get('errors', [])),
                'ttl': int((datetime.utcnow().timestamp() + 90 * 24 * 3600))  # 90 days TTL
            }

            investigations_table.put_item(Item=item)

        except ClientError as e:
            logger.error(f"Failed to store investigation result: {str(e)}")
            raise

    def route_investigation_result(self, investigation_result: Dict) -> None:
        """Route investigation result based on decision."""

        decision = investigation_result['decision']
        transaction_id = investigation_result['transaction_id']

        try:
            if decision == 'block_and_alert' and self.fraud_alert_topic:
                # Send fraud alert
                alert_message = {
                    'alert_type': 'fraud_detection',
                    'transaction_id': transaction_id,
                    'decision': decision,
                    'confidence': investigation_result['confidence'],
                    'summary': investigation_result.get('investigation_summary', {}).get('summary', 'High-risk transaction detected'),
                    'timestamp': investigation_result['timestamp']
                }

                self.sns.publish(
                    TopicArn=self.fraud_alert_topic,
                    Subject=f"FRAUD ALERT: Transaction {transaction_id}",
                    Message=json.dumps(alert_message, indent=2, default=str)
                )

                logger.info(f"Fraud alert sent for transaction {transaction_id}")

        except Exception as e:
            logger.error(f"Failed to route investigation result: {str(e)}")
            # Don't raise - routing failure shouldn't stop investigation processing

    def send_investigation_metrics(
        self,
        investigation_result: Dict,
        duration: float
    ) -> None:
        """Send investigation metrics to CloudWatch."""

        try:
            metric_data = [
                {
                    'MetricName': 'InvestigationDuration',
                    'Value': duration,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Decision', 'Value': investigation_result['decision']}
                    ]
                },
                {
                    'MetricName': 'InvestigationCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Decision', 'Value': investigation_result['decision']}
                    ]
                }
            ]

            if investigation_result['confidence']:
                metric_data.append({
                    'MetricName': 'InvestigationConfidence',
                    'Value': investigation_result['confidence'],
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'Decision', 'Value': investigation_result['decision']}
                    ]
                })

            self.cloudwatch.put_metric_data(
                Namespace='FinancialAnomalyDetection/Investigation',
                MetricData=metric_data
            )

        except Exception as e:
            logger.error(f"Failed to send investigation metrics: {str(e)}")
            # Don't raise - metrics failure shouldn't stop processing

    def delete_message(self, receipt_handle: str) -> None:
        """Delete processed message from SQS queue."""

        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )

        except ClientError as e:
            logger.error(f"Failed to delete message from queue: {str(e)}")
            # Don't raise - deletion failure is handled by SQS


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""

    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_requested = True


def main():
    """Main entry point for the investigation worker."""

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize and start worker
        worker = InvestigationWorker()
        worker.run()

    except Exception as e:
        logger.error(f"Worker initialization failed: {str(e)}")
        sys.exit(1)

    logger.info("Investigation worker stopped")


if __name__ == '__main__':
    main()