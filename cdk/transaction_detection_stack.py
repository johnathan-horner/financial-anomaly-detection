"""
Core Infrastructure Stack for Financial Transaction Anomaly Detection

Deploys Kinesis Data Stream, DynamoDB tables, SageMaker endpoint,
S3 buckets, and Lambda functions for transaction scoring.
"""

from typing import Dict, Any
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_kinesis as kinesis,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_lambda as lambda_,
    aws_lambda_event_sources as lambda_event_sources,
    aws_iam as iam,
    aws_kms as kms,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_logs as logs,
)
from constructs import Construct


class CoreResources:
    """Container for core infrastructure resources to pass between stacks."""

    def __init__(self):
        self.kinesis_stream: kinesis.Stream = None
        self.transactions_table: dynamodb.Table = None
        self.customer_history_table: dynamodb.Table = None
        self.merchant_risk_table: dynamodb.Table = None
        self.investigations_table: dynamodb.Table = None
        self.s3_bucket: s3.Bucket = None
        self.sagemaker_endpoint: sagemaker.CfnEndpoint = None
        self.investigation_queue: sqs.Queue = None
        self.feedback_queue: sqs.Queue = None
        self.fraud_alert_topic: sns.Topic = None
        self.kms_key: kms.Key = None


class TransactionDetectionStack(Stack):
    """Core infrastructure stack for transaction anomaly detection."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.core_resources = CoreResources()

        # Create KMS key for encryption
        self.create_kms_key()

        # Create S3 bucket for data storage
        self.create_s3_bucket()

        # Create DynamoDB tables
        self.create_dynamodb_tables()

        # Create Kinesis stream
        self.create_kinesis_stream()

        # Create SQS queues
        self.create_sqs_queues()

        # Create SNS topics
        self.create_sns_topics()

        # Create SageMaker model and endpoint
        self.create_sagemaker_endpoint()

        # Create Lambda function for transaction scoring
        self.create_scoring_lambda()

    def create_kms_key(self) -> None:
        """Create KMS key for encryption at rest."""

        self.core_resources.kms_key = kms.Key(
            self, "FinancialAnomalyKMSKey",
            description="KMS key for Financial Anomaly Detection system",
            removal_policy=RemovalPolicy.DESTROY,
            enable_key_rotation=True,
        )

        # Add alias
        kms.Alias(
            self, "FinancialAnomalyKMSKeyAlias",
            alias_name="alias/financial-anomaly-detection",
            target_key=self.core_resources.kms_key
        )

    def create_s3_bucket(self) -> None:
        """Create S3 bucket for model artifacts and data storage."""

        self.core_resources.s3_bucket = s3.Bucket(
            self, "FinancialAnomalyDataBucket",
            bucket_name=f"financial-anomaly-data-{self.account}-{self.region}",
            versioning=True,
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.core_resources.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=Duration.days(30)
                ),
                s3.LifecycleRule(
                    id="ArchiveOldData",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ]
        )

    def create_dynamodb_tables(self) -> None:
        """Create DynamoDB tables for transaction data and investigations."""

        # Transactions table
        self.core_resources.transactions_table = dynamodb.Table(
            self, "TransactionsTable",
            table_name="transactions",
            partition_key=dynamodb.Attribute(
                name="transaction_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.core_resources.kms_key,
            point_in_time_recovery=True,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY
        )

        # Customer history table
        self.core_resources.customer_history_table = dynamodb.Table(
            self, "CustomerHistoryTable",
            table_name="customer-history",
            partition_key=dynamodb.Attribute(
                name="customer_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.core_resources.kms_key,
            point_in_time_recovery=True,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY
        )

        # Merchant risk table
        self.core_resources.merchant_risk_table = dynamodb.Table(
            self, "MerchantRiskTable",
            table_name="merchant-risk",
            partition_key=dynamodb.Attribute(
                name="merchant_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.core_resources.kms_key,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Investigations table
        self.core_resources.investigations_table = dynamodb.Table(
            self, "InvestigationsTable",
            table_name="investigations",
            partition_key=dynamodb.Attribute(
                name="transaction_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.core_resources.kms_key,
            point_in_time_recovery=True,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY
        )

        # Add GSI for customer history queries by date
        self.core_resources.customer_history_table.add_global_secondary_index(
            index_name="timestamp-index",
            partition_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            )
        )

    def create_kinesis_stream(self) -> None:
        """Create Kinesis Data Stream for real-time transaction ingestion."""

        self.core_resources.kinesis_stream = kinesis.Stream(
            self, "TransactionStream",
            stream_name="financial-transactions",
            shard_count=2,  # Start with 2 shards, can scale up
            retention_period=Duration.hours(24),
            encryption=kinesis.StreamEncryption.KMS,
            encryption_key=self.core_resources.kms_key
        )

    def create_sqs_queues(self) -> None:
        """Create SQS queues for investigation workflow."""

        # Dead letter queue for failed investigations
        investigation_dlq = sqs.Queue(
            self, "InvestigationDLQ",
            queue_name="investigation-dlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.core_resources.kms_key
        )

        # Investigation queue
        self.core_resources.investigation_queue = sqs.Queue(
            self, "InvestigationQueue",
            queue_name="investigation-queue",
            visibility_timeout=Duration.minutes(5),
            message_retention_period=Duration.hours(12),
            receive_message_wait_time=Duration.seconds(20),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.core_resources.kms_key,
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=investigation_dlq
            )
        )

        # Feedback queue for analyst input
        self.core_resources.feedback_queue = sqs.Queue(
            self, "FeedbackQueue",
            queue_name="feedback-queue",
            visibility_timeout=Duration.minutes(2),
            message_retention_period=Duration.days(7),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=self.core_resources.kms_key
        )

    def create_sns_topics(self) -> None:
        """Create SNS topics for alerts and notifications."""

        self.core_resources.fraud_alert_topic = sns.Topic(
            self, "FraudAlertTopic",
            topic_name="fraud-alerts",
            display_name="Financial Fraud Alerts",
            master_key=self.core_resources.kms_key
        )

    def create_sagemaker_endpoint(self) -> None:
        """Create SageMaker model and endpoint for anomaly detection."""

        # IAM role for SageMaker
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ]
        )

        # Grant access to S3 bucket
        self.core_resources.s3_bucket.grant_read_write(sagemaker_role)

        # SageMaker model (placeholder - will be updated during deployment)
        model = sagemaker.CfnModel(
            self, "AnomalyDetectionModel",
            model_name="financial-anomaly-detector",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image="246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                model_data_url=f"s3://{self.core_resources.s3_bucket.bucket_name}/model-artifacts/placeholder.tar.gz"
            )
        )

        # SageMaker endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "AnomalyDetectionEndpointConfig",
            endpoint_config_name="financial-anomaly-detector-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=model.model_name,
                    initial_instance_count=1,
                    instance_type="ml.t2.medium",
                    initial_variant_weight=1.0
                )
            ]
        )

        endpoint_config.add_dependency(model)

        # SageMaker endpoint
        self.core_resources.sagemaker_endpoint = sagemaker.CfnEndpoint(
            self, "AnomalyDetectionEndpoint",
            endpoint_name="financial-anomaly-detector",
            endpoint_config_name=endpoint_config.endpoint_config_name
        )

        self.core_resources.sagemaker_endpoint.add_dependency(endpoint_config)

    def create_scoring_lambda(self) -> None:
        """Create Lambda function for transaction scoring."""

        # IAM role for Lambda
        lambda_role = iam.Role(
            self, "ScoringLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # Grant permissions
        self.core_resources.transactions_table.grant_read_write_data(lambda_role)
        self.core_resources.investigation_queue.grant_send_messages(lambda_role)

        lambda_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "sagemaker:InvokeEndpoint"
            ],
            resources=[
                f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{self.core_resources.sagemaker_endpoint.endpoint_name}"
            ]
        ))

        lambda_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "cloudwatch:PutMetricData"
            ],
            resources=["*"]
        ))

        # Lambda function
        scoring_lambda = lambda_.Function(
            self, "TransactionScoringLambda",
            function_name="financial-anomaly-scoring",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/score"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=lambda_role,
            environment={
                "SAGEMAKER_ENDPOINT": self.core_resources.sagemaker_endpoint.endpoint_name,
                "INVESTIGATION_QUEUE_URL": self.core_resources.investigation_queue.queue_url,
                "TRANSACTIONS_TABLE": self.core_resources.transactions_table.table_name,
                "AUTO_APPROVE_THRESHOLD": "0.3"
            },
            log_retention=logs.RetentionDays.ONE_MONTH
        )

        # Add Kinesis event source
        scoring_lambda.add_event_source(
            lambda_event_sources.KinesisEventSource(
                stream=self.core_resources.kinesis_stream,
                starting_position=lambda_.StartingPosition.LATEST,
                batch_size=100,
                max_batching_window=Duration.seconds(5),
                retry_attempts=3
            )
        )