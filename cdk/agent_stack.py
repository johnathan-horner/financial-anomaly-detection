"""
Agent Infrastructure Stack for Financial Transaction Investigation

Deploys ECS Fargate cluster and services for running the LangGraph
investigation agents that process flagged transactions.
"""

from typing import Dict, Any
from aws_cdk import (
    Stack,
    Duration,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_logs as logs,
    aws_ecr as ecr,
    aws_applicationautoscaling as autoscaling,
)
from constructs import Construct


class AgentResources:
    """Container for agent infrastructure resources."""

    def __init__(self):
        self.vpc: ec2.Vpc = None
        self.cluster: ecs.Cluster = None
        self.service: ecs.FargateService = None
        self.task_definition: ecs.FargateTaskDefinition = None


class AgentStack(Stack):
    """ECS Fargate infrastructure for investigation agents."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        core_resources,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.core_resources = core_resources
        self.agent_resources = AgentResources()

        # Create VPC
        self.create_vpc()

        # Create ECS cluster
        self.create_ecs_cluster()

        # Create task definition
        self.create_task_definition()

        # Create ECS service
        self.create_ecs_service()

        # Setup auto scaling
        self.setup_auto_scaling()

    def create_vpc(self) -> None:
        """Create VPC with private subnets for ECS tasks."""

        self.agent_resources.vpc = ec2.Vpc(
            self, "InvestigationVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

    def create_ecs_cluster(self) -> None:
        """Create ECS Fargate cluster."""

        self.agent_resources.cluster = ecs.Cluster(
            self, "InvestigationCluster",
            cluster_name="financial-anomaly-investigation",
            vpc=self.agent_resources.vpc,
            container_insights=True
        )

    def create_task_definition(self) -> None:
        """Create Fargate task definition for investigation worker."""

        # IAM role for task execution
        execution_role = iam.Role(
            self, "InvestigationTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ]
        )

        # IAM role for task
        task_role = iam.Role(
            self, "InvestigationTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        # Grant permissions to task role
        self.grant_task_permissions(task_role)

        # Create task definition
        self.agent_resources.task_definition = ecs.FargateTaskDefinition(
            self, "InvestigationTaskDefinition",
            family="financial-anomaly-investigation",
            cpu=1024,
            memory_limit_mib=2048,
            execution_role=execution_role,
            task_role=task_role
        )

        # ECR repository for container image
        ecr_repository = ecr.Repository(
            self, "InvestigationECRRepo",
            repository_name="financial-anomaly-investigation",
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10
                )
            ]
        )

        # Add container
        container = self.agent_resources.task_definition.add_container(
            "investigation-worker",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repository, "latest"),
            environment={
                "INVESTIGATION_QUEUE_URL": self.core_resources.investigation_queue.queue_url,
                "INVESTIGATIONS_TABLE": self.core_resources.investigations_table.table_name,
                "FRAUD_ALERT_TOPIC_ARN": self.core_resources.fraud_alert_topic.topic_arn,
                "MAX_MESSAGES_PER_BATCH": "10",
                "VISIBILITY_TIMEOUT": "300",
                "WAIT_TIME_SECONDS": "20"
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="investigation-worker",
                log_retention=logs.RetentionDays.ONE_MONTH
            )
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8080,
                protocol=ecs.Protocol.TCP
            )
        )

    def grant_task_permissions(self, task_role: iam.Role) -> None:
        """Grant necessary permissions to the ECS task role."""

        # DynamoDB permissions
        for table in [
            self.core_resources.transactions_table,
            self.core_resources.customer_history_table,
            self.core_resources.merchant_risk_table,
            self.core_resources.investigations_table
        ]:
            table.grant_read_write_data(task_role)

        # SQS permissions
        self.core_resources.investigation_queue.grant_consume_messages(task_role)

        # SNS permissions
        self.core_resources.fraud_alert_topic.grant_publish(task_role)

        # Bedrock permissions
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "bedrock:InvokeModel"
            ],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            ]
        ))

        # CloudWatch metrics
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "cloudwatch:PutMetricData"
            ],
            resources=["*"]
        ))

    def create_ecs_service(self) -> None:
        """Create ECS Fargate service."""

        self.agent_resources.service = ecs.FargateService(
            self, "InvestigationService",
            service_name="financial-anomaly-investigation",
            cluster=self.agent_resources.cluster,
            task_definition=self.agent_resources.task_definition,
            desired_count=1,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            enable_logging=True
        )

    def setup_auto_scaling(self) -> None:
        """Setup auto scaling for the ECS service based on SQS queue depth."""

        # Create scalable target
        scalable_target = autoscaling.ScalableTarget(
            self, "InvestigationServiceScalableTarget",
            service_namespace=autoscaling.ServiceNamespace.ECS,
            resource_id=f"service/{self.agent_resources.cluster.cluster_name}/{self.agent_resources.service.service_name}",
            scalable_dimension="ecs:service:DesiredCount",
            min_capacity=1,
            max_capacity=10
        )

        # Scale based on SQS queue depth
        scalable_target.scale_on_metric(
            "QueueDepthScaling",
            metric=self.core_resources.investigation_queue.metric_approximate_number_of_visible_messages(),
            scaling_steps=[
                autoscaling.ScalingInterval(lower=0, upper=10, change=0),
                autoscaling.ScalingInterval(lower=10, upper=50, change=1),
                autoscaling.ScalingInterval(lower=50, change=2)
            ],
            adjustment_type=autoscaling.AdjustmentType.CHANGE_IN_CAPACITY
        )