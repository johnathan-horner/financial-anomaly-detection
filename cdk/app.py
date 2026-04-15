#!/usr/bin/env python3
"""
AWS CDK App for Financial Transaction Anomaly Detection System

Main CDK application that orchestrates all infrastructure stacks
for the production-grade financial fraud detection system.
"""

import os
from aws_cdk import App, Environment

from transaction_detection_stack import TransactionDetectionStack
from agent_stack import AgentStack
from api_stack import ApiStack
from monitoring_stack import MonitoringStack


def main():
    """Initialize and deploy all CDK stacks."""

    app = App()

    # Get environment configuration
    account = os.environ.get('CDK_DEFAULT_ACCOUNT')
    region = os.environ.get('CDK_DEFAULT_REGION', 'us-east-1')

    env = Environment(account=account, region=region)

    # Deploy order is important due to dependencies

    # 1. Core infrastructure (Kinesis, DynamoDB, SageMaker, S3)
    core_stack = TransactionDetectionStack(
        app, "FinancialAnomalyDetectionCore",
        env=env,
        description="Core infrastructure for financial transaction anomaly detection"
    )

    # 2. Investigation agents (ECS, SQS)
    agent_stack = AgentStack(
        app, "FinancialAnomalyDetectionAgents",
        core_resources=core_stack.core_resources,
        env=env,
        description="Investigation agents and processing infrastructure"
    )

    # 3. API Gateway and Lambda functions
    api_stack = ApiStack(
        app, "FinancialAnomalyDetectionAPI",
        core_resources=core_stack.core_resources,
        agent_resources=agent_stack.agent_resources,
        env=env,
        description="API Gateway and dashboard endpoints"
    )

    # 4. Monitoring, alerts, and retraining
    monitoring_stack = MonitoringStack(
        app, "FinancialAnomalyDetectionMonitoring",
        core_resources=core_stack.core_resources,
        api_resources=api_stack.api_resources,
        env=env,
        description="Monitoring, alerting, and automated retraining"
    )

    # Add dependencies
    agent_stack.add_dependency(core_stack)
    api_stack.add_dependency(core_stack)
    api_stack.add_dependency(agent_stack)
    monitoring_stack.add_dependency(core_stack)
    monitoring_stack.add_dependency(api_stack)

    # Add global tags
    for stack in [core_stack, agent_stack, api_stack, monitoring_stack]:
        stack.tags.set_tag("Project", "FinancialAnomalyDetection")
        stack.tags.set_tag("Environment", os.environ.get('ENVIRONMENT', 'dev'))
        stack.tags.set_tag("Owner", "MLOps-Team")
        stack.tags.set_tag("CostCenter", "AI-Platform")

    app.synth()


if __name__ == "__main__":
    main()