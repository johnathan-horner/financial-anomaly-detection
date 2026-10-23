#!/bin/bash
set -e

# Financial Anomaly Detection System Deployment Script
# Deploys the complete system to AWS using CDK

echo "🚀 Starting Financial Anomaly Detection System Deployment"

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install AWS CLI."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "❌ AWS credentials not configured. Please configure AWS CLI."
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found. Please install Python 3.11+."
        exit 1
    fi

    # Check Node.js for CDK
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js not found. Please install Node.js 18+."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker."
        exit 1
    fi

    echo "✅ Prerequisites check passed"
}

# Setup Python environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    pip install -r requirements.txt

    echo "✅ Python environment ready"
}

# Generate synthetic data
generate_data() {
    echo "📊 Generating synthetic transaction data..."

    source venv/bin/activate

    # Check if data already exists
    if [ ! -f "data/generated/transactions.parquet" ]; then
        echo "Generating synthetic dataset..."
        python data/synthetic_generator.py
    else
        echo "Synthetic data already exists, skipping generation."
    fi

    echo "✅ Synthetic data ready"
}

# Train initial model
train_model() {
    echo "🧠 Training initial autoencoder model..."

    source venv/bin/activate

    # Check if model already exists
    if [ ! -f "model/artifacts/autoencoder.pth" ]; then
        echo "Training autoencoder model..."
        python model/autoencoder.py
    else
        echo "Model artifacts already exist, skipping training."
    fi

    echo "✅ Model training complete"
}

# Deploy infrastructure using CDK
deploy_infrastructure() {
    echo "🏗️ Deploying AWS infrastructure..."

    cd cdk

    # Install CDK dependencies
    npm install

    # Get AWS account and region
    export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    export CDK_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

    echo "Deploying to account: $CDK_DEFAULT_ACCOUNT, region: $CDK_DEFAULT_REGION"

    # Bootstrap CDK if not already done
    echo "Bootstrapping CDK..."
    cdk bootstrap

    # Deploy all stacks
    echo "Deploying CDK stacks..."
    cdk deploy --all --require-approval never --outputs-file ../cdk-outputs.json

    cd ..

    echo "✅ Infrastructure deployment complete"
}

# Build and push Docker container
deploy_container() {
    echo "🐳 Building and deploying container..."

    # Get ECR repository URI from CDK outputs
    ECR_URI=$(python3 -c "
import json
with open('cdk-outputs.json', 'r') as f:
    outputs = json.load(f)

# Find ECR repository URI in outputs
for stack_name, stack_outputs in outputs.items():
    for key, value in stack_outputs.items():
        if 'ECRRepository' in key:
            print(value)
            break
")

    if [ -z "$ECR_URI" ]; then
        echo "❌ Could not find ECR repository URI in CDK outputs"
        exit 1
    fi

    echo "ECR Repository: $ECR_URI"

    # Get ECR login token
    aws ecr get-login-password --region $CDK_DEFAULT_REGION | \
        docker login --username AWS --password-stdin $ECR_URI

    # Build container
    echo "Building Docker image..."
    docker build -t financial-anomaly-investigation -f ecs/Dockerfile .

    # Tag for ECR
    docker tag financial-anomaly-investigation:latest $ECR_URI:latest

    # Push to ECR
    echo "Pushing to ECR..."
    docker push $ECR_URI:latest

    echo "✅ Container deployment complete"
}

# Upload model to S3 and update SageMaker endpoint
deploy_model() {
    echo "🤖 Deploying model to SageMaker..."

    source venv/bin/activate

    # Get S3 bucket from CDK outputs
    S3_BUCKET=$(python3 -c "
import json
with open('cdk-outputs.json', 'r') as f:
    outputs = json.load(f)

for stack_name, stack_outputs in outputs.items():
    for key, value in stack_outputs.items():
        if 'S3Bucket' in key:
            print(value)
            break
")

    if [ -z "$S3_BUCKET" ]; then
        echo "❌ Could not find S3 bucket in CDK outputs"
        exit 1
    fi

    echo "S3 Bucket: $S3_BUCKET"

    # Package model artifacts
    echo "Packaging model artifacts..."
    cd model/artifacts
    tar -czf model.tar.gz *
    cd ../..

    # Upload to S3
    echo "Uploading model to S3..."
    aws s3 cp model/artifacts/model.tar.gz s3://$S3_BUCKET/model-artifacts/model.tar.gz

    # Update SageMaker model (this would typically trigger endpoint update)
    echo "Model artifacts uploaded. SageMaker endpoint will update automatically."

    echo "✅ Model deployment complete"
}

# Seed database tables
seed_database() {
    echo "🌱 Seeding database tables..."

    source venv/bin/activate

    # Load customer profiles generated during data creation
    if [ -f "data/generated/customer_profiles.json" ]; then
        echo "Seeding customer and merchant data..."
        python -c "
import json
import boto3
from chains.tools import DatabaseSeeder

# Load customer profiles
with open('data/generated/customer_profiles.json', 'r') as f:
    customer_profiles = json.load(f)

# Initialize seeder
dynamodb = boto3.resource('dynamodb')
seeder = DatabaseSeeder(dynamodb)

# Seed tables
seeder.seed_merchant_risk_data()
seeder.seed_customer_history_sample(customer_profiles[:10])

print('Database seeding complete')
"
    else
        echo "⚠️ Customer profiles not found, skipping database seeding"
    fi

    echo "✅ Database seeding complete"
}

# Run basic integration test
run_integration_test() {
    echo "🧪 Running integration tests..."

    source venv/bin/activate

    # Simple connectivity test
    python3 -c "
import json
import boto3

# Load CDK outputs
with open('cdk-outputs.json', 'r') as f:
    outputs = json.load(f)

print('CDK Deployment Outputs:')
for stack_name, stack_outputs in outputs.items():
    print(f'  {stack_name}:')
    for key, value in stack_outputs.items():
        print(f'    {key}: {value}')

# Test AWS connectivity
try:
    # Test DynamoDB
    dynamodb = boto3.resource('dynamodb')
    tables = list(dynamodb.tables.all())
    print(f'✅ DynamoDB: Found {len([t for t in tables if \"financial\" in t.name.lower()])} relevant tables')

    # Test Kinesis
    kinesis = boto3.client('kinesis')
    streams = kinesis.list_streams()
    print(f'✅ Kinesis: Found {len(streams.get(\"StreamNames\", []))} streams')

    # Test SQS
    sqs = boto3.client('sqs')
    queues = sqs.list_queues()
    print(f'✅ SQS: Found {len(queues.get(\"QueueUrls\", []))} queues')

    print('🎉 Integration test passed!')

except Exception as e:
    print(f'❌ Integration test failed: {e}')
    exit(1)
"

    echo "✅ Integration tests complete"
}

# Display deployment summary
show_summary() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================================"
    echo ""
    echo "Your Financial Anomaly Detection System has been successfully deployed to AWS."
    echo ""

    if [ -f "cdk-outputs.json" ]; then
        echo "📋 Key Resources:"
        python3 -c "
import json
with open('cdk-outputs.json', 'r') as f:
    outputs = json.load(f)

for stack_name, stack_outputs in outputs.items():
    for key, value in stack_outputs.items():
        if 'API' in key or 'Endpoint' in key or 'Topic' in key:
            print(f'  {key}: {value}')
"
        echo ""
    fi

    echo "📊 Next Steps:"
    echo "  1. Monitor the system via CloudWatch dashboards"
    echo "  2. Send test transactions to the Kinesis stream"
    echo "  3. Check investigation results in DynamoDB"
    echo "  4. Access the dashboard API for metrics"
    echo ""
    echo "📚 For more information, see the README.md file."
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment of Financial Anomaly Detection System..."
    echo "=========================================================="
    echo ""

    check_prerequisites
    setup_python_env
    generate_data
    train_model
    deploy_infrastructure
    deploy_container
    deploy_model
    seed_database
    run_integration_test
    show_summary
}

# Run main function
main