# Real-Time Financial Transaction Anomaly Detection System

![CI](https://github.com/johnathan-horner/financial-anomaly-detection/actions/workflows/ci.yml/badge.svg)

A production-grade, cloud-native AI system for detecting financial transaction anomalies using PyTorch, LangGraph + LangChain for agentic investigation orchestration, and Amazon Bedrock for LLM inference.

## Architecture

![AWS Architecture](docs/Transaction_Anomaly_Detection_AWS_Architecture.png)

## 🏗️ System Architecture

### AWS Architecture Overview

```mermaid
flowchart TB
    subgraph "Ingestion Layer"
        A[Transaction Stream] --> B[Amazon Kinesis Data Stream]
    end

    subgraph "Scoring Layer"
        B --> C[Lambda: Score Transaction]
        C --> D[SageMaker Endpoint<br/>PyTorch Autoencoder]
        C --> E{Score >= 0.3?}
    end

    subgraph "Storage Layer"
        E -->|No| F[DynamoDB: Transactions<br/>Auto-Approved]
        E -->|Yes| G[SQS: Investigation Queue]
        H[DynamoDB: Customer History]
        I[DynamoDB: Merchant Risk]
        J[DynamoDB: Investigations]
    end

    subgraph "Investigation Layer"
        G --> K[ECS Fargate: Investigation Agents]
        K --> L[LangGraph State Machine]
        L --> M[Amazon Bedrock<br/>Claude 3 Sonnet]
        L --> H
        L --> I
        L --> J
    end

    subgraph "Decision & Alerting"
        L --> N{Investigation Decision}
        N -->|Auto-Approve| F
        N -->|Hold for Review| O[Analyst Review Queue]
        N -->|Block & Alert| P[SNS: Fraud Alerts]
    end

    subgraph "API & Dashboard"
        Q[API Gateway] --> R[Lambda: Dashboard API]
        R --> F
        R --> J
        S[Web Dashboard] --> Q
    end

    subgraph "ML Pipeline"
        T[EventBridge: Weekly Schedule] --> U[Lambda: Retrain Trigger]
        U --> V[SageMaker Training Job]
        V --> W[S3: Model Artifacts]
        W --> D
    end

    style B fill:#ff9999
    style D fill:#99ccff
    style M fill:#99ff99
    style K fill:#ffcc99
```

### LangGraph Investigation Workflow

```mermaid
stateDiagram-v2
    [*] --> ScoreTransaction: Flagged Transaction
    ScoreTransaction --> PullCustomerHistory: Score >= 0.3

    PullCustomerHistory --> MerchantCheck
    MerchantCheck --> PatternAnalysis

    PatternAnalysis --> DecisionPoint: Analyze Patterns
    DecisionPoint --> GenerateSummary: Complex Case
    DecisionPoint --> RouteDecision: Clear Pattern

    GenerateSummary --> RouteDecision

    RouteDecision --> AutoApprove: Clear False Positive
    RouteDecision --> HoldReview: Mixed Signals
    RouteDecision --> BlockAlert: High Fraud Risk

    AutoApprove --> [*]
    HoldReview --> [*]
    BlockAlert --> [*]

    note right of PatternAnalysis
        Uses Amazon Bedrock Claude 3
        - Behavioral baseline analysis
        - Geographic anomaly detection
        - Velocity pattern analysis
        - Merchant risk assessment
    end note

    note right of GenerateSummary
        Natural language investigation report
        - Key findings
        - Supporting evidence
        - Confidence scores
        - Recommended actions
    end note
```

## 🚀 Model Architecture

### PyTorch Autoencoder for Anomaly Detection

The system uses a deep autoencoder neural network trained exclusively on normal transaction patterns:

```python
# Model Architecture
Input Layer (15 features)
    ↓
Encoder: [64] → [32] → [16] (bottleneck)
    ↓
Decoder: [32] → [64] → [15] (reconstruction)
    ↓
Reconstruction Error → Anomaly Score
```

**Key Features:**
- **Amount features**: Log-transformed transaction amounts
- **Temporal features**: Sin/cos encoded hour/day patterns
- **Geographic features**: Distance from customer's home location
- **Behavioral features**: Category preferences, velocity metrics
- **Statistical features**: Z-scores, rolling window aggregations

**Training Strategy:**
- Train only on normal transactions (unsupervised)
- Anomaly detection via reconstruction error
- Threshold calibration using precision-recall curves
- Target: <1% false positive rate

## 🧠 LangChain + Bedrock Integration

### Pattern Analysis Agent
```python
class PatternAnalysisAgent:
    """Analyzes transaction patterns using Claude 3 Sonnet"""

    def analyze_patterns(self, context):
        # Behavioral baseline calculation
        # Geographic impossibility detection
        # Velocity spike analysis
        # Merchant category risk assessment
        return structured_analysis
```

### Investigation Summary Agent
```python
class InvestigationSummaryAgent:
    """Generates investigation reports using Claude 3 Sonnet"""

    def generate_summary(self, investigation_data):
        # Natural language summary
        # Evidence compilation
        # Risk assessment
        # Action recommendations
        return investigation_report
```

## 📊 Performance Metrics & Thresholds

### Model Performance (Validation Set)
Evaluation metrics will be published here once measured on a held-out split.

### Threshold Calibration
- **Auto-Approve Threshold**: 0.3 (anomaly score)
- **Alert Threshold**: 0.7 (immediate block)
- **Calibration Method**: Precision-recall curve analysis
- **Target FPR**: <1% (business requirement)

### Investigation Performance
Evaluation metrics will be published here once measured on a held-out split.

## 🏛️ Design Decisions & Rationale

### Why Kinesis over SQS for Ingestion?
- **Real-time processing**: Low-latency streaming ingestion
- **Throughput**: 2 provisioned shards (`shard_count=2` in CDK), scalable for burst capacity
- **Replay capability**: 24-hour retention for error recovery
- **Partitioning**: Distribute load across Lambda instances

### Why ECS Fargate over Lambda for Agents?
- **Long-running workflows**: LangGraph investigations chain multiple LLM calls
- **Memory requirements**: LangChain + Bedrock clients need 2GB+ RAM
- **State management**: Complex investigation state across multiple LLM calls
- **Cost efficiency**: More economical for sustained processing

### Why Amazon Bedrock over Direct APIs?
- **No API key management**: Native AWS IAM integration
- **Enterprise compliance**: Data stays within AWS boundaries
- **Rate limiting**: Built-in throttling and retry mechanisms
- **Cost predictability**: AWS pricing model vs. token-based billing

### Threshold Selection Rationale
The 0.3 auto-approve threshold was selected through:
1. **Business impact analysis**: Cost of false positives vs. fraud losses
2. **ROC curve optimization**: Maximize true positive rate at 1% FPR
3. **Operational constraints**: Human review capacity limits
4. **Regulatory requirements**: Audit trail and explainability needs

## 📋 SR 11-7 Regulatory Compliance

### Model Documentation
- **Model inventory**: Complete model lineage and versioning
- **Performance monitoring**: Continuous validation metrics
- **Bias testing**: Demographic fairness across customer segments
- **Explainability**: Feature importance and decision reasoning

### Validation Framework
- **Backtesting**: Performance on historical fraud cases
- **Stress testing**: Model behavior under extreme scenarios
- **Benchmark comparison**: Performance vs. rule-based systems
- **Independent validation**: Third-party model audit capability

### Monitoring & Alerting
- **Data drift detection**: Population stability index monitoring
- **Performance degradation**: F1-score alert thresholds
- **Operational metrics**: Latency, error rates, throughput
- **Model staleness**: Automated retraining triggers

### Audit Trail
- **Decision logging**: Every investigation result stored
- **Model versioning**: Git-based model artifact tracking
- **Change management**: CDK infrastructure-as-code
- **Access controls**: IAM roles and KMS encryption

## 🚦 Deployment Instructions

### Prerequisites
- AWS CLI configured with appropriate permissions
- Python 3.11+
- Node.js 18+ (for CDK)
- Docker (for container builds)

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd financial-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
# Generate 100K normal transactions + 2K injected anomaly events
# (velocity spikes expand to 3-6 rows each: 3,387 anomalous rows with seed 42)
python data/synthetic_generator.py

# Verify data generation
ls -la data/generated/
```

### 3. Train Initial Model
```bash
# Train autoencoder locally
python model/autoencoder.py

# Verify model artifacts
ls -la model/artifacts/
```

### 4. Deploy Infrastructure
```bash
# Navigate to CDK directory
cd cdk

# Install CDK dependencies
npm install

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy all stacks
cdk deploy --all --require-approval never

# Note the outputs (endpoints, queue URLs, etc.)
```

### 5. Build and Push Container
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Build and push investigation worker
docker build -t financial-anomaly-investigation -f ecs/Dockerfile .
docker tag financial-anomaly-investigation:latest <account>.dkr.ecr.us-east-1.amazonaws.com/financial-anomaly-investigation:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/financial-anomaly-investigation:latest
```

### 6. Upload Model to SageMaker
```bash
# Package model artifacts
python scripts/package_model.py

# Upload to S3 and update SageMaker endpoint
python scripts/deploy_model.py
```

### 7. Seed Database Tables
```bash
# Load customer profiles and merchant data
python scripts/seed_database.py
```

### 8. Verify Deployment
```bash
# Run integration tests
python tests/integration/test_end_to_end.py

# Check dashboard
curl https://<api-gateway-url>/dashboard/metrics
```

## 🔧 Testing Strategy

### Unit Tests
```bash
# Run unit tests for all components
pytest tests/unit/ -v

# Test coverage report
pytest tests/unit/ --cov=. --cov-report=html
```

### Integration Tests
```bash
# Test complete transaction flow
pytest tests/integration/test_transaction_flow.py

# Test investigation workflow
pytest tests/integration/test_investigation_agent.py

# Test API endpoints
pytest tests/integration/test_dashboard_api.py
```

### Load Testing
```bash
# Simulate transaction volume
python tests/load/generate_transaction_load.py --rate 1000 --duration 300

# Monitor performance metrics
python tests/load/monitor_performance.py
```

## 🗑️ Tear-Down Instructions

### Complete Cleanup
```bash
# Delete all CDK stacks
cdk destroy --all --force

# Clean up S3 bucket contents
aws s3 rm s3://<bucket-name> --recursive

# Remove ECR images
aws ecr batch-delete-image --repository-name financial-anomaly-investigation --image-ids imageTag=latest
```

### Partial Cleanup (Keep Data)
```bash
# Scale down ECS service to 0
aws ecs update-service --cluster financial-anomaly-investigation --service financial-anomaly-investigation --desired-count 0

# Delete SageMaker endpoint
aws sagemaker delete-endpoint --endpoint-name financial-anomaly-detector
```

## 🎯 Future Enhancements

### Model Improvements
- **Ensemble methods**: Combine multiple anomaly detection techniques
- **Graph neural networks**: Analyze transaction networks and relationships
- **Real-time feature engineering**: Dynamic customer behavior baselines
- **Federated learning**: Multi-institution fraud pattern sharing

### Investigation Enhancements
- **Multi-modal analysis**: Incorporate device fingerprinting and location data
- **External data integration**: Credit bureau and identity verification services
- **Advanced reasoning**: Multi-step investigation workflows with tool use
- **Explanation generation**: Natural language explanations for decisions

### Operational Improvements
- **A/B testing framework**: Compare model versions and investigation strategies
- **Synthetic fraud generation**: Adversarial training data creation
- **Real-time dashboard**: Live fraud detection monitoring
- **Analyst feedback loop**: Continuous learning from human decisions

## Architecture Decisions

See [docs/decisions/](docs/decisions/) for architecture decision records explaining why each service and pattern was chosen.

## Model Documentation

See [MODEL_CARD.md](MODEL_CARD.md) for detailed model specifications, performance metrics, and compliance information.

## API Documentation

Run the FastAPI server with `uvicorn api:app --reload` and visit `/docs` for interactive Swagger documentation.

## 🔌 MCP Server Integration

The Financial Anomaly Detection System includes a Model Context Protocol (MCP) server that exposes fraud detection functionality through a standardized interface for AI assistant integration.

### Available Tools

- `detect_transaction_anomaly()` - Analyze transactions for anomalous patterns
- `investigate_customer_pattern()` - Investigate customer historical behavior
- `analyze_merchant_risk()` - Assess merchant fraud risk profiles
- `get_investigation_queue()` - Check current investigation workload
- `generate_fraud_alert()` - Create alerts for high-risk transactions
- `get_system_metrics()` - Retrieve performance and operational metrics

### Available Resources

- `model://autoencoder` - PyTorch autoencoder model information
- `data://risk_thresholds` - Risk scoring and routing configuration

### Available Prompts

- `fraud_investigation` - Complete fraud investigation workflow
- `batch_analysis` - Batch transaction analysis for patterns and trends

Run with: `python mcp_server.py`

## CI/CD Pipeline

GitHub Actions automatically runs on push/PR to main:
- Python 3.11 setup and dependency installation
- Code linting with flake8 (max line length 120)
- Import testing for core modules

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for full pipeline configuration.

---

**Built with ❤️ for production-grade financial security**

This system represents enterprise-ready financial fraud detection with modern AI/ML practices, regulatory compliance, and operational excellence.