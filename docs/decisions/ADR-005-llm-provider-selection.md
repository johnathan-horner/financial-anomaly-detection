# ADR-005: Amazon Bedrock for LLM Reasoning

## Status: Accepted

## Context
We needed an LLM service for fraud investigation reasoning, customer communication generation, and complex pattern analysis that meets financial industry security and compliance requirements.

## Decision
We chose Amazon Bedrock with Claude models for LLM-powered investigation reasoning and communication generation.

## Alternatives Considered
- **OpenAI GPT-4**: Industry-leading LLM with strong reasoning capabilities
- **Anthropic Claude Direct**: Direct API access to Claude models
- **AWS SageMaker JumpStart**: Self-hosted open source models
- **Google Vertex AI**: Google's managed AI platform

## Consequences

### Positive
- **Security**: Data never leaves AWS environment, meets financial compliance
- **IAM Integration**: Native AWS authentication and authorization
- **Regional Control**: Data processing within specified AWS regions
- **SLA Guarantees**: Enterprise-grade availability and performance SLAs
- **Cost Transparency**: Integrated AWS billing with detailed usage tracking

### Negative
- **Model Limitations**: Restricted to Bedrock-supported models
- **Feature Lag**: New capabilities may arrive later than direct APIs
- **Vendor Lock-in**: Deeper dependency on AWS ecosystem

### Neutral
- **Performance**: Comparable latency to direct API calls (~2-3 seconds)
- **Quality**: Claude's reasoning quality excellent for fraud investigation
- **Rate Limits**: Similar throttling with enterprise escalation options

### Use Cases
- **Investigation Reasoning**: Complex fraud pattern analysis and decision support
- **Customer Communication**: Fraud alert messages and resolution correspondence
- **Risk Assessment**: Natural language explanations of fraud likelihood
- **Report Generation**: Automated investigation summaries for compliance

### Compliance Benefits
- **PCI DSS**: Compliant infrastructure for payment card data
- **SOX**: Audit trail and data governance capabilities
- **Privacy**: No model training on customer data, data residency control