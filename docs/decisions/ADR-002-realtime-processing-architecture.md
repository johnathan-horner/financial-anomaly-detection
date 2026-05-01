# ADR-002: Kinesis + Lambda for Real-Time Processing

## Status: Accepted

## Context
We needed to design a real-time transaction processing architecture that could handle 100K+ transactions per day, provide sub-second fraud detection, and scale automatically during peak periods.

## Decision
We chose Amazon Kinesis Data Streams with Lambda functions for real-time transaction processing.

## Alternatives Considered
- **Direct API Gateway + Lambda**: Synchronous processing with immediate response
- **SQS + ECS**: Queue-based processing with containerized workers
- **Apache Kafka + Custom Workers**: Self-managed streaming platform
- **AWS Kinesis Analytics**: SQL-based stream processing

## Consequences

### Positive
- **Real-Time Processing**: Sub-second latency for fraud detection
- **Auto-Scaling**: Lambda scales automatically with transaction volume
- **Durability**: Kinesis provides ordered, durable message delivery
- **Cost Efficiency**: Pay-per-use Lambda billing vs always-on infrastructure
- **Integration**: Native AWS service integration with SageMaker and DynamoDB

### Negative
- **Lambda Limitations**: 15-minute timeout limits complex processing
- **Cold Starts**: Initial latency spikes during scaling events
- **Kinesis Costs**: Per-shard pricing can be expensive at high volumes

### Neutral
- **Monitoring**: CloudWatch integration for comprehensive observability
- **Error Handling**: DLQ patterns for failed transaction processing
- **Ordering**: Partition-based ordering ensures transaction sequence integrity

### Performance Characteristics
- **Throughput**: 1000 transactions/second per shard
- **Latency**: 200ms average processing time
- **Retention**: 24-hour replay capability for failed transactions