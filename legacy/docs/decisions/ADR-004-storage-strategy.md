# ADR-004: DynamoDB for Transaction Storage

## Status: Accepted

## Context
We needed to select a database for storing transaction records, investigation results, and customer/merchant profiles that could handle high write throughput, fast lookups, and compliance requirements.

## Decision
We chose Amazon DynamoDB with multiple tables for transaction data, customer profiles, and merchant risk information.

## Alternatives Considered
- **Amazon RDS (PostgreSQL)**: Traditional relational database
- **Amazon DocumentDB**: MongoDB-compatible document database
- **Amazon Redshift**: Data warehouse for analytics workloads
- **Amazon S3 + Athena**: Object storage with query capabilities

## Consequences

### Positive
- **Write Performance**: Single-digit millisecond writes for high-volume transactions
- **Scalability**: Auto-scaling handles traffic spikes without capacity planning
- **Global Tables**: Multi-region replication for disaster recovery
- **Pay-per-Use**: On-demand billing scales with actual usage
- **Integration**: Native AWS service integration with Lambda and streams

### Negative
- **Query Limitations**: Limited query flexibility compared to SQL databases
- **Eventual Consistency**: Default eventual consistency may impact real-time reads
- **Cost Unpredictability**: On-demand pricing less predictable than provisioned

### Neutral
- **Backup/Recovery**: Automated backups with point-in-time recovery
- **Security**: Encryption at rest and in transit with IAM integration
- **Monitoring**: CloudWatch metrics for performance and capacity monitoring

### Table Design
- **Transactions**: Primary key on transaction_id, GSI on customer_id + timestamp
- **Customer Profiles**: Primary key on customer_id, includes spending patterns
- **Merchant Risk**: Primary key on merchant_id, includes fraud history
- **Investigations**: Primary key on investigation_id, tracks case status