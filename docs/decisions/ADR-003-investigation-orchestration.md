# ADR-003: LangGraph for Investigation Orchestration

## Status: Accepted

## Context
We needed to orchestrate complex fraud investigation workflows that involve customer pattern analysis, merchant risk assessment, external data enrichment, and decision routing based on multiple factors and confidence levels.

## Decision
We chose LangGraph to orchestrate multi-agent fraud investigation workflows with conditional routing and parallel analysis.

## Alternatives Considered
- **AWS Step Functions**: AWS-native workflow orchestration service
- **Simple Sequential Chain**: Basic agent chaining without complex routing
- **Apache Airflow**: Data pipeline orchestration framework
- **Custom State Machine**: Home-built workflow engine

## Consequences

### Positive
- **Parallel Processing**: Customer and merchant analysis run simultaneously
- **Conditional Routing**: Smart routing based on risk scores and confidence levels
- **State Management**: Maintains investigation context across agent interactions
- **LLM Integration**: Seamless integration with Bedrock for reasoning tasks
- **Visual Debugging**: Graph visualization for complex investigation flows

### Negative
- **Learning Curve**: New framework requires team training
- **Deployment Complexity**: Containerized deployment vs simpler Lambda functions
- **Memory Overhead**: Stateful execution requires more resources

### Neutral
- **Performance**: Similar to Step Functions for I/O-bound investigation tasks
- **Cost**: ECS Fargate costs comparable to Step Function execution
- **Monitoring**: Custom metrics vs native Step Function observability

### Investigation Flow
- **Risk Triage**: Route based on initial anomaly score
- **Parallel Analysis**: Customer patterns + merchant risk assessment
- **Evidence Synthesis**: Combine findings for final decision
- **Action Routing**: Queue assignment or immediate escalation