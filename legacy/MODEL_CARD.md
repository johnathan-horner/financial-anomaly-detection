# Model Card: Financial Transaction Anomaly Detection System

## Model Details
- Model: PyTorch Autoencoder Neural Network
- Framework: PyTorch + SageMaker
- Type: Unsupervised anomaly detection (autoencoder)
- Version: 1.2.0
- Owner: Johnathan Horner, Shoot It Analytics LLC

## Intended Use
- Primary: Detect anomalous financial transactions in real-time for fraud prevention
- Users: Financial institutions, payment processors, fraud investigation teams
- Out of scope: Not intended for credit scoring or investment decisions. Human review required for high-risk transactions

## Training Data
- Base: Synthetic transactions from `data/synthetic_generator.py` (seed 42): 100,000 normal transactions across 1,000 customers over a 90-day window, plus 3,387 injected anomalous rows (103,387 total)
- Features: Transaction amount, merchant category, location risk, customer patterns, temporal features
- Classes: Normal transactions vs. anomalous patterns (3.3% anomaly rate in generated data)

## Evaluation Metrics
Evaluation metrics will be published here once measured on a held-out split.

## Risk Scoring Thresholds
- Low Risk (<0.3): Auto-approve, no human review
- Medium Risk (0.3-0.7): Queue for analyst investigation
- High Risk (>0.7): Immediate alert, customer contact required

## Monitoring
- Real-time performance tracking via SageMaker Model Monitor
- Daily model drift detection on input feature distributions
- Alert thresholds for precision/recall degradation
- A/B testing framework for model updates

## Ethical Considerations
- Model should not discriminate based on protected characteristics
- Regular bias audits across demographic groups
- Transparent appeals process for false positives
- Privacy-preserving design with minimal data retention

## Compliance
- PCI DSS: Card data handling compliance
- SOX: Financial reporting controls
- GDPR: Right to explanation for automated decisions
- Fair Credit Reporting Act: Dispute resolution process