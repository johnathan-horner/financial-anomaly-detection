"""
LangChain Agents using Amazon Bedrock for Financial Transaction Analysis

Implements specialized agents for pattern analysis and investigation summary
generation using Claude 3 Sonnet via Amazon Bedrock.
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PatternAnalysisResult(BaseModel):
    """Structured output for pattern analysis."""

    risk_level: str = Field(description="Risk level: low, medium, high")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    patterns: List[str] = Field(description="List of detected anomaly patterns")
    behavioral_baseline: Dict[str, Any] = Field(description="Customer behavioral baseline analysis")
    anomaly_indicators: List[Dict[str, Any]] = Field(description="Specific anomaly indicators found")
    geographic_analysis: Dict[str, Any] = Field(description="Geographic transaction analysis")
    velocity_analysis: Dict[str, Any] = Field(description="Transaction velocity analysis")
    merchant_analysis: Dict[str, Any] = Field(description="Merchant category analysis")


class InvestigationSummaryResult(BaseModel):
    """Structured output for investigation summary."""

    recommendation: str = Field(description="Recommended action: auto_approve, hold_for_review, block_and_alert")
    confidence: float = Field(description="Confidence in recommendation 0-1", ge=0, le=1)
    summary: str = Field(description="Natural language summary of investigation")
    key_findings: List[str] = Field(description="Key findings from investigation")
    evidence: List[Dict[str, Any]] = Field(description="Supporting evidence for decision")
    false_positive_likelihood: float = Field(description="Likelihood this is a false positive 0-1", ge=0, le=1)
    fraud_likelihood: float = Field(description="Likelihood this is fraud 0-1", ge=0, le=1)
    next_steps: List[str] = Field(description="Recommended next steps")


class PatternAnalysisAgent:
    """LangChain agent for analyzing transaction patterns."""

    def __init__(self, bedrock_client):
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )

        self.output_parser = PydanticOutputParser(pydantic_object=PatternAnalysisResult)
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> PromptTemplate:
        """Create the pattern analysis prompt template."""

        template = """You are an expert financial fraud analyst specializing in transaction pattern analysis.

Your task is to analyze a flagged financial transaction against customer history and merchant data to identify specific anomaly patterns.

TRANSACTION TO ANALYZE:
{flagged_transaction}

CUSTOMER BEHAVIORAL BASELINE:
Transaction History (last 90 days): {customer_history}

MERCHANT INFORMATION:
{merchant_risk}

ANOMALY SCORE: {anomaly_score} (higher = more anomalous)

ANALYSIS FRAMEWORK:
Analyze the flagged transaction across these dimensions:

1. BEHAVIORAL BASELINE
   - Calculate customer's typical transaction amounts (mean, median, ranges)
   - Identify usual merchant categories and time patterns
   - Determine normal geographic patterns and spending velocity

2. ANOMALY PATTERN DETECTION
   - Amount deviation: How does this transaction compare to typical amounts?
   - Geographic anomaly: Is this location unusual for the customer?
   - Velocity spike: Are there multiple transactions in short timeframe?
   - Time anomaly: Is this an unusual time of day/week for the customer?
   - New merchant category: Is this a category the customer never uses?
   - Merchant risk: Does the merchant have fraud flags or high risk rating?

3. FRAUD VS. FALSE POSITIVE INDICATORS
   Fraud indicators:
   - Impossible geographic velocity (transactions minutes apart, hundreds of miles away)
   - Massive amount spikes (10x+ typical spending)
   - Multiple rapid transactions at different merchants
   - Transactions at high-risk merchants
   - Patterns consistent with card testing or account takeover

   False positive indicators:
   - Legitimate customer behavior changes (vacation, major purchase)
   - One-time larger but reasonable purchases
   - New merchant category that makes sense for customer
   - Geographic changes with reasonable travel time

4. CONFIDENCE ASSESSMENT
   - High confidence (>0.8): Clear pattern, strong supporting evidence
   - Medium confidence (0.4-0.8): Some indicators, mixed evidence
   - Low confidence (<0.4): Unclear pattern, limited evidence

CRITICAL ANALYSIS RULES:
- Be conservative with fraud classification - false positives harm customer experience
- Consider legitimate reasons for unusual behavior
- Weight multiple weak signals higher than single strong signal
- Account for customer's historical risk profile
- Consider merchant legitimacy and business context

{format_instructions}

Provide your analysis focusing on factual patterns and evidence-based reasoning."""

        return PromptTemplate(
            template=template,
            input_variables=[
                "flagged_transaction",
                "customer_history",
                "merchant_risk",
                "anomaly_score"
            ],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def analyze_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction patterns using the LLM agent."""

        try:
            # Prepare input data
            flagged_transaction = json.dumps(context["flagged_transaction"], indent=2)
            customer_history = json.dumps(context["customer_history"][:20], indent=2)  # Limit history for token efficiency
            merchant_risk = json.dumps(context["merchant_risk"], indent=2)
            anomaly_score = context["anomaly_score"]

            # Format prompt
            prompt_text = self.prompt.format(
                flagged_transaction=flagged_transaction,
                customer_history=customer_history,
                merchant_risk=merchant_risk,
                anomaly_score=anomaly_score
            )

            # Call LLM
            messages = [HumanMessage(content=prompt_text)]
            response = self.llm.invoke(messages)

            # Parse structured output
            result = self.output_parser.parse(response.content)

            return result.dict()

        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            # Return fallback result
            return {
                "risk_level": "medium",
                "confidence": 0.3,
                "patterns": ["analysis_error"],
                "behavioral_baseline": {},
                "anomaly_indicators": [{"type": "processing_error", "description": str(e)}],
                "geographic_analysis": {},
                "velocity_analysis": {},
                "merchant_analysis": {}
            }


class InvestigationSummaryAgent:
    """LangChain agent for generating investigation summaries."""

    def __init__(self, bedrock_client):
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "max_tokens": 3000,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )

        self.output_parser = PydanticOutputParser(pydantic_object=InvestigationSummaryResult)
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> PromptTemplate:
        """Create the investigation summary prompt template."""

        template = """You are a senior fraud investigation analyst creating a final summary for a flagged financial transaction.

INVESTIGATION DATA:

Transaction Details:
{transaction}

Anomaly Score: {anomaly_score}

Customer History Analysis:
{customer_history_summary}

Merchant Risk Assessment:
{merchant_risk}

Pattern Analysis Results:
{pattern_analysis}

Processing Errors (if any):
{errors}

INVESTIGATION SUMMARY TASK:

Create a comprehensive investigation summary that will be used by:
1. Automated systems for routing decisions
2. Human analysts for manual review
3. Audit trails for regulatory compliance

Your summary must include:

1. RECOMMENDATION (choose one):
   - "auto_approve": Clear false positive, safe to approve automatically
   - "hold_for_review": Requires human analyst review before decision
   - "block_and_alert": High fraud probability, block transaction and alert

2. CONFIDENCE SCORE (0-1):
   - 0.9-1.0: Extremely confident in recommendation
   - 0.7-0.9: High confidence, strong evidence
   - 0.5-0.7: Moderate confidence, mixed signals
   - 0.3-0.5: Low confidence, unclear indicators
   - 0.0-0.3: Very uncertain, requires immediate human review

3. INVESTIGATION SUMMARY:
   Clear, concise narrative explaining what was investigated and what was found.
   Use plain language that both technical and non-technical stakeholders can understand.

4. KEY FINDINGS:
   Bullet points of the most important discoveries from the investigation.
   Focus on actionable insights and clear evidence.

5. SUPPORTING EVIDENCE:
   Specific data points, calculations, or patterns that support your recommendation.
   Include relevant comparisons to customer baseline behavior.

6. FRAUD/FALSE POSITIVE LIKELIHOOD:
   Quantify the probability this is actual fraud vs. a false positive.
   Base this on concrete evidence, not speculation.

DECISION FRAMEWORK:

AUTO-APPROVE when:
- Clear legitimate customer behavior (vacation, major purchase with context)
- Minor deviations from baseline with reasonable explanations
- Low anomaly score with weak supporting fraud indicators

HOLD FOR REVIEW when:
- Mixed signals requiring human judgment
- Moderate anomaly indicators without clear fraud patterns
- New customer behavior patterns that need verification
- Technical issues that prevented complete analysis

BLOCK AND ALERT when:
- Clear fraud patterns (impossible velocity, massive amount spikes)
- Multiple strong fraud indicators
- Known merchant fraud risks
- Customer account compromise indicators

Remember: The cost of false positives (blocking legitimate transactions) must be balanced against fraud prevention. Be conservative but decisive.

{format_instructions}"""

        return PromptTemplate(
            template=template,
            input_variables=[
                "transaction",
                "anomaly_score",
                "customer_history_summary",
                "merchant_risk",
                "pattern_analysis",
                "errors"
            ],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

    def generate_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investigation summary using the LLM agent."""

        try:
            # Prepare input data
            transaction = json.dumps(context["transaction"], indent=2)
            anomaly_score = context["anomaly_score"]

            # Summarize customer history
            customer_history = context.get("customer_history", [])
            history_summary = self._summarize_customer_history(customer_history)

            merchant_risk = json.dumps(context.get("merchant_risk", {}), indent=2)
            pattern_analysis = json.dumps(context.get("pattern_analysis", {}), indent=2)
            errors = json.dumps(context.get("errors", []), indent=2)

            # Format prompt
            prompt_text = self.prompt.format(
                transaction=transaction,
                anomaly_score=anomaly_score,
                customer_history_summary=history_summary,
                merchant_risk=merchant_risk,
                pattern_analysis=pattern_analysis,
                errors=errors
            )

            # Call LLM
            messages = [HumanMessage(content=prompt_text)]
            response = self.llm.invoke(messages)

            # Parse structured output
            result = self.output_parser.parse(response.content)

            return result.dict()

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            # Return fallback result
            return {
                "recommendation": "hold_for_review",
                "confidence": 0.1,
                "summary": f"Investigation summary could not be generated due to error: {str(e)}",
                "key_findings": ["Processing error occurred"],
                "evidence": [],
                "false_positive_likelihood": 0.5,
                "fraud_likelihood": 0.5,
                "next_steps": ["Manual review required due to system error"]
            }

    def _summarize_customer_history(self, history: List[Dict]) -> str:
        """Create a concise summary of customer transaction history."""

        if not history:
            return "No customer history available"

        amounts = [txn.get("amount", 0) for txn in history]
        categories = [txn.get("merchant_category", "unknown") for txn in history]

        summary = f"""
Customer Transaction History Summary ({len(history)} transactions):
- Transaction count: {len(history)}
- Average amount: ${np.mean(amounts):.2f}
- Median amount: ${np.median(amounts):.2f}
- Amount range: ${min(amounts):.2f} - ${max(amounts):.2f}
- Common merchant categories: {list(set(categories))[:5]}
- Transaction frequency: {len(history)/90:.1f} transactions per day (90-day average)
"""

        return summary.strip()