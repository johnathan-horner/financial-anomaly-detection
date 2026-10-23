"""
LangGraph Investigation Agent for Financial Transaction Anomalies

Implements a state graph that orchestrates a multi-step investigation
of flagged transactions using various data sources and analysis tools.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from chains.bedrock_agents import (
    PatternAnalysisAgent,
    InvestigationSummaryAgent
)
from chains.tools import (
    CustomerHistoryTool,
    MerchantRiskTool,
    TransactionScorer
)

logger = logging.getLogger(__name__)


@dataclass
class InvestigationState:
    """State object for the investigation workflow."""

    # Input transaction
    transaction_id: str
    customer_id: str
    transaction_data: Dict[str, Any]
    anomaly_score: float

    # Investigation data
    customer_history: Optional[List[Dict]] = None
    merchant_risk_data: Optional[Dict] = None
    pattern_analysis: Optional[Dict] = None
    investigation_summary: Optional[Dict] = None

    # Decision routing
    decision: Optional[str] = None
    confidence: Optional[float] = None

    # Metadata
    investigation_started: Optional[datetime] = None
    investigation_completed: Optional[datetime] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.investigation_started is None:
            self.investigation_started = datetime.utcnow()


class InvestigationGraph:
    """LangGraph state machine for transaction investigation."""

    def __init__(
        self,
        bedrock_client,
        customer_history_tool: CustomerHistoryTool,
        merchant_risk_tool: MerchantRiskTool,
        transaction_scorer: TransactionScorer,
        pattern_agent: PatternAnalysisAgent,
        summary_agent: InvestigationSummaryAgent
    ):
        self.bedrock_client = bedrock_client
        self.customer_history_tool = customer_history_tool
        self.merchant_risk_tool = merchant_risk_tool
        self.transaction_scorer = transaction_scorer
        self.pattern_agent = pattern_agent
        self.summary_agent = summary_agent

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the investigation state graph."""

        workflow = StateGraph(InvestigationState)

        # Add nodes
        workflow.add_node("pull_customer_history", self.pull_customer_history)
        workflow.add_node("merchant_check", self.merchant_check)
        workflow.add_node("pattern_analysis", self.pattern_analysis)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("route_decision", self.route_decision)

        # Set entry point
        workflow.set_entry_point("pull_customer_history")

        # Add edges
        workflow.add_edge("pull_customer_history", "merchant_check")
        workflow.add_edge("merchant_check", "pattern_analysis")

        # Conditional edge from pattern_analysis
        workflow.add_conditional_edges(
            "pattern_analysis",
            self.should_generate_summary,
            {
                "generate_summary": "generate_summary",
                "route_decision": "route_decision"
            }
        )

        workflow.add_edge("generate_summary", "route_decision")
        workflow.add_edge("route_decision", END)

        return workflow.compile()

    def pull_customer_history(self, state: InvestigationState) -> InvestigationState:
        """Pull customer transaction history from DynamoDB."""

        logger.info(f"Pulling history for customer {state.customer_id}")

        try:
            # Get last 90 days of transactions
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)

            history = self.customer_history_tool.get_customer_history(
                customer_id=state.customer_id,
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )

            state.customer_history = history
            logger.info(f"Retrieved {len(history)} historical transactions")

        except Exception as e:
            error_msg = f"Failed to retrieve customer history: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.customer_history = []

        return state

    def merchant_check(self, state: InvestigationState) -> InvestigationState:
        """Check merchant risk profile and flags."""

        merchant_id = state.transaction_data.get('merchant_id')
        logger.info(f"Checking merchant risk for {merchant_id}")

        try:
            merchant_data = self.merchant_risk_tool.get_merchant_risk(merchant_id)
            state.merchant_risk_data = merchant_data
            logger.info(f"Merchant risk level: {merchant_data.get('risk_level', 'unknown')}")

        except Exception as e:
            error_msg = f"Failed to retrieve merchant data: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.merchant_risk_data = {}

        return state

    def pattern_analysis(self, state: InvestigationState) -> InvestigationState:
        """Analyze transaction patterns using LLM agent."""

        logger.info(f"Performing pattern analysis for transaction {state.transaction_id}")

        try:
            # Prepare analysis context
            analysis_context = {
                "flagged_transaction": state.transaction_data,
                "anomaly_score": state.anomaly_score,
                "customer_history": state.customer_history,
                "merchant_risk": state.merchant_risk_data
            }

            # Run pattern analysis
            pattern_result = self.pattern_agent.analyze_patterns(analysis_context)
            state.pattern_analysis = pattern_result

            logger.info(f"Pattern analysis completed. Risk level: {pattern_result.get('risk_level')}")

        except Exception as e:
            error_msg = f"Pattern analysis failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.pattern_analysis = {"risk_level": "medium", "patterns": [], "confidence": 0.5}

        return state

    def should_generate_summary(self, state: InvestigationState) -> str:
        """Determine whether to generate full summary or route directly."""

        if not state.pattern_analysis:
            return "route_decision"

        risk_level = state.pattern_analysis.get('risk_level', 'medium')
        confidence = state.pattern_analysis.get('confidence', 0.5)

        # Skip summary for clear false positives
        if risk_level == 'low' and confidence > 0.8:
            logger.info("Clear false positive detected, skipping summary generation")
            return "route_decision"

        return "generate_summary"

    def generate_summary(self, state: InvestigationState) -> InvestigationState:
        """Generate investigation summary using LLM agent."""

        logger.info(f"Generating investigation summary for transaction {state.transaction_id}")

        try:
            # Prepare summary context
            summary_context = {
                "transaction": state.transaction_data,
                "anomaly_score": state.anomaly_score,
                "customer_history": state.customer_history,
                "merchant_risk": state.merchant_risk_data,
                "pattern_analysis": state.pattern_analysis,
                "errors": state.errors
            }

            # Generate summary
            summary_result = self.summary_agent.generate_summary(summary_context)
            state.investigation_summary = summary_result

            logger.info(f"Investigation summary generated. Recommendation: {summary_result.get('recommendation')}")

        except Exception as e:
            error_msg = f"Summary generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)

            # Fallback summary
            state.investigation_summary = {
                "recommendation": "hold_for_review",
                "confidence": 0.3,
                "summary": "Investigation completed with errors. Manual review required.",
                "key_findings": [],
                "evidence": []
            }

        return state

    def route_decision(self, state: InvestigationState) -> InvestigationState:
        """Make final routing decision based on investigation results."""

        logger.info(f"Making routing decision for transaction {state.transaction_id}")

        try:
            decision_factors = {
                "anomaly_score": state.anomaly_score,
                "pattern_risk": state.pattern_analysis.get('risk_level', 'medium') if state.pattern_analysis else 'medium',
                "pattern_confidence": state.pattern_analysis.get('confidence', 0.5) if state.pattern_analysis else 0.5,
                "merchant_risk": state.merchant_risk_data.get('risk_level', 'medium') if state.merchant_risk_data else 'medium',
                "summary_recommendation": state.investigation_summary.get('recommendation', 'hold_for_review') if state.investigation_summary else 'hold_for_review'
            }

            decision, confidence = self._make_decision(decision_factors)

            state.decision = decision
            state.confidence = confidence
            state.investigation_completed = datetime.utcnow()

            logger.info(f"Final decision: {decision} (confidence: {confidence:.3f})")

        except Exception as e:
            error_msg = f"Decision routing failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.decision = "hold_for_review"
            state.confidence = 0.0

        return state

    def _make_decision(self, factors: Dict[str, Any]) -> tuple[str, float]:
        """Make routing decision based on multiple factors."""

        # Decision logic
        anomaly_score = factors["anomaly_score"]
        pattern_risk = factors["pattern_risk"]
        pattern_confidence = factors["pattern_confidence"]
        merchant_risk = factors["merchant_risk"]
        summary_recommendation = factors["summary_recommendation"]

        # Score mapping
        risk_scores = {"low": 0.1, "medium": 0.5, "high": 0.9}

        pattern_score = risk_scores.get(pattern_risk, 0.5)
        merchant_score = risk_scores.get(merchant_risk, 0.5)

        # Weighted decision score
        decision_score = (
            anomaly_score * 0.4 +
            pattern_score * 0.3 +
            merchant_score * 0.2 +
            (1.0 if summary_recommendation == "block" else 0.5 if summary_recommendation == "hold_for_review" else 0.0) * 0.1
        )

        # Apply pattern confidence
        adjusted_score = decision_score * pattern_confidence + 0.5 * (1 - pattern_confidence)

        # Decision thresholds
        if adjusted_score < 0.3:
            return "auto_approve", pattern_confidence
        elif adjusted_score > 0.7:
            return "block_and_alert", pattern_confidence
        else:
            return "hold_for_review", pattern_confidence

    def investigate(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete investigation workflow."""

        # Initialize state
        initial_state = InvestigationState(
            transaction_id=transaction_data["transaction_id"],
            customer_id=transaction_data["customer_id"],
            transaction_data=transaction_data,
            anomaly_score=transaction_data.get("anomaly_score", 0.5)
        )

        logger.info(f"Starting investigation for transaction {initial_state.transaction_id}")

        # Run graph
        final_state = self.graph.invoke(initial_state)

        # Convert to output format
        result = {
            "transaction_id": final_state.transaction_id,
            "decision": final_state.decision,
            "confidence": final_state.confidence,
            "investigation_duration": (
                final_state.investigation_completed - final_state.investigation_started
            ).total_seconds() if final_state.investigation_completed else None,
            "customer_history_count": len(final_state.customer_history) if final_state.customer_history else 0,
            "merchant_risk_level": final_state.merchant_risk_data.get('risk_level') if final_state.merchant_risk_data else None,
            "pattern_analysis": final_state.pattern_analysis,
            "investigation_summary": final_state.investigation_summary,
            "errors": final_state.errors,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"Investigation completed for {initial_state.transaction_id}. Decision: {result['decision']}")

        return result


class InvestigationGraphFactory:
    """Factory for creating investigation graphs with proper dependencies."""

    @staticmethod
    def create_graph(
        bedrock_client,
        dynamodb_client,
        sagemaker_client
    ) -> InvestigationGraph:
        """Create a fully configured investigation graph."""

        # Initialize tools
        customer_history_tool = CustomerHistoryTool(dynamodb_client)
        merchant_risk_tool = MerchantRiskTool(dynamodb_client)
        transaction_scorer = TransactionScorer(sagemaker_client)

        # Initialize agents
        pattern_agent = PatternAnalysisAgent(bedrock_client)
        summary_agent = InvestigationSummaryAgent(bedrock_client)

        # Create graph
        return InvestigationGraph(
            bedrock_client=bedrock_client,
            customer_history_tool=customer_history_tool,
            merchant_risk_tool=merchant_risk_tool,
            transaction_scorer=transaction_scorer,
            pattern_agent=pattern_agent,
            summary_agent=summary_agent
        )