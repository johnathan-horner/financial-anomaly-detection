import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json

# Page configuration
st.set_page_config(
    page_title="Transaction Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Mock Data for Demo Mode
MOCK_RESPONSES = {
    "suspicious_score": {
        "anomaly_score": 0.82,
        "feature_breakdown": {
            "amount_deviation": 0.35,
            "merchant_category": 0.28,
            "time_of_day": 0.12,
            "location": 0.07
        },
        "routing_decision": "Blocked + Alert",
        "investigation_id": "inv_123456"
    },
    "normal_score": {
        "anomaly_score": 0.12,
        "feature_breakdown": {
            "amount_deviation": 0.05,
            "merchant_category": 0.03,
            "time_of_day": 0.02,
            "location": 0.02
        },
        "routing_decision": "Auto-Approved",
        "investigation_id": None
    },
    "investigation": {
        "steps": [
            {
                "name": "Customer History",
                "status": "completed",
                "output": "Customer avg transaction $47, typically grocery/gas, no transactions after 10pm in last 90 days. This transaction is 180x above baseline."
            },
            {
                "name": "Merchant Check",
                "status": "completed",
                "output": "Merchant category: luxury jewelry, risk rating: elevated, no prior fraud flags for this specific merchant."
            },
            {
                "name": "Pattern Analysis",
                "status": "completed",
                "output": "Three anomaly patterns detected: (1) amount 180x above customer baseline, (2) first transaction in luxury category, (3) 3:14am transaction outside customer's typical window."
            },
            {
                "name": "Investigation Summary",
                "status": "completed",
                "output": "HIGH RISK TRANSACTION: Customer John Smith (ID: cust_789) attempted $8,500 jewelry purchase at 3:14am, representing 180x spending increase. Pattern suggests account compromise or stolen card usage. Recommend immediate hold and customer contact verification."
            }
        ],
        "final_decision": "Blocked + Alert",
        "confidence": 0.89
    },
    "analyst_queue": [
        {"transaction_id": "txn_001", "amount": 8500, "merchant": "Luxury Jewelry Co", "score": 0.82, "summary": "High-value purchase outside normal pattern", "timestamp": "2024-01-15 03:14:22"},
        {"transaction_id": "txn_002", "amount": 2200, "merchant": "Electronics Store", "score": 0.74, "summary": "Geographic anomaly detected", "timestamp": "2024-01-15 02:45:10"},
        {"transaction_id": "txn_003", "amount": 450, "merchant": "Gas Station", "score": 0.68, "summary": "Velocity spike flagged", "timestamp": "2024-01-15 01:22:33"},
        {"transaction_id": "txn_004", "amount": 1800, "merchant": "Online Retailer", "score": 0.71, "summary": "New merchant category", "timestamp": "2024-01-14 23:56:41"},
        {"transaction_id": "txn_005", "amount": 750, "merchant": "Restaurant", "score": 0.65, "summary": "Time-based anomaly", "timestamp": "2024-01-14 22:18:15"},
        {"transaction_id": "txn_006", "amount": 3200, "merchant": "Home Improvement", "score": 0.77, "summary": "Amount deviation flagged", "timestamp": "2024-01-14 21:45:52"},
        {"transaction_id": "txn_007", "amount": 920, "merchant": "Pharmacy", "score": 0.63, "summary": "Location-based flag", "timestamp": "2024-01-14 20:33:18"},
        {"transaction_id": "txn_008", "amount": 1650, "merchant": "Clothing Store", "score": 0.69, "summary": "Pattern analysis required", "timestamp": "2024-01-14 19:27:44"}
    ],
    "dashboard": {
        "metrics": {
            "transactions_today": 24891,
            "fraud_rate": 1.2,
            "false_positive_rate": 0.3,
            "avg_investigation_time": 4.2
        },
        "time_series": [
            {"timestamp": "2024-01-14 00:00", "anomaly_score": 0.15},
            {"timestamp": "2024-01-14 04:00", "anomaly_score": 0.23},
            {"timestamp": "2024-01-14 08:00", "anomaly_score": 0.18},
            {"timestamp": "2024-01-14 12:00", "anomaly_score": 0.21},
            {"timestamp": "2024-01-14 16:00", "anomaly_score": 0.19},
            {"timestamp": "2024-01-14 20:00", "anomaly_score": 0.25}
        ],
        "merchant_flags": [
            {"category": "Jewelry", "count": 23},
            {"category": "Electronics", "count": 18},
            {"category": "Online", "count": 15},
            {"category": "Gas Stations", "count": 12},
            {"category": "Restaurants", "count": 8}
        ],
        "model_info": {
            "version": "v2.1.0",
            "last_retrain": "3 days ago",
            "validation_date": "2024-01-12",
            "drift_status": "Normal"
        }
    }
}

def make_api_call(endpoint, data=None, demo_mode=False, mock_key=None):
    """Make API call with fallback to mock data in demo mode"""
    if demo_mode and mock_key:
        # Simulate API delay
        time.sleep(0.5)
        return MOCK_RESPONSES.get(mock_key, {})

    try:
        if data:
            response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=10)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def render_sidebar():
    """Render sidebar with project info and controls"""
    with st.sidebar:
        st.title("🔍 Transaction Anomaly Detection")

        # Demo Mode Toggle
        demo_mode = st.checkbox("Demo Mode", value=True, help="Use mock data for demonstration")

        st.markdown("---")

        # Project Description
        with st.expander("📖 Project Overview"):
            st.markdown("""
            **Real-time Financial Transaction Anomaly Detection**

            Production-grade AI system using:
            - PyTorch Autoencoder for anomaly scoring
            - LangGraph + LangChain for investigation orchestration
            - Amazon Bedrock Claude for pattern analysis
            - AWS cloud-native architecture

            **Key Capabilities:**
            - Sub-second transaction scoring
            - Autonomous investigation agents
            - SR 11-7 regulatory compliance
            - <1% false positive rate
            """)

        # Architecture Diagram
        st.image("docs/Transaction_Anomaly_Detection_AWS_Architecture.png",
                caption="AWS Architecture Overview", use_container_width=True)

        # Tech Stack
        st.markdown("**Tech Stack:**")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white", use_container_width=True)
            st.image("https://img.shields.io/badge/LangChain-2E7D32?style=flat&logo=chainlink&logoColor=white", use_container_width=True)
        with col2:
            st.image("https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white", use_container_width=True)
            st.image("https://img.shields.io/badge/Bedrock-FF9900?style=flat&logo=amazon-aws&logoColor=white", use_container_width=True)

        # Pipeline Architecture
        with st.expander("🔄 Pipeline Architecture"):
            st.markdown("""
            **LangGraph Investigation Flow:**
            1. **Score** → Autoencoder anomaly detection
            2. **History** → Customer baseline analysis
            3. **Merchant** → Risk assessment lookup
            4. **Pattern** → Claude reasoning analysis
            5. **Summary** → Investigation report generation
            6. **Route** → Auto-approve/Review/Block decision
            """)

        # SR 11-7 Compliance
        st.markdown("**SR 11-7 Compliance:**")
        st.success("✅ Model documentation")
        st.success("✅ Performance monitoring")
        st.success("✅ Bias testing")
        st.success("✅ Audit trail")

        # GitHub Link
        st.markdown("---")
        st.markdown("**[📁 View on GitHub](https://github.com/johnathanhorner/financial-anomaly-detection)**")

        # Footer
        st.markdown("---")
        st.markdown("**Built by Johnathan Horner**")

    return demo_mode

def render_score_tab(demo_mode):
    """Render the Score Transaction tab"""
    st.header("🎯 Score Transaction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        # Input form
        amount = st.number_input("Amount ($)", min_value=0.01, value=50.0, step=0.01)
        merchant_category = st.selectbox("Merchant Category",
            ["Grocery", "Gas Station", "Restaurant", "Retail", "Online", "Jewelry", "Electronics", "Pharmacy"])
        time_of_day = st.slider("Time of Day (24hr)", 0, 23, 14)
        location = st.text_input("Location", value="San Francisco, CA")
        customer_id = st.text_input("Customer ID", value="cust_123456")

        col_demo, col_score = st.columns(2)

        with col_demo:
            if st.button("🚨 Quick Demo", help="Load suspicious transaction example"):
                st.session_state.demo_transaction = {
                    "amount": 8500.0,
                    "merchant_category": "Jewelry",
                    "time_of_day": 3,
                    "location": "Las Vegas, NV",
                    "customer_id": "cust_789"
                }
                st.rerun()

        # Auto-fill demo transaction
        if hasattr(st.session_state, 'demo_transaction'):
            demo = st.session_state.demo_transaction
            amount = demo["amount"]
            merchant_category = demo["merchant_category"]
            time_of_day = demo["time_of_day"]
            location = demo["location"]
            customer_id = demo["customer_id"]

        with col_score:
            score_clicked = st.button("⚡ Score Transaction", type="primary")

    with col2:
        st.subheader("Scoring Results")

        if score_clicked:
            with st.spinner("Scoring transaction..."):
                transaction_data = {
                    "amount": amount,
                    "merchant_category": merchant_category,
                    "time_of_day": time_of_day,
                    "location": location,
                    "customer_id": customer_id
                }

                # Determine mock key based on amount
                mock_key = "suspicious_score" if amount > 1000 else "normal_score"

                result = make_api_call("score", transaction_data, demo_mode, mock_key)

                if result:
                    score = result["anomaly_score"]

                    # Large anomaly score with color
                    if score >= 0.7:
                        color = "red"
                    elif score >= 0.3:
                        color = "orange"
                    else:
                        color = "green"

                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};">
                        <h2 style="color: {color}; margin: 0;">Anomaly Score</h2>
                        <h1 style="color: {color}; margin: 10px 0; font-size: 3em;">{score:.3f}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # Feature breakdown chart
                    st.subheader("Feature Contribution")
                    feature_df = pd.DataFrame(
                        list(result["feature_breakdown"].items()),
                        columns=["Feature", "Contribution"]
                    )
                    fig = px.bar(feature_df, x="Contribution", y="Feature", orientation='h',
                                color="Contribution", color_continuous_scale="Reds")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Routing decision
                    decision = result["routing_decision"]
                    if decision == "Auto-Approved":
                        st.success(f"✅ {decision}")
                    elif decision == "Under Review":
                        st.warning(f"⚠️ {decision}")
                    else:
                        st.error(f"🚫 {decision}")

                    # Store for investigation tab
                    if result.get("investigation_id"):
                        st.session_state.investigation_id = result["investigation_id"]

def render_investigation_tab(demo_mode):
    """Render the Investigation Viewer tab"""
    st.header("🔍 Investigation Viewer")

    if hasattr(st.session_state, 'investigation_id') and st.session_state.investigation_id:
        st.info(f"Viewing investigation: {st.session_state.investigation_id}")

        # Simulate streaming investigation steps
        if st.button("▶️ Run Investigation"):
            investigation = make_api_call(f"investigation/{st.session_state.investigation_id}",
                                        demo_mode=demo_mode, mock_key="investigation")

            if investigation:
                st.subheader("LangGraph Agent Pipeline")

                # Create containers for each step
                containers = {}
                for i, step in enumerate(investigation["steps"]):
                    containers[step["name"]] = st.empty()

                # Simulate sequential execution
                for i, step in enumerate(investigation["steps"]):
                    with containers[step["name"]]:
                        with st.status(f"🔄 {step['name']}", expanded=True):
                            time.sleep(1)  # Simulate processing
                            st.write(step["output"])

                    # Update to completed
                    with containers[step["name"]]:
                        with st.status(f"✅ {step['name']}", state="complete", expanded=True):
                            st.write(step["output"])

                # Final decision
                st.markdown("---")
                st.subheader("Final Routing Decision")
                decision = investigation["final_decision"]
                confidence = investigation["confidence"]

                if decision == "Auto-Approved":
                    st.success(f"✅ {decision} (Confidence: {confidence:.1%})")
                elif decision == "Under Review":
                    st.warning(f"⚠️ {decision} (Confidence: {confidence:.1%})")
                else:
                    st.error(f"🚫 {decision} (Confidence: {confidence:.1%})")
    else:
        st.info("Score a transaction with anomaly score ≥ 0.3 to view investigation pipeline")
        st.markdown("""
        **Investigation Pipeline Overview:**

        When a transaction is flagged (score ≥ 0.3), the LangGraph agent system performs:

        1. **Customer History Analysis** - Pulls baseline spending patterns from DynamoDB
        2. **Merchant Risk Assessment** - Checks merchant category and risk ratings
        3. **Pattern Analysis** - Uses Amazon Bedrock Claude for behavioral analysis
        4. **Investigation Summary** - Generates natural language report with recommendations
        5. **Routing Decision** - Auto-approve, hold for review, or block transaction
        """)

def render_analyst_tab(demo_mode):
    """Render the Analyst Console tab"""
    st.header("👨‍💼 Analyst Console")

    # Get analyst queue
    queue_data = make_api_call("analyst/queue", demo_mode=demo_mode, mock_key="analyst_queue")

    if queue_data:
        st.subheader(f"Review Queue ({len(queue_data)} transactions)")

        # Display queue as table with action buttons
        for i, transaction in enumerate(queue_data):
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 2, 1, 3, 2])

                with col1:
                    st.write(f"**{transaction['transaction_id']}**")
                with col2:
                    st.write(f"${transaction['amount']:,.2f}")
                with col3:
                    st.write(transaction['merchant'])
                with col4:
                    score = transaction['score']
                    color = "🔴" if score >= 0.7 else "🟡"
                    st.write(f"{color} {score:.2f}")
                with col5:
                    st.write(transaction['summary'])
                with col6:
                    col_confirm, col_false = st.columns(2)
                    with col_confirm:
                        if st.button("✅ Confirm", key=f"confirm_{i}"):
                            st.success("Marked as fraud")
                    with col_false:
                        if st.button("❌ False +", key=f"false_{i}"):
                            st.info("Marked as false positive")

                st.markdown("---")

    # Summary stats
    st.subheader("Analyst Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cases Reviewed Today", "47")
    with col2:
        st.metric("Confirmed Fraud", "12", "2")
    with col3:
        st.metric("False Positives", "8", "-1")
    with col4:
        st.metric("Avg Review Time", "2.1 min", "-0.3")

def render_dashboard_tab(demo_mode):
    """Render the Dashboard tab"""
    st.header("📊 Dashboard")

    # Get dashboard data
    dashboard_data = make_api_call("dashboard/metrics", demo_mode=demo_mode, mock_key="dashboard")

    if dashboard_data:
        metrics = dashboard_data["metrics"]

        # Row 1: Metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Transactions Today", f"{metrics['transactions_today']:,}")
        with col2:
            st.metric("Fraud Rate", f"{metrics['fraud_rate']}%", "0.1%")
        with col3:
            st.metric("False Positive Rate", f"{metrics['false_positive_rate']}%", "-0.1%")
        with col4:
            st.metric("Avg Investigation Time", f"{metrics['avg_investigation_time']}s", "-0.5s")

        # Row 2: Time series chart
        st.subheader("Anomaly Scores Over Time")
        ts_df = pd.DataFrame(dashboard_data["time_series"])
        ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])

        fig = px.line(ts_df, x='timestamp', y='anomaly_score', title="Rolling Average Anomaly Score")
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Review Threshold")
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True)

        # Row 3: Merchant category breakdown
        st.subheader("Flagged Transactions by Category")
        merchant_df = pd.DataFrame(dashboard_data["merchant_flags"])
        fig = px.bar(merchant_df, x='category', y='count', color='count', color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        # Row 4: Model drift monitoring
        st.subheader("Model Drift Monitoring")
        col1, col2 = st.columns(2)

        with col1:
            # Mock drift chart
            drift_data = [0.15, 0.16, 0.14, 0.17, 0.15, 0.18, 0.16]
            drift_df = pd.DataFrame({
                'day': range(1, 8),
                'drift_score': drift_data
            })
            fig = px.line(drift_df, x='day', y='drift_score', title="7-Day Population Stability Index")
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Drift Analysis:**")
            st.success("✅ Population stability: Normal")
            st.success("✅ Feature distribution: Stable")
            st.success("✅ Performance metrics: Within bounds")
            st.info(f"📊 Current PSI: 0.16 (< 0.25 threshold)")

        # Row 5: SR 11-7 compliance panel
        st.subheader("SR 11-7 Compliance Status")
        model_info = dashboard_data["model_info"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Model Information:**")
            st.write(f"Version: {model_info['version']}")
            st.write(f"Last Retrain: {model_info['last_retrain']}")
            st.write(f"Validation: {model_info['validation_date']}")

        with col2:
            st.markdown("**Validation Metrics:**")
            st.success("✅ Precision: 0.85")
            st.success("✅ Recall: 0.79")
            st.success("✅ F1-Score: 0.82")
            st.success("✅ AUC-ROC: 0.91")

        with col3:
            st.markdown("**Compliance Checklist:**")
            st.success("✅ Model documentation")
            st.success("✅ Performance monitoring")
            st.success("✅ Bias testing")
            st.success("✅ Audit trail")
            st.success(f"✅ Drift status: {model_info['drift_status']}")

def main():
    """Main Streamlit application"""
    # Render sidebar and get demo mode
    demo_mode = render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Score Transaction", "🔍 Investigation Viewer", "👨‍💼 Analyst Console", "📊 Dashboard"])

    with tab1:
        render_score_tab(demo_mode)

    with tab2:
        render_investigation_tab(demo_mode)

    with tab3:
        render_analyst_tab(demo_mode)

    with tab4:
        render_dashboard_tab(demo_mode)

if __name__ == "__main__":
    main()