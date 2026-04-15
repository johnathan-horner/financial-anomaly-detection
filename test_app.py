#!/usr/bin/env python3
"""Simple test app to diagnose Streamlit Cloud issues"""

import streamlit as st
import os

st.title("🔧 Financial Anomaly Detection - Deployment Test")

st.write("## Environment Check")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in directory:")

for file in os.listdir("."):
    st.write(f"- {file}")

st.write("## Import Test")
try:
    import requests
    import pandas as pd
    import plotly
    import numpy as np
    st.success("✅ All imports successful!")
except Exception as e:
    st.error(f"❌ Import error: {e}")

st.write("## File Check")
if os.path.exists("docs/Transaction_Anomaly_Detection_AWS_Architecture.png"):
    st.success("✅ Architecture diagram found")
    st.image("docs/Transaction_Anomaly_Detection_AWS_Architecture.png", caption="Architecture", width=300)
else:
    st.error("❌ Architecture diagram missing")

st.write("## App Test")
if st.button("Test Button"):
    st.balloons()
    st.success("Button works!")