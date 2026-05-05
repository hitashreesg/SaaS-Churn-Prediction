# SaaS Customer Churn Predictor — FINAL (Main UI + Info preserved)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor · SaaS", page_icon="🔄", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
st.markdown("""
<style>
.main .block-container { padding-top: 1.2rem; }
#MainMenu, footer, header { visibility: hidden; }

.prob-num { font-size:80px; font-weight:800; }
.risk-pill { padding:6px 16px; border-radius:20px; font-weight:700; }
.pill-low { background:#DCFCE7; color:#065F46; }
.pill-medium { background:#FEF9C3; color:#78350F; }
.pill-high { background:#FEE2E2; color:#7F1D1D; }

.info-box {
    background:#F8FAFC;
    padding:10px 14px;
    border-radius:8px;
    border:1px solid #E2E8F0;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (OPTIONAL)
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("churn_model.pkl", "rb"))
        return model, True
    except:
        return None, False

model, model_loaded = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# CHURN LOGIC
def compute_churn(v):
    raw = (
        (1 - v["logins"]/30)*0.2 +
        (v["inactive"]/90)*0.17 +
        (1 - v["features"]/12)*0.13 +
        (1 - (v["nps"]+100)/200)*0.11 +
        (1 - (v["sat"]/10))*0.1
    )
    return round(min(max(raw*100,2),98),1)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
st.title("🔄 SaaS Customer Churn Predictor")

col_info1, col_info2 = st.columns([2,1])

with col_info1:
    st.markdown(
        "<div class='info-box'>Adjust inputs → churn updates instantly ⚡</div>",
        unsafe_allow_html=True
    )

with col_info2:
    if model_loaded:
        st.success("🤖 Trained ML Model Active")
    else:
        st.info("⚙️ Demo Scoring Engine (No model loaded)")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT PANEL
st.markdown("## ⚙️ Customer Inputs")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### 💰 Revenue")
    mrr = st.slider("Monthly MRR (₹)", 500, 50000, 2000, 500)
    age = st.slider("Account age (days)", 1, 1095, 45)
    upgrades = st.slider("Plan upgrades", 0, 10, 0)

    plan = st.selectbox("Plan", ["Free","Basic","Pro","Enterprise"])
    billing = st.selectbox("Billing", ["Monthly","Quarterly","Annual"])

with c2:
    st.markdown("#### 📊 Engagement")
    logins = st.slider("Logins/month", 0, 60, 2)
    features = st.slider("Features used", 0, 20, 1)
    inactive = st.slider("Days inactive", 0, 120, 45)

with c3:
    st.markdown("#### 🎧 Support")
    tickets = st.slider("Support tickets", 0, 30, 8)
    sat = st.slider("CSAT", 1, 10, 3)
    nps = st.slider("NPS", -100, 100, -30)

with c4:
    st.markdown("#### 🏢 Company")
    size = st.selectbox("Company size", ["Solo","Small","Mid","Enterprise"])

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE
vals = dict(
    mrr=mrr, age=age, upgrades=upgrades,
    logins=logins, features=features, inactive=inactive,
    tickets=tickets, sat=sat, nps=nps
)

prob = compute_churn(vals)

# Risk
if prob < 30:
    label, cls, color = "Low Risk", "pill-low", "#16A34A"
elif prob < 60:
    label, cls, color = "Medium Risk", "pill-medium", "#CA8A04"
else:
    label, cls, color = "High Risk", "pill-high", "#DC2626"

# ─────────────────────────────────────────────────────────────────────────────
# GAUGE
def draw_gauge(prob):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    for lo,hi,col in [(0,30,"green"),(30,60,"yellow"),(60,100,"red")]:
        t = np.linspace(np.pi*(1-lo/100), np.pi*(1-hi/100), 50)
        ax.plot(t,[1]*50,lw=20,color=col)
    angle = np.pi*(1-prob/100)
    ax.plot([angle],[1],"o",markersize=10)
    ax.set_axis_off()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
cA, cB, cC = st.columns([1,1,2])

with cA:
    st.markdown(f"<div class='prob-num' style='color:{color}'>{prob}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='risk-pill {cls}'>{label}</div>", unsafe_allow_html=True)

with cB:
    st.pyplot(draw_gauge(prob))

with cC:
    st.metric("MRR at Risk", f"₹{int(mrr*prob/100):,}")
    st.metric("Logins/month", logins)
    st.metric("Days inactive", inactive)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
st.caption("AI/ML Internship Project · SaaS Churn Prediction")