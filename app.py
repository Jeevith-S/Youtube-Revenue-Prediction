
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="📺",
    layout="wide"
)

# ── Load model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('ridge_model.pkl', 'rb') as f:
        return pickle.load(f)

data    = load_model()
model   = data['model']
scaler  = data['scaler']
columns = data['columns']

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF0000;
        text-align: center;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        border-left: 4px solid #FF0000;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #FF0000, #cc0000);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-label {
        font-size: 1rem;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📺 YouTube Ad Revenue Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict how much ad revenue your video will earn</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Input Section ──────────────────────────────────────────────────────
st.markdown('<p class="section-header">Enter Your Video Details</p>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📈 Performance Metrics**")
    views       = st.number_input("Views",
                    min_value=0, value=10000, step=1000)
    likes       = st.number_input("Likes",
                    min_value=0, value=1000,  step=100)
    comments    = st.number_input("Comments",
                    min_value=0, value=200,   step=10)

with col2:
    st.markdown("**⏱️ Watch Metrics**")
    watch_time  = st.number_input("Watch Time (minutes)",
                    min_value=0, value=30000, step=1000)
    video_len   = st.number_input("Video Length (minutes)",
                    min_value=1, value=10,    step=1)
    subscribers = st.number_input("Subscribers",
                    min_value=0, value=500000, step=10000)

with col3:
    st.markdown("**🗂️ Video Context**")
    category = st.selectbox("Category",
                 ["Education", "Entertainment", "Gaming",
                  "Lifestyle", "Music", "Tech"])
    device   = st.selectbox("Primary Device",
                 ["Mobile", "TV", "Tablet"])
    country  = st.selectbox("Country",
                 ["CA", "DE", "IN", "UK", "US"])

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────────────────
predict_btn = st.button("🚀 Predict Ad Revenue")

if predict_btn:

    # ── Build input dataframe ──────────────────────────────────────────
    engagement_rate = (likes + comments) / views if views > 0 else 0

    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    input_df.at[0, 'views']                = views
    input_df.at[0, 'likes']                = likes
    input_df.at[0, 'comments']             = comments
    input_df.at[0, 'watch_time_minutes']   = watch_time
    input_df.at[0, 'video_length_minutes'] = video_len
    input_df.at[0, 'subscribers']          = subscribers
    input_df.at[0, 'engagement_rate']      = engagement_rate
    input_df.at[0, 'year']                 = 2025
    input_df.at[0, 'month']               = 6
    input_df.at[0, 'day']                 = 15
    input_df.at[0, 'hour']               = 12
    input_df.at[0, 'minute']             = 0
    input_df.at[0, 'weekday']            = 1

    for col in columns:
        if col.startswith('category_') and col == f'category_{category}':
            input_df.at[0, col] = 1
        if col.startswith('device_')   and col == f'device_{device}':
            input_df.at[0, col] = 1
        if col.startswith('country_')  and col == f'country_{country}':
            input_df.at[0, col] = 1

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]

    # ── Result — 4 metric cards only ──────────────────────────────────
    st.markdown("### 💰 Prediction Result")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("💵 Predicted Revenue",  f"${prediction:.2f}")
    with m2:
        st.metric("📊 Engagement Rate",
                  f"{engagement_rate * 100:.2f}%")
    with m3:
        st.metric("📅 Est. Monthly (×30)", f"${prediction * 30:.0f}")
    with m4:
        st.metric("📆 Est. Yearly (×365)", f"${prediction * 365:.0f}")
