import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import timedelta, date
import time

# ==========================================
# 1. PAGE CONFIGURATION & CRAZY CSS
# ==========================================
st.set_page_config(page_title="Nexus AI Analytics - India", page_icon="🇮🇳", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    
    /* Breathing Animated Background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #0b0f19, #131731, #050505, #111e30);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.6) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    header {visibility: hidden;} footer {visibility: hidden;}
    
    /* Super Smooth Glass Cards with Pulse Glow Hover */
    @keyframes pulseGlow {
        0% { box-shadow: 0 10px 30px rgba(6, 182, 212, 0.2); }
        50% { box-shadow: 0 10px 40px rgba(6, 182, 212, 0.6); }
        100% { box-shadow: 0 10px 30px rgba(6, 182, 212, 0.2); }
    }

    .glass-card {
        background: rgba(15, 20, 35, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.6);
        animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-card:hover { 
        transform: translateY(-8px) scale(1.02); 
        border-color: #06b6d4; 
        animation: pulseGlow 2s infinite;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #6366f1, #06b6d4, #10b981);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; margin-bottom: 0;
        animation: textShine 3s linear infinite;
    }
    
    @keyframes textShine {
        to { background-position: 200% center; }
    }

    @keyframes slideUp {
        0% { opacity: 0; transform: translateY(50px) scale(0.95); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .delay-1 { animation-delay: 0.1s; } .delay-2 { animation-delay: 0.2s; } .delay-3 { animation-delay: 0.3s; }
    
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 10px 0; letter-spacing: -1px;}
    .metric-label { font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;}
    .metric-trend-up { color: #10b981; font-weight: 600; font-size: 0.95rem; background: rgba(16, 185, 129, 0.1); padding: 4px 12px; border-radius: 20px; display: inline-block;}
    .metric-trend-down { color: #ef4444; font-weight: 600; font-size: 0.95rem; background: rgba(239, 68, 68, 0.1); padding: 4px 12px; border-radius: 20px; display: inline-block;}
    
    /* Custom Button Styling for Exports */
    .stDownloadButton button {
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stDownloadButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Helper: Indian Currency Formatter
def format_inr(number):
    if number >= 10000000: return f"₹{number/10000000:.2f} Cr"
    elif number >= 100000: return f"₹{number/100000:.2f} L"
    else: return f"₹{number:,.0f}"

# Helper: Ultra Smooth Plotly Theme
def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False),
        hovermode="x unified"
    )
    return fig

# Helper: Add Neon Glow to lines
def add_neon_glow(fig, x, y, color, name, dash='solid'):
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color.replace(')', ', 0.1)').replace('rgb', 'rgba'), width=12, shape='spline', dash=dash), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color.replace(')', ', 0.3)').replace('rgb', 'rgba'), width=6, shape='spline', dash=dash), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=3, shape='spline', dash=dash), name=name, mode='lines'))
    return fig

# ==========================================
# 2. INDIAN DATA GENERATOR
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    num_records = 2500
    categories = ['Electronics', 'Furniture', 'Software', 'Services']
    
    cities_map = {
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'Delhi NCR': {'lat': 28.7041, 'lon': 77.1025},
        'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
        'Pune': {'lat': 18.5204, 'lon': 73.8567}
    }
    city_list = list(cities_map.keys())
    chosen_cities = np.random.choice(city_list, num_records)
    
    data = {
        'OrderID': [f"IND-{10000+i}" for i in range(num_records)],
        'Date': sorted([date(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(num_records)]),
        'Client': [f"Enterprise_{i}" for i in np.random.randint(1, 400, num_records)],
        'Category': np.random.choice(categories, num_records, p=[0.45, 0.05, 0.35, 0.15]),
        'City': chosen_cities,
        'Lat': [cities_map[c]['lat'] + np.random.uniform(-0.5, 0.5) for c in chosen_cities], 
        'Lon': [cities_map[c]['lon'] + np.random.uniform(-0.5, 0.5) for c in chosen_cities],
        'Sales': np.round(np.random.uniform(50000, 2500000, num_records), 0)
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Profit'] = df.apply(
        lambda row: row['Sales'] * np.random.uniform(0.12, 0.45) if row['Category'] != 'Furniture' 
        else row['Sales'] * np.random.uniform(-0.02, 0.10), axis=1
    )
    return df

df = load_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown('<h2 style="color: #06b6d4; font-weight: 800; font-size: 2rem;">⚡ NEXUS AI</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1rem; margin-top: -10px;'>Bharat Enterprise Analytics</p><br>", unsafe_allow_html=True)
    
    page = st.radio("Navigation", [
        "📊 Executive Dashboard", 
        "🇮🇳 India Geo-Spatial",
        "🚨 Fraud Detection",
        "📈 Predictive AI & Simulator", 
        "👥 Segment Analysis", 
        "🤖 AI Copilot",
        "📑 Export Reports"
    ], label_visibility="collapsed")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🟢 Server: Mumbai (ap-south-1)\n\nModel: Nexus-Bharat v4.0")

# ==========================================
# 4. MODULE 1: EXECUTIVE DASHBOARD
# ==========================================
if page == "📊 Executive Dashboard":
    st.markdown('<h1 class="gradient-text">India Operations.</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="glass-card delay-1"><div class="metric-label">Total Revenue (FY24)</div><div class="metric-value">{format_inr(df['Sales'].sum())}</div><div class="metric-trend-up">↑ +22.4% vs FY23</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="glass-card delay-2"><div class="metric-label">Net Profit (FY24)</div><div class="metric-value">{format_inr(df['Profit'].sum())}</div><div class="metric-trend-up">↑ +14.1% vs FY23</div></div>""", unsafe_allow_html=True)
    with col3:
        margin = (df['Profit'].sum() / df['Sales'].sum()) * 100
        st.markdown(f"""<div class="glass-card delay-3"><div class="metric-label">Avg Profit Margin</div><div class="metric-value">{margin:.1f}%</div><div class="metric-trend-down">↓ Supply Chain Impact</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card delay-3">', unsafe_allow_html=True)
    st.markdown('### 🌊 Revenue Velocity (Smooth Spline)')
    monthly_sales = df.resample('ME', on='Date')['Sales'].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales['Date'], y=monthly_sales['Sales'], fill='tozeroy', mode='none', fillcolor='rgba(6, 182, 212, 0.1)', line_shape='spline'))
    fig = add_neon_glow(fig, monthly_sales['Date'], monthly_sales['Sales'], 'rgb(6, 182, 212)', 'Revenue')
    
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. MODULE 2: INDIA GEO-SPATIAL
# ==========================================
elif page == "🇮🇳 India Geo-Spatial":
    st.markdown('<h1 class="gradient-text">Bharat Footprint.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card delay-1">', unsafe_allow_html=True)
    
    fig = px.scatter_geo(
        df, lat='Lat', lon='Lon', color='City', size='Sales',
        hover_name='City', color_discrete_sequence=['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
    )
    fig.update_geos(
        center=dict(lon=78.9629, lat=22.5937), projection_scale=4.8, 
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.2)",
        showland=True, landcolor="rgba(20, 25, 40, 0.9)",
        bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 6. MODULE 3: FRAUD DETECTION
# ==========================================
elif page == "🚨 Fraud Detection":
    st.markdown('<h1 class="gradient-text">Anomaly AI.</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card delay-1">', unsafe_allow_html=True)
    
    features = ['Sales', 'Profit']
    clf = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    df['Anomaly'] = clf.fit_predict(df[features])
    
    normal = df[df['Anomaly'] == 1]
    anomalies = df[df['Anomaly'] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal['Sales'], y=normal['Profit'], mode='markers', marker=dict(color='rgba(59, 130, 246, 0.4)', size=8), name='Standard Trax'))
    fig.add_trace(go.Scatter(x=anomalies['Sales'], y=anomalies['Profit'], mode='markers', marker=dict(color='#ef4444', size=14, line=dict(color='#fca5a5', width=2)), name='Flagged (Review)'))
    
    fig = apply_dark_theme(fig)
    fig.update_layout(xaxis_title="Transaction Value (INR)", yaxis_title="Profit Margin (INR)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. MODULE 4: PREDICTIVE AI & SIMULATOR
# ==========================================
elif page == "📈 Predictive AI & Simulator":
    st.markdown('<h1 class="gradient-text">Future Reality.</h1>', unsafe_allow_html=True)

    monthly_sales = df.resample('ME', on='Date')['Sales'].sum().reset_index()
    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y = monthly_sales['Sales']
    model = LinearRegression().fit(X, y)

    st.markdown('<div class="glass-card delay-1">', unsafe_allow_html=True)
    st.markdown("### 🎛️ AI What-If Engine (Adjust Market Variables)")
    col1, col2 = st.columns(2)
    with col1:
        marketing_boost = st.slider("Diwali Campaign Spend Increase (%)", -10, 50, 0)
    with col2:
        price_change = st.slider("Product Price Adjustment (%)", -10, 20, 0)
    st.markdown('</div><br>', unsafe_allow_html=True)

    future_idx = np.array([[len(monthly_sales)], [len(monthly_sales)+1], [len(monthly_sales)+2]])
    base_prediction = model.predict(future_idx)
    multiplier = 1 + (marketing_boost / 100 * 0.9) + (price_change / 100 * 0.4)
    sim_prediction = base_prediction * multiplier
    future_dates = pd.date_range(start=monthly_sales['Date'].iloc[-1] + pd.Timedelta(days=1), periods=3, freq='ME')

    st.markdown('<div class="glass-card delay-2">', unsafe_allow_html=True)
    fig = go.Figure()
    fig = add_neon_glow(fig, monthly_sales['Date'], monthly_sales['Sales'], 'rgb(139, 92, 246)', 'Actual Revenue')
    # Standard ML Prediction
    fig.add_trace(go.Scatter(x=future_dates, y=base_prediction, mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dot', shape='spline'), name='Baseline ML Model'))
    # What-If Prediction
    if marketing_boost != 0 or price_change != 0:
        fig = add_neon_glow(fig, future_dates, sim_prediction, 'rgb(16, 185, 129)', 'AI Simulated Reality', dash='dash')
    
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. MODULE 5: SEGMENT ANALYSIS
# ==========================================
elif page == "👥 Segment Analysis":
    st.markdown('<h1 class="gradient-text">Client Clusters.</h1>', unsafe_allow_html=True)
    
    cust_data = df.groupby('Client').agg(Total_Spent=('Sales', 'sum'), Orders=('Date', 'count')).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cust_data['Cluster'] = kmeans.fit_predict(cust_data[['Total_Spent', 'Orders']])
    cluster_map = {0: "Growth (Tier 2)", 1: "Whales (Enterprise)", 2: "Retail (Tier 3)"}
    cust_data['Segment'] = cust_data['Cluster'].map(cluster_map)

    st.markdown('<div class="glass-card delay-1">', unsafe_allow_html=True)
    fig = px.scatter(
        cust_data, x='Total_Spent', y='Orders', color='Segment',
        color_discrete_map={"Growth (Tier 2)": "#6366f1", "Whales (Enterprise)": "#10b981", "Retail (Tier 3)": "#ef4444"},
        hover_data=['Client']
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='rgba(255,255,255,0.8)')))
    fig.update_layout(xaxis_title="Total Spend (INR)", yaxis_title="Total Invoices")
    st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 9. MODULE 6: AI COPILOT
# ==========================================
elif page == "🤖 AI Copilot":
    st.markdown('<h1 class="gradient-text">Nexus Bharat.</h1>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Namaste! I am Nexus AI. I have analyzed 2,500 enterprise transactions across major Indian tech hubs. How can I assist you?"}]

    st.markdown('<div class="glass-card delay-1" style="min-height: 50vh;">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Bangalore sales, top clients, or fraud..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            p_lower = prompt.lower()
            
            if "city" in p_lower or "hub" in p_lower or "bangalore" in p_lower:
                ans = f"Currently, **{df.groupby('City')['Sales'].sum().idxmax()}** is generating the maximum enterprise volume, heavily driven by the Software sector."
            elif "crore" in p_lower or "profit" in p_lower:
                ans = f"Your total net profit across India operations is **{format_inr(df['Profit'].sum())}**. The Enterprise 'Whale' segment contributes 68% of this."
            elif "predict" in p_lower or "future" in p_lower:
                ans = "Based on our Linear Regression model, assuming no external market shocks, baseline revenue is expected to grow steadily over the next 3 months. Use the 'Predictive AI' tab to simulate Diwali campaign impacts."
            else:
                ans = "Analyzing telemetry... Please ask specific queries regarding 'cities', 'profit margins', 'predictions', or 'anomalies'."

            for chunk in ans.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 10. MODULE 7: EXPORT REPORTS (RESTORED)
# ==========================================
elif page == "📑 Export Reports":
    st.markdown('<h1 class="gradient-text">Data Vault.</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8; margin-bottom: 30px;">Download raw telemetry and AI-processed datasets.</p>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card delay-1">', unsafe_allow_html=True)
    st.markdown("### 📥 Primary Dataset")
    st.write("Export the full, cleaned historical dataset containing 2,500 records mapped across Indian cities, complete with calculated profit margins.")
    
    csv = df.to_csv(index=False).encode('utf-8')
    
    # Render the custom styled download button
    st.download_button(
        label="Download Raw Dataset (CSV)",
        data=csv,
        file_name='Nexus_Bharat_Sales_Data.csv',
        mime='text/csv',
    )
    st.markdown('</div><br>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card delay-2">', unsafe_allow_html=True)
    st.markdown("### 📊 Client Segmentation Report")
    st.write("Export the K-Means Machine Learning results mapping each client to their AI-determined behavioral cluster.")
    
    cust_data = df.groupby('Client').agg(Total_Spent=('Sales', 'sum'), Orders=('Date', 'count')).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cust_data['Cluster'] = kmeans.fit_predict(cust_data[['Total_Spent', 'Orders']])
    cluster_map = {0: "Growth (Tier 2)", 1: "Whales (Enterprise)", 2: "Retail (Tier 3)"}
    cust_data['Segment'] = cust_data['Cluster'].map(cluster_map)
    
    csv_segments = cust_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download AI Segments (CSV)",
        data=csv_segments,
        file_name='Client_Segmentation_AI.csv',
        mime='text/csv',
    )
    st.markdown('</div>', unsafe_allow_html=True)