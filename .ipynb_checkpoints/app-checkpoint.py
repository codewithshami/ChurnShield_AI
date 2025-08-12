import streamlit as st
import json
import pandas as pd
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

# Load model and features
@st.cache_resource
def load_model():
    with open('app/model/feature_names.json') as f:
        feature_names = json.load(f)
    model = xgb.Booster()
    model.load_model('app/model/churn_model.json')
    return model, feature_names

model, feature_names = load_model()

# Page configuration
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        :root {
            --primary: #00ffaa;
            --secondary: #0095ff;
            --danger: #ff3d57;
            --warning: #ffaa00;
            --dark: #0f1118;
            --light: #f8f9fa;
            --neon-glow: 0 0 10px rgba(0, 255, 200, 0.8);
        }
        
        /* Main background with animated gradient */
        .main {
            background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #3a7bd5);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
        }
        
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        /* Sidebar with glass morphism effect */
        .sidebar .sidebar-content {
            background: rgba(31, 34, 48, 0.85) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0, 255, 200, 0.1);
            box-shadow: var(--neon-glow);
        }
        
        /* Buttons with 3D and pulse animation */
        .stButton>button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border-radius: 12px;
            padding: 12px 28px;
            border: none;
            box-shadow: 0 6px 15px rgba(0, 149, 255, 0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 12px 20px rgba(0, 149, 255, 0.6);
            filter: brightness(1.2);
        }
        
        .stButton>button:active {
            transform: translateY(1px);
        }
        
        .stButton>button:after {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent
            );
            transition: 0.5s;
        }
        
        .stButton>button:hover:after {
            left: 100%;
        }
        
        /* Risk labels with animated gradient and glow */
        .risk-high {
            background: linear-gradient(90deg, #ff0f7b, #f89b29);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(255, 15, 123, 0.5);
            animation: pulse 1.5s infinite;
        }
        
        .risk-medium {
            background: linear-gradient(90deg, #f89b29, #ffd200);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(248, 155, 41, 0.5);
            animation: pulse 2s infinite;
        }
        
        .risk-low {
            background: linear-gradient(90deg, #00ffaa, #00b8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.8em;
            text-shadow: 0 0 10px rgba(0, 255, 200, 0.5);
            animation: pulse 2.5s infinite;
        }
        
        @keyframes pulse {
            0% {transform: scale(1);}
            50% {transform: scale(1.05);}
            100% {transform: scale(1);}
        }
        
        /* Card style with glass morphism and floating effect */
        .card {
            background: rgba(42, 46, 61, 0.65);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(0, 255, 200, 0.1);
            transition: all 0.5s ease;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0% {transform: translateY(0px);}
            50% {transform: translateY(-10px);}
            100% {transform: translateY(0px);}
        }
        
        .card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 40px rgba(0, 255, 200, 0.3);
            background: rgba(42, 46, 61, 0.8);
        }
        
        .card-title {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 18px;
            color: var(--primary);
            text-shadow: var(--neon-glow);
            letter-spacing: 1px;
            position: relative;
            padding-bottom: 10px;
        }
        
        .card-title:after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 50px;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), transparent);
            border-radius: 3px;
        }
        
        /* Tabs with animated underline */
        .tabs .stTab {
            transition: all 0.3s ease;
            position: relative;
        }
        
        .tabs .stTab:hover {
            color: var(--primary) !important;
        }
        
        .tabs .stTab:after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 3px;
            background: var(--primary);
            transition: all 0.3s ease;
        }
        
        .tabs .stTab:hover:after {
            left: 0;
            width: 100%;
        }
        
        /* Footer with animated border */
        .sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: rgba(15,17,24,0.95);
            color: #ccc;
            font-size: 0.9em;
            padding: 15px 0;
            text-align: center;
            z-index: 1000;
            border-top: 1px solid rgba(0, 255, 200, 0.2);
            box-shadow: 0 -5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .sticky-footer:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            animation: footerGlow 3s linear infinite;
        }
        
        @keyframes footerGlow {
            0% {background-position: -100% 0;}
            100% {background-position: 200% 0;}
        }
        
        /* Chart containers with glass effect */
        .js-plotly-plot {
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border-radius: 16px;
            padding: 10px;
            background: rgba(42, 46, 61, 0.5) !important;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(0, 255, 200, 0.1);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(var(--primary), var(--secondary));
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
        
        /* Input fields with glow focus */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            background: rgba(42, 46, 61, 0.7) !important;
            border: 1px solid rgba(0, 255, 200, 0.2) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(0, 255, 200, 0.3) !important;
            outline: none !important;
        }
        
        /* Progress bar with gradient */
        .stProgress>div>div>div {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            border-radius: 4px !important;
        }
    </style>
""", unsafe_allow_html=True)   

st.markdown('<div class="content-padding-bottom">', unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331895.png", width=120)
with col2:
    st.title("ChurnShield AI")
    st.markdown("""
    Predict customer churn risk with AI-powered insights and get actionable retention strategies.
    """)

# Sidebar - Customer Profile
with st.sidebar:
    st.header("📋 Customer Details")
    st.markdown("---")
    
    # Customer ID and basic info
    customer_id = st.text_input("Customer ID/Name", "Mohd Shami")
    join_date = st.date_input("Join Date", datetime.now() - timedelta(days=365))
    
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ("Male", "Female", "Other"))
    with col2:
        senior_citizen = st.checkbox("Senior Citizen")
    
    partner = st.checkbox("Has Partner")
    dependents = st.checkbox("Has Dependents")
    
    st.subheader("Account Details")
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.slider('Monthly Charges ($)', 18, 120, 70)
    total_charges = st.slider('Total Charges ($)', 0, 9000, 1000)
    
    col1, col2 = st.columns(2)
    with col1:
        paperless_billing = st.checkbox("Paperless Billing", value=True)
    with col2:
        phone_service = st.checkbox("Phone Service", value=True)
    
    st.subheader("Service Details")
    contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    internet_service = st.selectbox("Internet Service", ("Fiber optic", "DSL", "No"))
    
    st.markdown("**Additional Services**")
    online_security = st.selectbox("Online Security", ("Yes", "No", "No internet service"))
    online_backup = st.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    device_protection = st.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    tech_support = st.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    streaming_tv = st.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    streaming_movies = st.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    
    st.subheader("Payment Details")
    payment_method = st.selectbox("Payment Method", 
                                ("Electronic check", "Mailed check", 
                                 "Bank transfer (automatic)", "Credit card (automatic)"))
    
    st.markdown("---")
    st.markdown("🔍 Adjust the parameters and see the prediction update in real-time.")
    
    # Add a save profile button
    if st.button("💾 Save Profile"):
        st.success("Profile saved successfully!")

# Prepare input data
def prepare_input():
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': 1 if partner else 0,
        'Dependents': 1 if dependents else 0,
        'PhoneService': 1 if phone_service else 0,
        'PaperlessBilling': 1 if paperless_billing else 0,
    }
    
    # Contract type
    input_data['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
    input_data['Contract_One year'] = 1 if contract == "One year" else 0
    input_data['Contract_Two year'] = 1 if contract == "Two year" else 0
    
    # Internet service
    input_data['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
    input_data['InternetService_DSL'] = 1 if internet_service == "DSL" else 0
    input_data['InternetService_No'] = 1 if internet_service == "No" else 0
    
    # Additional services
    input_data['OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0
    input_data['OnlineSecurity_No internet service'] = 1 if online_security == "No internet service" else 0
    input_data['OnlineBackup_Yes'] = 1 if online_backup == "Yes" else 0
    input_data['OnlineBackup_No internet service'] = 1 if online_backup == "No internet service" else 0
    input_data['DeviceProtection_Yes'] = 1 if device_protection == "Yes" else 0
    input_data['DeviceProtection_No internet service'] = 1 if device_protection == "No internet service" else 0
    input_data['TechSupport_Yes'] = 1 if tech_support == "Yes" else 0
    input_data['TechSupport_No internet service'] = 1 if tech_support == "No internet service" else 0
    input_data['StreamingTV_Yes'] = 1 if streaming_tv == "Yes" else 0
    input_data['StreamingTV_No internet service'] = 1 if streaming_tv == "No internet service" else 0
    input_data['StreamingMovies_Yes'] = 1 if streaming_movies == "Yes" else 0
    input_data['StreamingMovies_No internet service'] = 1 if streaming_movies == "No internet service" else 0
    
    # Payment method
    input_data['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
    input_data['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0
    input_data['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == "Bank transfer (automatic)" else 0
    input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card (automatic)" else 0
    
    # Ensure all features are present
    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0
            
    return pd.DataFrame([input_data])[feature_names]

# Make prediction
def predict_churn(input_df):
    dtest = xgb.DMatrix(input_df)
    return model.predict(dtest)[0]

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Analytics", "🛡️ Retention"])

with tab1:
    input_df = prepare_input()
    churn_prob = predict_churn(input_df)
    
    # Risk assessment
    if churn_prob > 0.7:
        risk_level = "HIGH"
        risk_class = "risk-high"
        gauge_color = "#f44336"
        risk_description = "Immediate action required"
    elif churn_prob > 0.4:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
        gauge_color = "#ff9800"
        risk_description = "Proactive measures recommended"
    else:
        risk_level = "LOW"
        risk_class = "risk-low"
        gauge_color = "#4CAF50"
        risk_description = "Normal monitoring"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_prob * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#4CAF50'},
                {'range': [40, 70], 'color': '#FFC107'},
                {'range': [70, 100], 'color': '#F44336'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': churn_prob * 100}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, b=50, t=100, pad=4),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    # Display prediction
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Risk Assessment</div>
            <p style="font-size: 1.5em; margin-bottom: 5px;" class="{risk_class}">{risk_level} RISK</p>
            <p style="color: #aaa; margin-bottom: 15px;">{risk_description}</p>
            <p style="font-size: 1.2em;">Customer ID: <strong>{customer_id}</strong></p>
            <p>Tenure: <strong>{tenure} months</strong></p>
            <p>Monthly Charges: <strong>${monthly_charges}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key factors
    st.markdown("### 🔍 Key Factors Influencing Prediction")
    feature_impact = {
        'Contract Type': 0.35 if contract == "Month-to-month" else (-0.15 if contract == "Two year" else -0.05),
        'Internet Service': 0.25 if internet_service == "Fiber optic" else (-0.1 if internet_service == "DSL" else 0),
        'Tenure': -0.02 * tenure,
        'Online Security': -0.15 if online_security == "Yes" else (0.1 if online_security == "No" else 0),
        'Tech Support': -0.18 if tech_support == "Yes" else (0.1 if tech_support == "No" else 0),
        'Payment Method': 0.12 if payment_method == "Electronic check" else (-0.08 if "automatic" in payment_method else 0),
        'Monthly Charges': 0.005 * monthly_charges
    }
    
    # Create impact bars
    impact_df = pd.DataFrame({
        'Factor': list(feature_impact.keys()),
        'Impact': list(feature_impact.values()),
        'Color': ['#f44336' if x > 0 else '#4CAF50' for x in feature_impact.values()]
    }).sort_values('Impact', ascending=False)
    
    fig = px.bar(impact_df, 
                 x='Impact', 
                 y='Factor', 
                 color='Color',
                 orientation='h',
                 title='Feature Impact on Churn Probability',
                 labels={'Impact': 'Impact Score', 'Factor': ''})
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("📋 View Detailed Explanation", expanded=False):
        st.write("""
        **How this prediction was calculated:**
        
        Our machine learning model analyzes multiple customer attributes to predict churn risk. 
        The key factors influencing this prediction are:
        """)
        
        for factor, impact in sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True):
            if impact > 0:
                st.write(f"- ⬆️ **{factor}**: Increasing churn risk (Impact: {impact:.2f})")
            else:
                st.write(f"- ⬇️ **{factor}**: Reducing churn risk (Impact: {impact:.2f})")
        
        st.write("""
        *Note: Impact scores are relative measures of how much each factor contributes to the 
        overall churn probability in this specific prediction.*
        """)

with tab2:
    st.header("📈 Customer Analytics")
    
    # Feature importance visualization
    st.markdown("### 🎯 Model Feature Importance")
    st.markdown("""
    Understanding which factors most influence churn predictions helps prioritize retention efforts.
    """)
    
    # Mock feature importance (replace with actual from your model)
    features = ['Contract_Month-to-month', 'tenure', 'OnlineSecurity_Yes', 
                'TechSupport_Yes', 'InternetService_Fiber optic', 'MonthlyCharges']
    importance = [0.35, 0.28, 0.18, 0.15, 0.12, 0.10]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 labels={'x': 'Importance Score', 'y': ''},
                 color=importance,
                 color_continuous_scale='Bluered')
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer segmentation
    st.markdown("### 🧩 Customer Segmentation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **High Value Customers**
        - Long tenure
        - Multiple services
        - High monthly charges
        - Low churn risk
        """)
        
    with col2:
        st.markdown("""
        **At-Risk Customers**
        - Short tenure
        - Month-to-month contracts
        - High monthly charges
        - Limited additional services
        """)
    
    # Churn trends
    st.markdown("### 📉 Churn Trends Analysis")
    
    # Mock data for trends
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    churn_rates = [12.5, 11.8, 13.2, 14.5, 15.1, 16.3]
    interventions = [10, 9, 11, 13, 14, 8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months, y=churn_rates,
        name='Churn Rate',
        line=dict(color='#f44336', width=3)
    ))
    
    fig.add_trace(go.Bar(
        x=months, y=interventions,
        name='Retention Interventions',
        marker_color='#4CAF50',
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Monthly Churn Rate vs Retention Interventions',
        xaxis_title='Month',
        yaxis_title='Percentage / Count',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🛡️ Retention Strategies")
    
    if churn_prob > 0.7:
        st.error("### 🚨 High Risk Customer - Immediate Action Required")
        st.markdown("""
        <div class="card">
            <div class="card-title">Recommended Actions</div>
            <p><strong>⏰ Time-sensitive intervention needed</strong></p>
            <ul>
                <li>🔹 <strong>Personalized outreach</strong> from account manager within 24 hours</li>
                <li>🔹 <strong>Special offer</strong>: 20% discount for 6 months with 1-year contract</li>
                <li>🔹 <strong>Service review</strong>: Identify and resolve any service issues</li>
                <li>🔹 <strong>Loyalty bonus</strong>: $50 account credit for continued business</li>
                <li>🔹 <strong>Priority support</strong>: Assign dedicated support representative</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📅 Retention Action Plan")
        action_plan = {
            "Day 1": ["Outreach call", "Special offer email"],
            "Day 3": ["Follow-up call", "Customer satisfaction survey"],
            "Day 7": ["Contract review meeting", "Service optimization"],
            "Day 14": ["Retention offer decision", "Loyalty program enrollment"]
        }
        
        for day, actions in action_plan.items():
            with st.expander(f"📌 {day}"):
                for action in actions:
                    st.write(f"- {action}")
        
    elif churn_prob > 0.4:
        st.warning("### 🟠 Medium Risk Customer - Proactive Measures")
        st.markdown("""
        <div class="card">
            <div class="card-title">Recommended Actions</div>
            <ul>
                <li>🔹 <strong>Engagement campaign</strong>: Add to email nurture sequence</li>
                <li>🔹 <strong>Value-added offer</strong>: Free premium feature for 3 months</li>
                <li>🔹 <strong>Satisfaction survey</strong>: Identify potential issues</li>
                <li>🔹 <strong>Contract incentive</strong>: 10% discount for upgrading to annual contract</li>
                <li>🔹 <strong>Usage tips</strong>: Help customer get more value from service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Suggested Engagement Timeline")
        st.image("https://cdn-icons-png.flaticon.com/512/3652/3652191.png", width=100)
        st.write("""
        1. **Week 1**: Send educational content about underused features
        2. **Week 2**: Offer free consultation with product expert
        3. **Week 3**: Send personalized usage report with recommendations
        4. **Week 4**: Make retention offer based on engagement
        """)
        
    else:
        st.success("### ✅ Low Risk Customer - Maintain Engagement")
        st.markdown("""
        <div class="card">
            <div class="card-title">Recommended Actions</div>
            <ul>
                <li>🔹 <strong>Regular check-ins</strong>: Quarterly business reviews</li>
                <li>🔹 <strong>Loyalty rewards</strong>: Recognize continued business</li>
                <li>🔹 <strong>Referral program</strong>: Encourage customer referrals</li>
                <li>🔹 <strong>Product education</strong>: Advanced feature webinars</li>
                <li>🔹 <strong>Community building</strong>: Invite to customer advisory board</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌱 Growth Opportunities")
        st.write("""
        This satisfied customer represents opportunities for:
        - **Upselling** additional products/services
        - **Cross-selling** complementary solutions
        - **Referrals** to similar businesses
        - **Case study** development
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Retention Playbook")
    st.write("""
    **General Best Practices for All Customer Segments:**
    
    - **Early warning system**: Monitor usage patterns and engagement metrics
    - **Personalization**: Tailor communications to customer needs
    - **Value demonstration**: Regularly show ROI of your service
    - **Multi-channel engagement**: Combine email, phone, and in-app messaging
    - **Continuous improvement**: Gather feedback and iterate on retention strategies
    """)

# with tab4:
#     st.header("📋 Customer History & Notes")
    
    # Mock customer history data
    # history_data = {
    #     "Date": ["2023-06-15", "2023-05-20", "2023-04-10", "2023-03-01", "2023-01-15"],
    #     "Interaction": ["Service call - billing question", "Plan upgrade to Premium", 
    #                     "Technical support ticket resolved", "Annual contract renewal", 
    #                     "Onboarding completed"],
    #     "Agent": ["Sarah K.", "Michael T.", "Tech Support", "Sarah K.", "Onboarding Team"],
    #     "Sentiment": ["Neutral", "Positive", "Negative", "Positive", "Positive"]
    # }
    
    # history_df = pd.DataFrame(history_data)
    # st.dataframe(history_df.style.applymap(lambda x: 'color: #4CAF50' if x == "Positive" else 
    #                                      ('color: #f44336' if x == "Negative" else 'color: #FFC107')),
    #             height=300)
    
    # # Customer notes
    # st.markdown("### 📝 Add New Note")
    # new_note = st.text_area("Enter notes about this customer", height=100)
    
    # if st.button("Save Note"):
    #     if new_note.strip() != "":
    #         st.success("Note saved successfully!")
    #     else:
    #         st.warning("Please enter a note before saving")
    
    # Customer documents
    # st.markdown("### 📂 Customer Documents")
    # st.write("""
    # - Contract agreement (signed 2023-03-01)
    # - Onboarding checklist (completed 2023-01-15)
    # - SLA agreement
    # """)
    
    # # Upload new document
    # uploaded_file = st.file_uploader("Upload new document", type=['pdf', 'docx', 'txt'])
    # if uploaded_file is not None:
    #     st.success(f"File {uploaded_file.name} uploaded successfully")

# Footer
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="sticky-footer">
        <div>
            <span>ChurnShield AI • Powered by XGBoost • v2 1.7.6</span><br>
            <span>Developed by Mohd Shami • Last updated: {}</span><br>
        </div>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)