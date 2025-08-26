import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import time as tm
import folium
from streamlit_folium import st_folium
import random
from geopy.distance import geodesic
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="📊 FoodBridge Analytics - Data Science Platform",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .ds-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .ml-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .alert-urgent {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-success {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .food-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .status-available {
        background-color: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-claimed {
        background-color: #ff9800;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .status-delivered {
        background-color: #2196f3;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state and generate comprehensive dataset
@st.cache_data
def generate_comprehensive_dataset():
    """Generate a comprehensive dataset for food rescue analytics"""
    np.random.seed(42)
    random.seed(42)
    
    # Generate 1000+ data points over 12 months
    n_records = 1200
    start_date = datetime.now() - timedelta(days=365)
    
    data = []
    
    # Define realistic parameters
    areas = ['Race Course', 'RS Puram', 'Gandhipuram', 'Peelamedu', 'Saibaba Colony', 
             'Singanallur', 'Vadavalli', 'Ukkadam', 'Town Hall', 'Coimbatore North']
    food_types = ['Cooked Meals', 'Bakery Items', 'Fruits & Vegetables', 'Raw Ingredients', 'Packaged Food', 'Dairy Products']
    donor_types = ['Restaurant', 'Hotel', 'Bakery', 'Grocery Store', 'Catering Service', 'Individual']
    categories = ['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Mixed']
    
    for i in range(n_records):
        # Generate realistic temporal patterns
        days_ago = np.random.randint(0, 365)
        donation_date = start_date + timedelta(days=days_ago)
        
        # Weekly patterns (more donations on weekends)
        weekday = donation_date.weekday()
        base_prob = 0.7 if weekday < 5 else 1.2
        
        # Seasonal patterns
        month = donation_date.month
        seasonal_multiplier = 1.3 if month in [11, 12, 1] else 1.0  # More during holidays
        
        area = np.random.choice(areas, p=[0.15, 0.12, 0.14, 0.11, 0.09, 0.08, 0.10, 0.07, 0.08, 0.06])
        food_type = np.random.choice(food_types, p=[0.35, 0.20, 0.15, 0.12, 0.10, 0.08])
        donor_type = np.random.choice(donor_types, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        category = np.random.choice(categories, p=[0.45, 0.30, 0.15, 0.10])
        
        # Realistic quantity based on food type and donor type
        if food_type == 'Cooked Meals':
            base_quantity = np.random.normal(50, 20)
        elif food_type == 'Bakery Items':
            base_quantity = np.random.normal(25, 10)
        elif food_type == 'Fruits & Vegetables':
            base_quantity = np.random.normal(30, 15)
        else:
            base_quantity = np.random.normal(20, 10)
            
        quantity = max(1, int(base_quantity * seasonal_multiplier))
        
        # Response time (affected by area, time of day, urgency)
        base_response = np.random.normal(2.5, 1.2)  # hours
        area_factor = np.random.uniform(0.8, 1.3)
        response_time = max(0.25, base_response * area_factor)
        
        # Success rate (whether donation was successfully delivered)
        success_prob = 0.85 if response_time < 3 else 0.65
        was_successful = np.random.random() < success_prob
        
        # Distance from donor to recipient
        distance = np.random.normal(5.2, 2.8)
        distance = max(0.5, distance)
        
        # Calculate derived metrics
        estimated_people_served = quantity * np.random.uniform(0.8, 1.2) if food_type == 'Cooked Meals' else quantity * 0.6
        waste_prevented = quantity * 0.5 if was_successful else 0  # kg
        co2_saved = waste_prevented * 2.5  # kg CO2 per kg food
        economic_value = quantity * np.random.uniform(50, 200)  # INR
        
        # Urgency score based on expiry time
        expiry_hours = np.random.choice([1, 2, 3, 4, 6, 8, 12], p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05])
        urgency_score = 10 - min(expiry_hours, 10)
        
        data.append({
            'donation_id': f'FD{i+1:04d}',
            'date': donation_date,
            'area': area,
            'food_type': food_type,
            'donor_type': donor_type,
            'category': category,
            'quantity': quantity,
            'response_time_hours': response_time,
            'was_successful': was_successful,
            'distance_km': distance,
            'estimated_people_served': int(estimated_people_served),
            'waste_prevented_kg': waste_prevented,
            'co2_saved_kg': co2_saved,
            'economic_value_inr': economic_value,
            'urgency_score': urgency_score,
            'expiry_hours': expiry_hours,
            'weekday': weekday,
            'month': month,
            'hour': np.random.randint(6, 23),  # Donations typically during day
            'volunteer_rating': np.random.uniform(3.5, 5.0) if was_successful else np.random.uniform(2.0, 4.0)
        })
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_data
def load_data():
    df = generate_comprehensive_dataset()
    df['date'] = pd.to_datetime(df['date'])
    return df

# Initialize data
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# App title and description
st.title("📊 FoodBridge Data Science Analytics Platform")
st.markdown("""
**Advanced Analytics | Machine Learning | Predictive Insights**

A comprehensive data science platform analyzing food rescue patterns, predicting demand, and optimizing 
resource allocation to maximize impact in reducing food waste and hunger.

---
""")

# Sidebar navigation
with st.sidebar:
    st.title("🔍 Analytics Menu")
    selected_page = st.radio(
        "Navigate to:",
        ["📈 Executive Dashboard", 
         "🔬 Data Exploration", 
         "🤖 Machine Learning Models",
         "📊 Statistical Analysis",
         "🗺️ Geospatial Analytics",
         "📋 Data Science Reports",
         "⚡ Real-time Predictions"]
    )

# Executive Dashboard
if selected_page == "📈 Executive Dashboard":
    st.header("📈 Executive Dashboard")
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    total_donations = len(df)
    successful_donations = df['was_successful'].sum()
    total_people_served = df[df['was_successful']]['estimated_people_served'].sum()
    total_waste_prevented = df['waste_prevented_kg'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦</h3>
            <h2>{total_donations:,}</h2>
            <p>Total Donations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        success_rate = (successful_donations / total_donations) * 100
        st.markdown(f"""
        <div class="ds-card">
            <h3>✅</h3>
            <h2>{success_rate:.1f}%</h2>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="ml-card">
            <h3>👥</h3>
            <h2>{total_people_served:,.0f}</h2>
            <p>People Served</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>♻️</h3>
            <h2>{total_waste_prevented:,.0f}kg</h2>
            <p>Waste Prevented</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Time series analysis
    st.subheader("📈 Temporal Trends Analysis")
    
    # Daily aggregation
    daily_stats = df.groupby(df['date'].dt.date).agg({
        'donation_id': 'count',
        'was_successful': 'sum',
        'waste_prevented_kg': 'sum',
        'estimated_people_served': 'sum'
    }).reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Donations', 'Success Rate', 'Waste Prevented (kg)', 'People Served'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['donation_id'], 
                            mode='lines', name='Donations', line=dict(color='blue')), row=1, col=1)
    
    daily_stats['success_rate'] = (daily_stats['was_successful'] / daily_stats['donation_id']) * 100
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['success_rate'], 
                            mode='lines', name='Success %', line=dict(color='green')), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['waste_prevented_kg'], 
                            mode='lines', name='Waste Prevented', line=dict(color='orange')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['estimated_people_served'], 
                            mode='lines', name='People Served', line=dict(color='purple')), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Key Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏘️ Performance by Area")
        area_stats = df.groupby('area').agg({
            'donation_id': 'count',
            'was_successful': 'mean',
            'response_time_hours': 'mean',
            'waste_prevented_kg': 'sum'
        }).round(2)
        area_stats['success_rate'] = (area_stats['was_successful'] * 100).round(1)
        
        fig_area = px.bar(
            x=area_stats.index, 
            y=area_stats['donation_id'],
            color=area_stats['success_rate'],
            color_continuous_scale='RdYlGn',
            title="Donations by Area (colored by success rate)",
            labels={'x': 'Area', 'y': 'Number of Donations', 'color': 'Success Rate %'}
        )
        st.plotly_chart(fig_area, use_container_width=True)
    
    with col2:
        st.subheader("🍽️ Food Type Distribution")
        food_stats = df['food_type'].value_counts()
        fig_pie = px.pie(
            values=food_stats.values,
            names=food_stats.index,
            title="Distribution of Food Types"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Data Exploration
elif selected_page == "🔬 Data Exploration":
    st.header("🔬 Data Exploration & Insights")
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['date'].dt.date.min()} to {df['date'].dt.date.max()}")
    with col3:
        st.metric("Features", f"{len(df.columns)}")
    
    # Display data sample
    st.subheader("📊 Data Sample")
    st.dataframe(df.head(10))
    
    # Data quality assessment
    st.subheader("🔍 Data Quality Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing_data[missing_data > 0])
    
    with col2:
        st.write("**Data Types:**")
        data_types = df.dtypes.value_counts()
        fig_types = px.pie(values=data_types.values, names=data_types.index, 
                          title="Distribution of Data Types")
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols].describe()
        st.dataframe(summary_stats)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📈 Distribution Analysis")
    
    selected_feature = st.selectbox(
        "Select feature to analyze:",
        numeric_cols
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, x=selected_feature,
            title=f"Distribution of {selected_feature}",
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df, y=selected_feature, x='area',
            title=f"{selected_feature} by Area"
        )
        fig_box.update_xaxis(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

# Machine Learning Models
elif selected_page == "🤖 Machine Learning Models":
    st.header("🤖 Machine Learning Models & Predictions")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type:",
        ["🎯 Success Prediction", "⏱️ Response Time Prediction", "🏷️ Food Type Classification", "📊 Clustering Analysis"]
    )
    
    if model_type == "🎯 Success Prediction":
        st.subheader("🎯 Donation Success Prediction Model")
        
        # Prepare features
        features_for_success = ['quantity', 'distance_km', 'urgency_score', 'expiry_hours', 'weekday', 'hour']
        
        # Add encoded categorical features
        le_area = LabelEncoder()
        le_food_type = LabelEncoder()
        le_donor_type = LabelEncoder()
        
        df_ml = df.copy()
        df_ml['area_encoded'] = le_area.fit_transform(df_ml['area'])
        df_ml['food_type_encoded'] = le_food_type.fit_transform(df_ml['food_type'])
        df_ml['donor_type_encoded'] = le_donor_type.fit_transform(df_ml['donor_type'])
        
        X = df_ml[features_for_success + ['area_encoded', 'food_type_encoded', 'donor_type_encoded']]
        y = df_ml['was_successful']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_classifier.predict(X_test)
        
        # Model performance
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy = (y_pred == y_test).mean()
            st.metric("Model Accuracy", f"{accuracy:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10),
                x='importance', y='feature',
                orientation='h',
                title="Top 10 Feature Importance"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Classification report
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, title="Confusion Matrix",
                             labels=dict(x="Predicted", y="Actual"),
                             x=['Failed', 'Successful'], y=['Failed', 'Successful'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Prediction interface
        st.subheader("🔮 Make Prediction")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_quantity = st.number_input("Quantity", min_value=1, max_value=200, value=50)
            pred_distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0)
            pred_urgency = st.slider("Urgency Score", 1, 10, 5)
        
        with col2:
            pred_expiry = st.number_input("Expiry Hours", min_value=1, max_value=24, value=4)
            pred_weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 1)
            pred_hour = st.slider("Hour of Day", 0, 23, 12)
        
        with col3:
            pred_area = st.selectbox("Area", df['area'].unique())
            pred_food_type = st.selectbox("Food Type", df['food_type'].unique())
            pred_donor_type = st.selectbox("Donor Type", df['donor_type'].unique())
        
        if st.button("🎯 Predict Success Probability"):
            # Prepare prediction data
            pred_data = pd.DataFrame({
                'quantity': [pred_quantity],
                'distance_km': [pred_distance],
                'urgency_score': [pred_urgency],
                'expiry_hours': [pred_expiry],
                'weekday': [pred_weekday],
                'hour': [pred_hour],
                'area_encoded': [le_area.transform([pred_area])[0]],
                'food_type_encoded': [le_food_type.transform([pred_food_type])[0]],
                'donor_type_encoded': [le_donor_type.transform([pred_donor_type])[0]]
            })
            
            success_prob = rf_classifier.predict_proba(pred_data)[0][1]
            
            if success_prob > 0.7:
                st.success(f"✅ High Success Probability: {success_prob:.2%}")
            elif success_prob > 0.5:
                st.warning(f"⚠️ Moderate Success Probability: {success_prob:.2%}")
            else:
                st.error(f"❌ Low Success Probability: {success_prob:.2%}")
    
    elif model_type == "⏱️ Response Time Prediction":
        st.subheader("⏱️ Response Time Prediction Model")
        
        # Prepare features for response time prediction
        features_for_time = ['quantity', 'distance_km', 'urgency_score', 'weekday', 'hour']
        
        df_ml = df.copy()
        df_ml['area_encoded'] = LabelEncoder().fit_transform(df_ml['area'])
        
        X = df_ml[features_for_time + ['area_encoded']]
        y = df_ml['response_time_hours']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_regressor.predict(X_test)
        
        # Model performance
        col1, col2 = st.columns(2)
        
        with col1:
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.metric("Mean Absolute Error", f"{mae:.2f} hours")
            st.metric("R² Score", f"{r2:.3f}")
            
            # Actual vs Predicted
            fig_scatter = px.scatter(
                x=y_test, y=y_pred,
                title="Actual vs Predicted Response Time",
                labels={'x': 'Actual (hours)', 'y': 'Predicted (hours)'}
            )
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash", color="red"),
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max()
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_regressor.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance', y='feature',
                orientation='h',
                title="Feature Importance for Response Time"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    elif model_type == "📊 Clustering Analysis":
        st.subheader("📊 Clustering Analysis - Donation Patterns")
        
        # Prepare features for clustering
        features_for_clustering = ['quantity', 'distance_km', 'response_time_hours', 'urgency_score', 'estimated_people_served']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features_for_clustering])
        
        # K-means clustering
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        # Visualize clusters
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cluster = px.scatter(
                df_clustered, x='quantity', y='response_time_hours',
                color='Cluster', size='estimated_people_served',
                title="Donation Clusters (Quantity vs Response Time)",
                hover_data=['food_type', 'area']
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            # Cluster characteristics
            cluster_summary = df_clustered.groupby('Cluster')[features_for_clustering].mean()
            
            st.write("**Cluster Characteristics:**")
            st.dataframe(cluster_summary.round(2))
            
            # Cluster distribution
            cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
            fig_dist = px.bar(
                x=cluster_counts.index, y=cluster_counts.values,
                title="Distribution of Donations Across Clusters",
                labels={'x': 'Cluster', 'y': 'Number of Donations'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# Statistical Analysis
elif selected_page == "📊 Statistical Analysis":
    st.header("📊 Advanced Statistical Analysis")
    
    # Hypothesis testing
    st.subheader("🧪 Hypothesis Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Success Rate by Food Type**")
        success_by_type = df.groupby('food_type')['was_successful'].agg(['mean', 'count']).round(3)
        success_by_type.columns = ['Success_Rate', 'Count']
        st.dataframe(success_by_type)
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(df['food_type'], df['was_successful'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        if p_value < 0.05:
            st.success(f"Significant association found (p-value: {p_value:.4f})")
        else:
            st.info(f"No significant association (p-value: {p_value:.4f})")
    
    with col2:
        st.write("**Response Time by Area**")
        response_by_area = df.groupby('area')['response_time_hours'].agg(['mean', 'std', 'count']).round(2)
        st.dataframe(response_by_area)
        
        # ANOVA test
        from scipy.stats import f_oneway
        area_groups = [group['response_time_hours'].values for name, group in df.groupby('area')]
        f_stat, p_value_anova = f_oneway(*area_groups)
        
        if p_value_anova < 0.05:
            st.success(f"Significant difference between areas (p-value: {p_value_anova:.4f})")
        else:
            st.info(f"No significant difference between areas (p-value: {p_value_anova:.4f})")
    
    # Time series decomposition
    st.subheader("🕒 Time Series Analysis")
    
    # Daily aggregation for time series
    daily_data = df.groupby(df['date'].dt.date).agg({
        'donation_id': 'count',
        'was_successful': 'mean',
        'waste_prevented_kg': 'sum'
    }).reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Moving averages
    daily_data['donations_ma_7'] = daily_data['donation_id'].rolling(window=7, center=True).mean()
    daily_data['donations_ma_30'] = daily_data['donation_id'].rolling(window=30, center=True).mean()
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['donation_id'], 
                               mode='lines', name='Daily Donations', line=dict(color='lightblue')))
    fig_ts.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['donations_ma_7'], 
                               mode='lines', name='7-Day MA', line=dict(color='blue')))
    fig_ts.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['donations_ma_30'], 
                               mode='lines', name='30-Day MA', line=dict(color='red')))
    
    fig_ts.update_layout(title="Daily Donations with Moving Averages", 
                        xaxis_title="Date", yaxis_title="Number of Donations")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("🌤️ Seasonal & Weekly Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly patterns
        weekly_pattern = df.groupby('weekday').agg({
            'donation_id': 'count',
            'was_successful': 'mean'
        }).reset_index()
        
        days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekly_pattern['day_name'] = weekly_pattern['weekday'].map(days_map)
        
        fig_weekly = px.bar(weekly_pattern, x='day_name', y='donation_id',
                           title="Donations by Day of Week")
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        # Monthly patterns
        monthly_pattern = df.groupby('month').agg({
            'donation_id': 'count',
            'waste_prevented_kg': 'sum'
        }).reset_index()
        
        fig_monthly = px.line(monthly_pattern, x='month', y='donation_id',
                             title="Donations by Month")
        st.plotly_chart(fig_monthly, use_container_width=True)

# Geospatial Analytics
elif selected_page == "🗺️ Geospatial Analytics":
    st.header("🗺️ Geospatial Analytics & Location Intelligence")
    
    # Generate coordinates for areas (mock data)
    area_coordinates = {
        'Race Course': [11.0168, 76.9558],
        'RS Puram': [11.0096, 76.9750],
        'Gandhipuram': [11.0183, 76.9725],
        'Peelamedu': [11.0296, 76.9378],
        'Saibaba Colony': [11.0240, 76.9350],
        'Singanallur': [11.0400, 76.9200],
        'Vadavalli': [11.0100, 76.9100],
        'Ukkadam': [11.0050, 76.9400],
        'Town Hall': [11.0170, 76.9600],
        'Coimbatore North': [11.0300, 76.9500]
    }
    
    # Create enhanced dataset with coordinates
    df_geo = df.copy()
    df_geo['lat'] = df_geo['area'].map(lambda x: area_coordinates[x][0] + np.random.normal(0, 0.002))
    df_geo['lon'] = df_geo['area'].map(lambda x: area_coordinates[x][1] + np.random.normal(0, 0.002))
    
    # Area performance analysis
    area_stats = df.groupby('area').agg({
        'donation_id': 'count',
        'was_successful': 'mean',
        'response_time_hours': 'mean',
        'waste_prevented_kg': 'sum',
        'estimated_people_served': 'sum'
    }).round(2)
    
    area_stats['success_rate'] = (area_stats['was_successful'] * 100).round(1)
    
    # Add coordinates to area stats
    for area in area_stats.index:
        area_stats.loc[area, 'lat'] = area_coordinates[area][0]
        area_stats.loc[area, 'lon'] = area_coordinates[area][1]
    
    # Interactive map
    st.subheader("🗺️ Interactive Performance Map")
    
    fig_map = px.scatter_mapbox(
        area_stats.reset_index(),
        lat="lat", lon="lon",
        size="donation_id",
        color="success_rate",
        hover_name="area",
        hover_data={
            "donation_id": True,
            "response_time_hours": True,
            "waste_prevented_kg": True
        },
        color_continuous_scale="RdYlGn",
        size_max=30,
        zoom=11,
        title="Food Rescue Performance by Area"
    )
    
    fig_map.update_layout(
        mapbox_style="open-street-map",
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Distance analysis
    st.subheader("📏 Distance Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance vs success rate
        df_geo['distance_bins'] = pd.cut(df_geo['distance_km'], bins=5, labels=['<2km', '2-4km', '4-6km', '6-8km', '>8km'])
        distance_success = df_geo.groupby('distance_bins')['was_successful'].mean()
        
        fig_distance = px.bar(
            x=distance_success.index.astype(str), 
            y=distance_success.values,
            title="Success Rate by Distance",
            labels={'x': 'Distance Range', 'y': 'Success Rate'}
        )
        st.plotly_chart(fig_distance, use_container_width=True)
    
    with col2:
        # Response time vs distance
        fig_scatter_dist = px.scatter(
            df_geo, x='distance_km', y='response_time_hours',
            color='was_successful',
            title="Distance vs Response Time",
            trendline="ols"
        )
        st.plotly_chart(fig_scatter_dist, use_container_width=True)
    
    # Heat map analysis
    st.subheader("🔥 Demand Heat Map Analysis")
    
    # Create a grid for heatmap
    lat_bins = np.linspace(df_geo['lat'].min(), df_geo['lat'].max(), 20)
    lon_bins = np.linspace(df_geo['lon'].min(), df_geo['lon'].max(), 20)
    
    # Count donations in each grid cell
    heatmap_data = []
    for i in range(len(lat_bins)-1):
        for j in range(len(lon_bins)-1):
            mask = ((df_geo['lat'] >= lat_bins[i]) & (df_geo['lat'] < lat_bins[i+1]) & 
                   (df_geo['lon'] >= lon_bins[j]) & (df_geo['lon'] < lon_bins[j+1]))
            count = mask.sum()
            if count > 0:
                heatmap_data.append({
                    'lat': (lat_bins[i] + lat_bins[i+1]) / 2,
                    'lon': (lon_bins[j] + lon_bins[j+1]) / 2,
                    'count': count
                })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        
        fig_heatmap = px.density_mapbox(
            heatmap_df, lat='lat', lon='lon', z='count',
            radius=10, zoom=11,
            mapbox_style="open-street-map",
            title="Donation Density Heatmap"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Data Science Reports
elif selected_page == "📋 Data Science Reports":
    st.header("📋 Comprehensive Data Science Reports")
    
    # Executive summary
    st.subheader("📊 Executive Summary")
    
    # Key insights
    total_impact = {
        'donations': len(df),
        'success_rate': df['was_successful'].mean() * 100,
        'avg_response_time': df['response_time_hours'].mean(),
        'total_waste_prevented': df['waste_prevented_kg'].sum(),
        'total_people_served': df[df['was_successful']]['estimated_people_served'].sum(),
        'total_co2_saved': df['co2_saved_kg'].sum(),
        'economic_value': df[df['was_successful']]['economic_value_inr'].sum()
    }
    
    st.markdown(f"""
    <div class="alert-success">
        <h4>Key Performance Indicators (Last 12 Months)</h4>
        <ul>
            <li><strong>Total Donations Processed:</strong> {total_impact['donations']:,}</li>
            <li><strong>Overall Success Rate:</strong> {total_impact['success_rate']:.1f}%</li>
            <li><strong>Average Response Time:</strong> {total_impact['avg_response_time']:.1f} hours</li>
            <li><strong>Food Waste Prevented:</strong> {total_impact['total_waste_prevented']:,.0f} kg</li>
            <li><strong>People Served:</strong> {total_impact['total_people_served']:,.0f}</li>
            <li><strong>CO₂ Emissions Saved:</strong> {total_impact['total_co2_saved']:,.0f} kg</li>
            <li><strong>Economic Value Created:</strong> ₹{total_impact['economic_value']:,.0f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Trend analysis
    st.subheader("📈 Trend Analysis & Insights")
    
    # Monthly trends
    monthly_trends = df.groupby(df['date'].dt.to_period('M')).agg({
        'donation_id': 'count',
        'was_successful': 'mean',
        'response_time_hours': 'mean',
        'waste_prevented_kg': 'sum'
    }).reset_index()
    
    monthly_trends['month'] = monthly_trends['date'].astype(str)
    
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Donations', 'Success Rate Trend', 'Response Time Trend', 'Waste Prevented'),
        vertical_spacing=0.12
    )
    
    fig_trends.add_trace(go.Scatter(x=monthly_trends['month'], y=monthly_trends['donation_id'], 
                                   mode='lines+markers', name='Donations'), row=1, col=1)
    fig_trends.add_trace(go.Scatter(x=monthly_trends['month'], y=monthly_trends['was_successful']*100, 
                                   mode='lines+markers', name='Success %'), row=1, col=2)
    fig_trends.add_trace(go.Scatter(x=monthly_trends['month'], y=monthly_trends['response_time_hours'], 
                                   mode='lines+markers', name='Response Time'), row=2, col=1)
    fig_trends.add_trace(go.Scatter(x=monthly_trends['month'], y=monthly_trends['waste_prevented_kg'], 
                                   mode='lines+markers', name='Waste Prevented'), row=2, col=2)
    
    fig_trends.update_layout(height=600, showlegend=False, title_text="12-Month Performance Trends")
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Performance benchmarks
    st.subheader("🎯 Performance Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performing areas
        top_areas = df.groupby('area').agg({
            'was_successful': 'mean',
            'response_time_hours': 'mean',
            'donation_id': 'count'
        }).round(2)
        top_areas = top_areas[top_areas['donation_id'] >= 20]  # Minimum sample size
        top_areas = top_areas.sort_values('was_successful', ascending=False)
        
        st.write("**Top Performing Areas:**")
        st.dataframe(top_areas.head().round(2))
    
    with col2:
        # Best food types
        top_food_types = df.groupby('food_type').agg({
            'was_successful': 'mean',
            'response_time_hours': 'mean',
            'donation_id': 'count'
        }).round(2)
        top_food_types = top_food_types[top_food_types['donation_id'] >= 10]
        top_food_types = top_food_types.sort_values('was_successful', ascending=False)
        
        st.write("**Best Performing Food Types:**")
        st.dataframe(top_food_types.round(2))
    
    # Recommendations
    st.subheader("💡 Data-Driven Recommendations")
    
    recommendations = []
    
    # Analyze patterns and generate recommendations
    avg_success_rate = df['was_successful'].mean()
    low_performing_areas = df.groupby('area')['was_successful'].mean()
    low_performing_areas = low_performing_areas[low_performing_areas < avg_success_rate - 0.1]
    
    if len(low_performing_areas) > 0:
        recommendations.append(f"Focus on improving operations in {', '.join(low_performing_areas.index)} - success rates below average")
    
    high_response_areas = df.groupby('area')['response_time_hours'].mean()
    high_response_areas = high_response_areas[high_response_areas > df['response_time_hours'].mean() + 0.5]
    
    if len(high_response_areas) > 0:
        recommendations.append(f"Deploy additional volunteers to {', '.join(high_response_areas.index)} to reduce response times")
    
    # Urgency analysis
    urgent_success = df[df['urgency_score'] >= 8]['was_successful'].mean()
    normal_success = df[df['urgency_score'] < 8]['was_successful'].mean()
    
    if urgent_success < normal_success:
        recommendations.append("Implement priority routing system for high-urgency donations to improve success rates")
    
    # Weekend analysis
    weekend_performance = df[df['weekday'].isin([5, 6])]['was_successful'].mean()
    weekday_performance = df[~df['weekday'].isin([5, 6])]['was_successful'].mean()
    
    if weekend_performance < weekday_performance:
        recommendations.append("Recruit more weekend volunteers to match weekday performance levels")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="alert-urgent">
            <strong>Recommendation {i}:</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

# Real-time Predictions
elif selected_page == "⚡ Real-time Predictions":
    st.header("⚡ Real-time Prediction Interface")
    
    st.markdown("""
    Use this interface to get real-time predictions for donation success, response times, 
    and optimal volunteer assignments based on current conditions.
    """)
    
    # Prediction interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location & Logistics")
        pred_area = st.selectbox("Area", df['area'].unique())
        pred_distance = st.slider("Distance to Recipient (km)", 0.5, 20.0, 5.0)
        current_hour = datetime.now().hour
        pred_hour = st.slider("Hour of Day", 0, 23, current_hour)
    
    with col2:
        st.subheader("🍽️ Food Details")
        pred_food_type = st.selectbox("Food Type", df['food_type'].unique())
        pred_quantity = st.number_input("Quantity", min_value=1, max_value=500, value=50)
        pred_expiry = st.slider("Hours Until Expiry", 1, 24, 4)
    
    with col3:
        st.subheader("🚨 Context")
        pred_weekday = st.slider("Day of Week (0=Mon)", 0, 6, datetime.now().weekday())
        pred_urgency = st.slider("Urgency Score", 1, 10, 10 - min(pred_expiry, 10))
        pred_donor_type = st.selectbox("Donor Type", df['donor_type'].unique())
    
    if st.button("🔮 Generate Predictions", type="primary"):
        # Prepare prediction data (simplified model for demo)
        
        # Success prediction based on historical patterns
        area_success_rate = df[df['area'] == pred_area]['was_successful'].mean()
        food_success_rate = df[df['food_type'] == pred_food_type]['was_successful'].mean()
        urgency_factor = min(pred_urgency / 10, 1.0)
        distance_penalty = max(0, (pred_distance - 5) * 0.05)
        
        predicted_success = min(1.0, (area_success_rate + food_success_rate) / 2 + urgency_factor * 0.2 - distance_penalty)
        
        # Response time prediction
        area_avg_response = df[df['area'] == pred_area]['response_time_hours'].mean()
        distance_factor = pred_distance * 0.1
        urgency_boost = max(0, (10 - pred_urgency) * 0.1)
        
        predicted_response_time = area_avg_response + distance_factor + urgency_boost
        
        # Display predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_percent = predicted_success * 100
            if success_percent > 75:
                st.success(f"**Success Probability: {success_percent:.1f}%**")
                st.write("✅ High likelihood of successful delivery")
            elif success_percent > 50:
                st.warning(f"**Success Probability: {success_percent:.1f}%**")
                st.write("⚠️ Moderate chance of success")
            else:
                st.error(f"**Success Probability: {success_percent:.1f}%**")
                st.write("❌ High risk of failure")
        
        with col2:
            if predicted_response_time < 2:
                st.success(f"**Estimated Response: {predicted_response_time:.1f}h**")
                st.write("⚡ Fast response expected")
            elif predicted_response_time < 4:
                st.warning(f"**Estimated Response: {predicted_response_time:.1f}h**")
                st.write("⏱️ Moderate response time")
            else:
                st.error(f"**Estimated Response: {predicted_response_time:.1f}h**")
                st.write("🐌 Slow response expected")
        
        with col3:
            estimated_impact = int(pred_quantity * 0.8) if pred_food_type == 'Cooked Meals' else int(pred_quantity * 0.6)
            st.info(f"**Estimated Impact:**")
            st.write(f"👥 {estimated_impact} people served")
            st.write(f"♻️ {pred_quantity * 0.5:.0f}kg waste prevented")
        
        # Optimization suggestions
        st.subheader("🎯 Optimization Suggestions")
        
        suggestions = []
        
        if predicted_success < 0.7:
            suggestions.append("Consider finding a closer recipient organization to improve success rate")
        
        if predicted_response_time > 3:
            suggestions.append("Alert additional volunteers in the area to reduce response time")
        
        if pred_urgency > 7 and predicted_response_time > 2:
            suggestions.append("This is high priority - consider emergency volunteer dispatch")
        
        if pred_expiry < 3:
            suggestions.append("Immediate action required - food expires soon")
        
        for suggestion in suggestions:
            st.markdown(f"💡 {suggestion}")
    
    # Real-time dashboard simulation
    st.subheader("📊 Live System Status")
    
    # Simulate real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_donations = np.random.randint(15, 35)
        st.metric("Active Donations", active_donations, delta=np.random.randint(-3, 5))
    
    with col2:
        available_volunteers = np.random.randint(8, 25)
        st.metric("Available Volunteers", available_volunteers, delta=np.random.randint(-2, 4))
    
    with col3:
        avg_wait_time = np.random.uniform(1.5, 3.5)
        st.metric("Avg Wait Time", f"{avg_wait_time:.1f}h", delta=f"{np.random.uniform(-0.5, 0.5):.1f}h")
    
    with col4:
        system_efficiency = np.random.uniform(75, 95)
        st.metric("System Efficiency", f"{system_efficiency:.1f}%", delta=f"{np.random.uniform(-2, 3):.1f}%")

# Footer with additional information
st.markdown("---")
st.markdown("""
### 🔬 About This Data Science Platform

This comprehensive analytics platform demonstrates advanced data science techniques applied to food rescue operations:

**Data Science Techniques Used:**
- **Machine Learning**: Random Forest models for success prediction and response time estimation
- **Statistical Analysis**: Hypothesis testing, ANOVA, correlation analysis
- **Time Series Analysis**: Trend decomposition, seasonal pattern detection
- **Clustering**: K-means clustering to identify donation patterns
- **Geospatial Analytics**: Location-based performance analysis and heat mapping
- **Predictive Modeling**: Real-time prediction interfaces for operational decisions

**Technical Stack:**
- **Data Processing**: Pandas, NumPy for data manipulation and analysis
- **Machine Learning**: Scikit-learn for predictive modeling and clustering
- **Visualization**: Plotly, Matplotlib, Seaborn for interactive charts
- **Statistical Testing**: SciPy for hypothesis testing and statistical analysis
- **Geospatial**: Folium for interactive mapping and location intelligence

**Business Impact:**
- Optimize volunteer allocation and routing
- Predict donation success rates to improve resource allocation  
- Identify high-performing areas and food types for strategic planning
- Enable data-driven decision making for operational efficiency

---
*Built with Python, Streamlit, and advanced data science libraries | For technical details, see the source code*
""")

# Sidebar with data science tools
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🛠️ Data Science Tools")
    st.markdown("""
    **Analysis Features:**
    - 📊 Statistical Testing
    - 🤖 Machine Learning Models  
    - 🕒 Time Series Analysis
    - 🗺️ Geospatial Analytics
    - 📈 Predictive Modeling
    - 🎯 Real-time Predictions
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    # Real-time metrics in sidebar
    total_records = len(df)
    date_range = (df['date'].max() - df['date'].min()).days
    success_rate = df['was_successful'].mean() * 100
    
    st.markdown(f"""
    📊 **Dataset Size:** {total_records:,} records  
    📅 **Date Range:** {date_range} days  
    ✅ **Success Rate:** {success_rate:.1f}%  
    🏘️ **Areas Covered:** {df['area'].nunique()}  
    🍽️ **Food Types:** {df['food_type'].nunique()}
    """)
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
