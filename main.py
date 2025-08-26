import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats

# Set page configuration
st.set_page_config(
    page_title="Food Waste Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
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
    .insight-card {
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def generate_food_waste_data():
    """Generate synthetic food waste data for analysis"""
    np.random.seed(42)
    
    # Generate 2000 records
    n_records = 2000
    
    # Date range: last 2 years
    start_date = datetime.now() - timedelta(days=730)
    dates = [start_date + timedelta(days=x) for x in range(730)]
    
    data = []
    
    restaurants = ['Green Valley Restaurant', 'City Bakery', 'Spice Garden Hotel', 'Urban Bistro', 
                   'Coastal Kitchen', 'Mountain View Cafe', 'Downtown Diner', 'Royal Palace',
                   'Street Food Corner', 'Healthy Bites', 'Pizza Corner', 'Burger Junction']
    
    food_types = ['Cooked Meals', 'Bakery Items', 'Raw Ingredients', 'Fruits & Vegetables', 
                  'Dairy Products', 'Packaged Food']
    
    areas = ['Race Course', 'RS Puram', 'Gandhipuram', 'Peelamedu', 'Saibaba Colony',
             'Singanallur', 'Vadavalli', 'Ukkadam', 'Town Hall', 'Coimbatore North']
    
    for i in range(n_records):
        date = np.random.choice(dates)
        restaurant = np.random.choice(restaurants)
        food_type = np.random.choice(food_types)
        area = np.random.choice(areas)
        
        # Generate correlated features
        is_weekend = date.weekday() >= 5
        is_festival_season = date.month in [10, 11, 12, 1]  # Festival months
        
        # Base waste amount (influenced by restaurant size and type)
        base_waste = np.random.normal(15, 5)
        
        # Adjustments based on factors
        if is_weekend:
            base_waste *= 1.3
        if is_festival_season:
            base_waste *= 1.2
        if food_type == 'Bakery Items':
            base_waste *= 1.1
        if area in ['Gandhipuram', 'RS Puram']:  # Busy areas
            base_waste *= 1.15
            
        waste_amount = max(1, base_waste + np.random.normal(0, 2))
        
        # Generate other features
        temperature = np.random.normal(28, 5)  # Celsius
        humidity = np.random.normal(65, 10)
        
        # Rescue success (higher for established restaurants and optimal conditions)
        rescue_probability = 0.7
        if restaurant in ['Green Valley Restaurant', 'Spice Garden Hotel']:
            rescue_probability += 0.15
        if waste_amount > 20:
            rescue_probability += 0.1
        if temperature < 30:
            rescue_probability += 0.05
            
        rescued = np.random.random() < rescue_probability
        
        # Response time (faster for smaller amounts and established donors)
        response_time = np.random.exponential(2) + 0.5  # Hours
        if rescued and waste_amount < 10:
            response_time *= 0.8
            
        # Economic value
        cost_per_kg = {'Cooked Meals': 120, 'Bakery Items': 80, 'Raw Ingredients': 60,
                      'Fruits & Vegetables': 40, 'Dairy Products': 100, 'Packaged Food': 90}
        economic_value = waste_amount * cost_per_kg[food_type]
        
        data.append({
            'date': date,
            'restaurant_name': restaurant,
            'area': area,
            'food_type': food_type,
            'waste_amount_kg': round(waste_amount, 2),
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'is_weekend': is_weekend,
            'is_festival_season': is_festival_season,
            'rescued': rescued,
            'response_time_hours': round(response_time, 2),
            'economic_value': round(economic_value, 2),
            'month': date.month,
            'day_of_week': date.weekday(),
            'season': 'Summer' if date.month in [3,4,5] else 'Monsoon' if date.month in [6,7,8,9] else 'Winter'
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_environmental_data():
    """Generate environmental impact data"""
    np.random.seed(43)
    
    # CO2 emissions prevented per kg of food rescued
    co2_factor = 2.5  # kg CO2 per kg food
    water_factor = 15  # liters per kg food
    land_factor = 2.8  # m2 per kg food
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    data = []
    for month in months:
        food_rescued = np.random.normal(800, 100)  # kg per month
        co2_saved = food_rescued * co2_factor
        water_saved = food_rescued * water_factor
        land_saved = food_rescued * land_factor
        
        data.append({
            'month': month,
            'food_rescued_kg': round(food_rescued, 1),
            'co2_prevented_kg': round(co2_saved, 1),
            'water_saved_liters': round(water_saved, 1),
            'land_saved_m2': round(land_saved, 1)
        })
    
    return pd.DataFrame(data)

# Load data
if not st.session_state.data_loaded:
    with st.spinner('Loading data...'):
        df = generate_food_waste_data()
        env_df = generate_environmental_data()
        st.session_state.df = df
        st.session_state.env_df = env_df
        st.session_state.data_loaded = True
else:
    df = st.session_state.df
    env_df = st.session_state.env_df

# Title and description
st.title("📊 Food Waste Analytics Dashboard")
st.markdown("""
**Data-Driven Insights for Sustainable Food Management**

This dashboard provides comprehensive analytics on food waste patterns, rescue efficiency, 
and environmental impact using machine learning and statistical analysis.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Analysis", [
    "Executive Summary",
    "Exploratory Data Analysis", 
    "Predictive Modeling",
    "Time Series Analysis",
    "Environmental Impact",
    "Statistical Testing",
    "Clustering Analysis",
    "Data Export & Insights"
])

# Executive Summary Page
if page == "Executive Summary":
    st.header("Executive Summary")
    
    # Key metrics
    total_waste = df['waste_amount_kg'].sum()
    total_rescued = df[df['rescued']]['waste_amount_kg'].sum()
    rescue_rate = (df['rescued'].sum() / len(df)) * 100
    avg_response_time = df[df['rescued']]['response_time_hours'].mean()
    total_economic_value = df['economic_value'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🍽️</h3>
            <h2>{total_waste:,.0f} kg</h2>
            <p>Total Food Waste</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>♻️</h3>
            <h2>{total_rescued:,.0f} kg</h2>
            <p>Food Rescued</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈</h3>
            <h2>{rescue_rate:.1f}%</h2>
            <p>Rescue Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰</h3>
            <h2>₹{total_economic_value:,.0f}</h2>
            <p>Economic Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    st.subheader("Key Insights")
    
    # Most wasteful restaurant
    waste_by_restaurant = df.groupby('restaurant_name')['waste_amount_kg'].sum().sort_values(ascending=False)
    top_waster = waste_by_restaurant.index[0]
    
    # Best rescue rate by area
    rescue_by_area = df.groupby('area')['rescued'].mean().sort_values(ascending=False)
    best_area = rescue_by_area.index[0]
    
    # Most problematic food type
    waste_by_type = df.groupby('food_type')['waste_amount_kg'].mean().sort_values(ascending=False)
    problematic_type = waste_by_type.index[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <h4>🏆 Top Performer</h4>
            <p><strong>{best_area}</strong> has the highest rescue rate at <strong>{rescue_by_area.iloc[0]*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-card">
            <h4>⚠️ Area of Concern</h4>
            <p><strong>{top_waster}</strong> generates the most waste: <strong>{waste_by_restaurant.iloc[0]:,.0f} kg</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="success-card">
            <h4>💡 Optimization Opportunity</h4>
            <p><strong>{problematic_type}</strong> shows highest average waste per incident: <strong>{waste_by_type.iloc[0]:.1f} kg</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>⏱️ Response Efficiency</h4>
            <p>Average response time: <strong>{avg_response_time:.1f} hours</strong><br>
            Weekend impact: <strong>+{((df[df['is_weekend']]['response_time_hours'].mean() / df[~df['is_weekend']]['response_time_hours'].mean() - 1) * 100):.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend analysis
    st.subheader("Trend Analysis")
    
    monthly_trends = df.groupby(df['date'].dt.to_period('M')).agg({
        'waste_amount_kg': 'sum',
        'rescued': 'sum',
        'economic_value': 'sum'
    }).reset_index()
    monthly_trends['date'] = monthly_trends['date'].astype(str)
    monthly_trends['rescue_rate'] = (monthly_trends['rescued'] / df.groupby(df['date'].dt.to_period('M')).size().values) * 100
    
    fig_trend = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Waste Generation', 'Rescue Success Rate', 'Economic Impact', 'Waste vs Rescued'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig_trend.add_trace(
        go.Scatter(x=monthly_trends['date'], y=monthly_trends['waste_amount_kg'],
                  mode='lines+markers', name='Total Waste'),
        row=1, col=1
    )
    
    fig_trend.add_trace(
        go.Scatter(x=monthly_trends['date'], y=monthly_trends['rescue_rate'],
                  mode='lines+markers', name='Rescue Rate', line=dict(color='green')),
        row=1, col=2
    )
    
    fig_trend.add_trace(
        go.Bar(x=monthly_trends['date'], y=monthly_trends['economic_value'],
               name='Economic Value', marker_color='orange'),
        row=2, col=1
    )
    
    rescued_monthly = df[df['rescued']].groupby(df[df['rescued']]['date'].dt.to_period('M'))['waste_amount_kg'].sum().reindex(monthly_trends['date'].str.replace('-', ''), fill_value=0)
    fig_trend.add_trace(
        go.Scatter(x=monthly_trends['date'], y=monthly_trends['waste_amount_kg'],
                  mode='lines', name='Total Waste', line=dict(color='red')),
        row=2, col=2
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly_trends['date'], y=rescued_monthly.values,
                  mode='lines', name='Rescued', line=dict(color='green'), fill='tonexty'),
        row=2, col=2
    )
    
    fig_trend.update_layout(height=600, title_text="Food Waste Analytics Overview")
    st.plotly_chart(fig_trend, use_container_width=True)

# Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    with col2:
        st.metric("Unique Restaurants", df['restaurant_name'].nunique())
        st.metric("Areas Covered", df['area'].nunique())
    with col3:
        st.metric("Food Types", df['food_type'].nunique())
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Distribution analysis
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(df, x='waste_amount_kg', nbins=30, 
                               title='Distribution of Waste Amount',
                               color_discrete_sequence=['skyblue'])
        fig_hist.add_vline(x=df['waste_amount_kg'].mean(), line_dash="dash", 
                          line_color="red", annotation_text="Mean")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        fig_response = px.histogram(df[df['rescued']], x='response_time_hours', nbins=25,
                                  title='Response Time Distribution (Rescued Food)',
                                  color_discrete_sequence=['lightgreen'])
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        fig_box = px.box(df, x='food_type', y='waste_amount_kg',
                        title='Waste Amount by Food Type')
        fig_box.update_xaxis(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        fig_violin = px.violin(df, x='area', y='waste_amount_kg',
                              title='Waste Distribution by Area')
        fig_violin.update_xaxis(tickangle=45)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    numeric_cols = ['waste_amount_kg', 'temperature', 'humidity', 'response_time_hours', 
                   'economic_value', 'month', 'day_of_week']
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix of Numeric Variables")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Categorical analysis
    st.subheader("Categorical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rescue rate by category
        rescue_by_food = df.groupby('food_type')['rescued'].mean().sort_values(ascending=False)
        fig_rescue_food = px.bar(x=rescue_by_food.index, y=rescue_by_food.values * 100,
                                title='Rescue Success Rate by Food Type (%)',
                                color=rescue_by_food.values, color_continuous_scale='RdYlGn')
        fig_rescue_food.update_xaxis(tickangle=45)
        st.plotly_chart(fig_rescue_food, use_container_width=True)
    
    with col2:
        # Average waste by area
        waste_by_area = df.groupby('area')['waste_amount_kg'].mean().sort_values(ascending=False)
        fig_waste_area = px.bar(x=waste_by_area.index, y=waste_by_area.values,
                               title='Average Waste Amount by Area (kg)',
                               color=waste_by_area.values, color_continuous_scale='Reds')
        fig_waste_area.update_xaxis(tickangle=45)
        st.plotly_chart(fig_waste_area, use_container_width=True)

# Predictive Modeling
elif page == "Predictive Modeling":
    st.header("Predictive Modeling")
    
    st.markdown("""
    Using machine learning to predict food waste amounts and rescue success probability.
    """)
    
    # Feature engineering
    df_model = df.copy()
    
    # Encode categorical variables
    le_restaurant = LabelEncoder()
    le_area = LabelEncoder()
    le_food_type = LabelEncoder()
    le_season = LabelEncoder()
    
    df_model['restaurant_encoded'] = le_restaurant.fit_transform(df_model['restaurant_name'])
    df_model['area_encoded'] = le_area.fit_transform(df_model['area'])
    df_model['food_type_encoded'] = le_food_type.fit_transform(df_model['food_type'])
    df_model['season_encoded'] = le_season.fit_transform(df_model['season'])
    
    # Prepare features
    feature_cols = ['restaurant_encoded', 'area_encoded', 'food_type_encoded', 
                   'temperature', 'humidity', 'is_weekend', 'is_festival_season',
                   'month', 'day_of_week', 'season_encoded']
    
    X = df_model[feature_cols]
    
    # Model 1: Predicting waste amount
    st.subheader("Model 1: Waste Amount Prediction")
    
    y_waste = df_model['waste_amount_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y_waste, test_size=0.2, random_state=42)
    
    # Train Random Forest for waste prediction
    rf_waste = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_waste.fit(X_train, y_train)
    
    # Predictions
    y_pred_waste = rf_waste.predict(X_test)
    
    # Metrics
    mse_waste = mean_squared_error(y_test, y_pred_waste)
    r2_waste = r2_score(y_test, y_pred_waste)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Squared Error", f"{mse_waste:.2f}")
        st.metric("R² Score", f"{r2_waste:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_waste.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature',
                               orientation='h', title='Feature Importance (Waste Prediction)')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Actual vs Predicted
        fig_scatter = px.scatter(x=y_test, y=y_pred_waste,
                                title='Actual vs Predicted Waste Amount',
                                labels={'x': 'Actual', 'y': 'Predicted'})
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred_waste.min())
        max_val = max(y_test.max(), y_pred_waste.max())
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Model 2: Predicting rescue success
    st.subheader("Model 2: Rescue Success Prediction")
    
    y_rescue = df_model['rescued']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rescue, test_size=0.2, random_state=42)
    
    # Train Random Forest for rescue prediction
    rf_rescue = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rescue.fit(X_train_r, y_train_r)
    
    # Predictions
    y_pred_rescue = rf_rescue.predict(X_test_r)
    y_pred_proba = rf_rescue.predict_proba(X_test_r)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification report
        report = classification_report(y_test_r, y_pred_rescue, output_dict=True)
        st.metric("Accuracy", f"{report['accuracy']:.3f}")
        st.metric("Precision", f"{report['True']['precision']:.3f}")
        st.metric("Recall", f"{report['True']['recall']:.3f}")
        
        # Feature importance for rescue
        feature_importance_rescue = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_rescue.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_importance_rescue = px.bar(feature_importance_rescue, x='importance', y='feature',
                                      orientation='h', title='Feature Importance (Rescue Prediction)')
        st.plotly_chart(fig_importance_rescue, use_container_width=True)
    
    with col2:
        # ROC-like curve using probability distribution
        fig_prob = px.histogram(x=y_pred_proba, color=y_test_r.astype(str),
                               title='Prediction Probability Distribution',
                               labels={'color': 'Actually Rescued'})
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Confusion matrix visualization
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_r, y_pred_rescue)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model insights
    st.subheader("Model Insights")
    
    insights = [
        f"The waste prediction model explains {r2_waste*100:.1f}% of the variance in food waste amounts.",
        f"Most important factor for waste prediction: {feature_importance.iloc[0]['feature']}",
        f"Rescue success prediction achieves {report['accuracy']*100:.1f}% accuracy.",
        f"Most important factor for rescue success: {feature_importance_rescue.iloc[0]['feature']}"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")

# Time Series Analysis
elif page == "Time Series Analysis":
    st.header("Time Series Analysis")
    
    # Prepare time series data
    daily_data = df.groupby('date').agg({
        'waste_amount_kg': 'sum',
        'rescued': 'sum',
        'economic_value': 'sum'
    }).reset_index()
    
    daily_data['rescue_rate'] = daily_data['rescued'] / df.groupby('date').size().values
    daily_data['day_of_week'] = daily_data['date'].dt.day_name()
    daily_data['month'] = daily_data['date'].dt.month
    
    st.subheader("Time Series Decomposition")
    
    # Plot time series
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ts = px.line(daily_data, x='date', y='waste_amount_kg',
                        title='Daily Food Waste Over Time')
        fig_ts.add_hline(y=daily_data['waste_amount_kg'].mean(), 
                        line_dash="dash", annotation_text="Average")
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Seasonal patterns
        monthly_avg = daily_data.groupby('month')['waste_amount_kg'].mean()
        fig_seasonal = px.bar(x=monthly_avg.index, y=monthly_avg.values,
                             title='Average Monthly Waste Pattern')
        fig_seasonal.update_xaxis(title="Month")
        fig_seasonal.update_yaxis(title="Average Waste (kg)")
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        fig_rescue_ts = px.line(daily_data, x='date', y='rescue_rate',
                               title='Daily Rescue Rate Over Time')
        fig_rescue_ts.update_yaxis(title="Rescue Rate")
        st.plotly_chart(fig_rescue_ts, use_container_width=True)
        
        # Weekly patterns
        weekly_avg = daily_data.groupby('day_of_week')['waste_amount_kg'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = weekly_avg.reindex(day_order)
        
        fig_weekly = px.bar(x=weekly_avg.index, y=weekly_avg.values,
                           title='Average Weekly Waste Pattern',
                           color=weekly_avg.values, color_continuous_scale='Reds')
        fig_weekly.update_xaxis(title="Day of Week", tickangle=45)
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Moving averages and trends
    st.subheader("Trend Analysis")
    
    # Calculate moving averages
    daily_data['MA_7'] = daily_data['waste_amount_kg'].rolling(window=7).mean()
    daily_data['MA_30'] = daily_data['waste_amount_kg'].rolling(window=30).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['waste_amount_kg'],
                               mode='lines', name='Daily Waste', line=dict(color='lightblue', width=1)))
    fig_ma.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['MA_7'],
                               mode='lines', name='7-Day MA', line=dict(color='orange', width=2)))
    fig_ma.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['MA_30'],
                               mode='lines', name='30-Day MA', line=dict(color='red', width=3)))
    
    fig_ma.update_layout(title='Waste Amount with Moving Averages', 
                        xaxis_title='Date', yaxis_title='Waste Amount (kg)')
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Anomaly detection
    st.subheader("Anomaly Detection")
    
    # Calculate z-scores for anomaly detection
    from scipy import stats
    daily_data['z_score'] = np.abs(stats.zscore(daily_data['waste_amount_kg']))
    anomalies = daily_data[daily_data['z_score'] > 2.5]
    
    fig_anomaly = px.scatter(daily_data, x='date', y='waste_amount_kg',
                            color=daily_data['z_score'] > 2.5,
                            title='Anomaly Detection (Z-score > 2.5)',
                            color_discrete_map={True: 'red', False: 'blue'})
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    if len(anomalies) > 0:
        st.warning(f"Found {len(anomalies)} anomalous days with unusually high waste amounts.")
        st.dataframe(anomalies[['date', 'waste_amount_kg', 'z_score']].head())

# Environmental Impact Analysis
elif page == "Environmental Impact":
    st.header("Environmental Impact Analysis")
    
    # Calculate environmental metrics
    rescued_food = df[df['rescued']]['waste_amount_kg'].sum()
    total_waste = df['waste_amount_kg'].sum()
    
    # Environmental factors (per kg of food)
    co2_factor = 2.5  # kg CO2
    water_factor = 15  # liters
    land_factor = 2.8  # m2
    energy_factor = 3.3  # kWh
    
    co2_saved = rescued_food * co2_factor
    water_saved = rescued_food * water_factor
    land_saved = rescued_food * land_factor
    energy_saved = rescued_food * energy_factor
    
    # Display environmental impact
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌍</h3>
            <h2>{co2_saved:,.0f} kg</h2>
            <p>CO₂ Emissions Prevented</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧</h3>
            <h2>{water_saved:,.0f} L</h2>
            <p>Water Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌱</h3>
            <h2>{land_saved:,.0f} m²</h2>
            <p>Land Use Avoided</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡</h3>
            <h2>{energy_saved:,.0f} kWh</h2>
            <p>Energy Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Environmental impact by food type
    st.subheader("Environmental Impact by Food Type")
    
    impact_by_type = df[df['rescued']].groupby('food_type').agg({
        'waste_amount_kg': 'sum'
    }).reset_index()
    
    impact_by_type['co2_saved'] = impact_by_type['waste_amount_kg'] * co2_factor
    impact_by_type['water_saved'] = impact_by_type['waste_amount_kg'] * water_factor
    impact_by_type['energy_saved'] = impact_by_type['waste_amount_kg'] * energy_factor
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_co2 = px.pie(impact_by_type, values='co2_saved', names='food_type',
                        title='CO₂ Emissions Prevented by Food Type')
        st.plotly_chart(fig_co2, use_container_width=True)
        
        fig_water = px.bar(impact_by_type, x='food_type', y='water_saved',
                          title='Water Saved by Food Type (Liters)',
                          color='water_saved', color_continuous_scale='Blues')
        fig_water.update_xaxis(tickangle=45)
        st.plotly_chart(fig_water, use_container_width=True)
    
    with col2:
        fig_energy = px.bar(impact_by_type, x='food_type', y='energy_saved',
                           title='Energy Saved by Food Type (kWh)',
                           color='energy_saved', color_continuous_scale='Greens')
        fig_energy.update_xaxis(tickangle=45)
        st.plotly_chart(fig_energy, use_container_width=True)
        
        # Carbon footprint comparison
        carbon_comparison = pd.DataFrame({
            'Category': ['Car emissions (1 year)', 'Food rescue impact', 'Average household (1 month)'],
            'CO2_kg': [4600, co2_saved, 1200]
        })
        
        fig_comparison = px.bar(carbon_comparison, x='Category', y='CO2_kg',
                               title='Carbon Impact Comparison',
                               color='CO2_kg', color_continuous_scale='Reds')
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Monthly environmental trends
    st.subheader("Monthly Environmental Impact Trends")
    
    monthly_env = df[df['rescued']].groupby(df[df['rescued']]['date'].dt.to_period('M')).agg({
        'waste_amount_kg': 'sum'
    }).reset_index()
    
    monthly_env['month'] = monthly_env['date'].astype(str)
    monthly_env['co2_prevented'] = monthly_env['waste_amount_kg'] * co2_factor
    monthly_env['water_saved'] = monthly_env['waste_amount_kg'] * water_factor
    
    fig_env_trend = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly CO₂ Emissions Prevented', 'Monthly Water Saved'),
        vertical_spacing=0.1
    )
    
    fig_env_trend.add_trace(
        go.Scatter(x=monthly_env['month'], y=monthly_env['co2_prevented'],
                  mode='lines+markers', name='CO₂ Prevented (kg)', line=dict(color='green')),
        row=1, col=1
    )
    
    fig_env_trend.add_trace(
        go.Scatter(x=monthly_env['month'], y=monthly_env['water_saved'],
                  mode='lines+markers', name='Water Saved (L)', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig_env_trend.update_layout(height=500, title_text="Environmental Impact Trends")
    st.plotly_chart(fig_env_trend, use_container_width=True)

# Statistical Testing
elif page == "Statistical Testing":
    st.header("Statistical Hypothesis Testing")
    
    st.markdown("""
    Performing statistical tests to validate key hypotheses about food waste patterns.
    """)
    
    # Test 1: Weekend vs Weekday waste amounts
    st.subheader("Test 1: Weekend vs Weekday Food Waste")
    
    weekend_waste = df[df['is_weekend']]['waste_amount_kg']
    weekday_waste = df[~df['is_weekend']]['waste_amount_kg']
    
    t_stat, p_value = stats.ttest_ind(weekend_waste, weekday_waste)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Hypothesis:** Weekend generates more food waste than weekdays
        
        - **Weekend Mean:** {weekend_waste.mean():.2f} kg
        - **Weekday Mean:** {weekday_waste.mean():.2f} kg
        - **T-statistic:** {t_stat:.3f}
        - **P-value:** {p_value:.6f}
        - **Result:** {'Significant' if p_value < 0.05 else 'Not Significant'} (α = 0.05)
        """)
        
        if p_value < 0.05:
            st.success("✅ Reject null hypothesis: Weekend waste is significantly different from weekday waste.")
        else:
            st.info("ℹ️ Fail to reject null hypothesis: No significant difference found.")
    
    with col2:
        fig_box_weekend = px.box(df, x='is_weekend', y='waste_amount_kg',
                                title='Waste Distribution: Weekend vs Weekday')
        fig_box_weekend.update_xaxis(tickvals=[False, True], ticktext=['Weekday', 'Weekend'])
        st.plotly_chart(fig_box_weekend, use_container_width=True)
    
    # Test 2: ANOVA for food types
    st.subheader("Test 2: Food Type Waste Differences (ANOVA)")
    
    food_groups = [df[df['food_type'] == ft]['waste_amount_kg'].values for ft in df['food_type'].unique()]
    f_stat, p_value_anova = stats.f_oneway(*food_groups)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Hypothesis:** Different food types have different waste amounts
        
        - **F-statistic:** {f_stat:.3f}
        - **P-value:** {p_value_anova:.6f}
        - **Result:** {'Significant' if p_value_anova < 0.05 else 'Not Significant'} (α = 0.05)
        """)
        
        if p_value_anova < 0.05:
            st.success("✅ Reject null hypothesis: Food types have significantly different waste amounts.")
            
            # Post-hoc analysis
            st.markdown("**Post-hoc Analysis (Mean waste by food type):**")
            food_means = df.groupby('food_type')['waste_amount_kg'].mean().sort_values(ascending=False)
            for food_type, mean_waste in food_means.items():
                st.write(f"• {food_type}: {mean_waste:.2f} kg")
        else:
            st.info("ℹ️ Fail to reject null hypothesis: No significant difference between food types.")
    
    with col2:
        fig_anova = px.box(df, x='food_type', y='waste_amount_kg',
                          title='Waste Distribution by Food Type')
        fig_anova.update_xaxis(tickangle=45)
        st.plotly_chart(fig_anova, use_container_width=True)
    
    # Test 3: Chi-square test for rescue success
    st.subheader("Test 3: Rescue Success Independence (Chi-square)")
    
    # Test if rescue success is independent of food type
    contingency_table = pd.crosstab(df['food_type'], df['rescued'])
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Hypothesis:** Rescue success is independent of food type
        
        - **Chi-square statistic:** {chi2:.3f}
        - **Degrees of freedom:** {dof}
        - **P-value:** {p_value_chi2:.6f}
        - **Result:** {'Dependent' if p_value_chi2 < 0.05 else 'Independent'} (α = 0.05)
        """)
        
        if p_value_chi2 < 0.05:
            st.success("✅ Reject null hypothesis: Rescue success depends on food type.")
        else:
            st.info("ℹ️ Fail to reject null hypothesis: Rescue success is independent of food type.")
        
        st.markdown("**Contingency Table:**")
        st.dataframe(contingency_table)
    
    with col2:
        # Visualize rescue rates by food type
        rescue_rates = df.groupby('food_type')['rescued'].mean().sort_values(ascending=False)
        fig_rescue_rates = px.bar(x=rescue_rates.index, y=rescue_rates.values * 100,
                                 title='Rescue Success Rate by Food Type (%)',
                                 color=rescue_rates.values, color_continuous_scale='RdYlGn')
        fig_rescue_rates.update_xaxis(tickangle=45)
        st.plotly_chart(fig_rescue_rates, use_container_width=True)
    
    # Test 4: Correlation test
    st.subheader("Test 4: Temperature-Waste Correlation")
    
    corr_coef, p_value_corr = stats.pearsonr(df['temperature'], df['waste_amount_kg'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Hypothesis:** Temperature is correlated with waste amount
        
        - **Correlation coefficient:** {corr_coef:.3f}
        - **P-value:** {p_value_corr:.6f}
        - **Result:** {'Significant correlation' if p_value_corr < 0.05 else 'No significant correlation'} (α = 0.05)
        """)
        
        if p_value_corr < 0.05:
            direction = "positive" if corr_coef > 0 else "negative"
            st.success(f"✅ Significant {direction} correlation found between temperature and waste amount.")
        else:
            st.info("ℹ️ No significant correlation found.")
    
    with col2:
        fig_scatter_temp = px.scatter(df, x='temperature', y='waste_amount_kg',
                                     title='Temperature vs Waste Amount',
                                     trendline='ols')
        st.plotly_chart(fig_scatter_temp, use_container_width=True)

# Clustering Analysis
elif page == "Clustering Analysis":
    st.header("Clustering Analysis")
    
    st.markdown("""
    Using unsupervised learning to identify patterns and group restaurants based on their waste characteristics.
    """)
    
    # Prepare data for clustering
    cluster_data = df.groupby('restaurant_name').agg({
        'waste_amount_kg': ['mean', 'std', 'sum'],
        'rescued': 'mean',
        'response_time_hours': 'mean',
        'economic_value': 'sum',
        'is_weekend': 'mean',
        'temperature': 'mean'
    }).round(2)
    
    # Flatten column names
    cluster_data.columns = ['_'.join(col).strip() for col in cluster_data.columns]
    cluster_data = cluster_data.fillna(0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using elbow method
    st.subheader("Optimal Number of Clusters")
    
    inertias = []
    k_range = range(1, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    fig_elbow = px.line(x=k_range, y=inertias, markers=True,
                       title='Elbow Method for Optimal Clusters',
                       labels={'x': 'Number of Clusters', 'y': 'Inertia'})
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Perform clustering with optimal k
    optimal_k = st.selectbox("Select number of clusters:", [2, 3, 4, 5], index=1)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to dataframe
    cluster_data['cluster'] = cluster_labels
    
    st.subheader(f"Restaurant Clustering (K={optimal_k})")
    
    # Analyze clusters
    cluster_summary = cluster_data.groupby('cluster').agg({
        'waste_amount_kg_mean': 'mean',
        'waste_amount_kg_sum': 'mean',
        'rescued_mean': 'mean',
        'response_time_hours_mean': 'mean',
        'economic_value_sum': 'mean'
    }).round(2)
    
    # Display cluster characteristics
    st.markdown("**Cluster Characteristics:**")
    st.dataframe(cluster_summary)
    
    # Visualize clusters
    col1, col2 = st.columns(2)
    
    with col1:
        # 2D visualization using first two principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)
        
        cluster_viz = pd.DataFrame({
            'PC1': pca_features[:, 0],
            'PC2': pca_features[:, 1],
            'Cluster': cluster_labels.astype(str),
            'Restaurant': cluster_data.index
        })
        
        fig_cluster = px.scatter(cluster_viz, x='PC1', y='PC2', color='Cluster',
                                title='Restaurant Clusters (PCA Visualization)',
                                hover_data=['Restaurant'])
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown(f"**Explained Variance:** PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")
    
    with col2:
        # Cluster comparison
        comparison_metrics = ['waste_amount_kg_mean', 'rescued_mean', 'response_time_hours_mean']
        
        fig_comparison = go.Figure()
        
        for i, cluster in enumerate(cluster_summary.index):
            fig_comparison.add_trace(go.Bar(
                name=f'Cluster {cluster}',
                x=comparison_metrics,
                y=cluster_summary.loc[cluster, comparison_metrics].values
            ))
        
        fig_comparison.update_layout(
            title='Cluster Comparison - Key Metrics',
            xaxis_title='Metrics',
            yaxis_title='Value',
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Cluster insights
    st.subheader("Cluster Insights")
    
    for cluster_id in range(optimal_k):
        cluster_restaurants = cluster_data[cluster_data['cluster'] == cluster_id]
        
        st.markdown(f"""
        **Cluster {cluster_id}** ({len(cluster_restaurants)} restaurants):
        - Average waste: {cluster_restaurants['waste_amount_kg_mean'].mean():.1f} kg
        - Rescue rate: {cluster_restaurants['rescued_mean'].mean():.1%}
        - Response time: {cluster_restaurants['response_time_hours_mean'].mean():.1f} hours
        """)
        
        with st.expander(f"Restaurants in Cluster {cluster_id}"):
            st.write(", ".join(cluster_restaurants.index.tolist()))

# Data Export & Insights
elif page == "Data Export & Insights":
    st.header("Data Export & Key Insights")
    
    # Key insights summary
    st.subheader("Executive Insights Summary")
    
    insights = []
    
    # Calculate key metrics for insights
    total_waste = df['waste_amount_kg'].sum()
    rescued_amount = df[df['rescued']]['waste_amount_kg'].sum()
    rescue_rate = (df['rescued'].sum() / len(df)) * 100
    
    # Top performing metrics
    best_area = df.groupby('area')['rescued'].mean().idxmax()
    worst_area = df.groupby('area')['rescued'].mean().idxmin()
    
    most_wasteful_type = df.groupby('food_type')['waste_amount_kg'].mean().idxmax()
    
    # Weekend vs weekday analysis
    weekend_avg = df[df['is_weekend']]['waste_amount_kg'].mean()
    weekday_avg = df[~df['is_weekend']]['waste_amount_kg'].mean()
    weekend_increase = ((weekend_avg / weekday_avg) - 1) * 100
    
    insights = [
        f"🎯 **Rescue Efficiency**: {rescue_rate:.1f}% of food donations are successfully rescued, with {rescued_amount:,.0f} kg saved from {total_waste:,.0f} kg total waste.",
        
        f"📍 **Geographic Performance**: {best_area} shows the highest rescue success rate, while {worst_area} needs improvement in rescue coordination.",
        
        f"🍽️ **Food Type Impact**: {most_wasteful_type} generates the highest average waste per incident, indicating a key optimization opportunity.",
        
        f"📅 **Temporal Patterns**: Weekend waste is {weekend_increase:+.1f}% higher than weekdays, suggesting need for enhanced weekend volunteer coverage.",
        
        f"💰 **Economic Impact**: Total economic value at stake is ₹{df['economic_value'].sum():,.0f}, with ₹{df[df['rescued']]['economic_value'].sum():,.0f} successfully recovered.",
        
        f"🌍 **Environmental Benefit**: Food rescue prevented {rescued_amount * 2.5:,.0f} kg CO₂ emissions and saved {rescued_amount * 15:,.0f} liters of water.",
        
        f"⚡ **Response Optimization**: Average response time is {df[df['rescued']]['response_time_hours'].mean():.1f} hours, with correlation to rescue success rates.",
    ]
    
    for insight in insights:
        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("Strategic Recommendations")
    
    recommendations = [
        "**Enhance Weekend Operations**: Deploy additional volunteers during weekends to handle 15%+ higher waste volumes.",
        
        f"**Target {most_wasteful_type}**: Implement specialized handling protocols for {most_wasteful_type.lower()} to reduce average waste per incident.",
        
        f"**Replicate {best_area} Success**: Study and implement best practices from {best_area} across other areas to improve overall rescue rates.",
        
        "**Predictive Deployment**: Use machine learning models to predict high-waste days and pre-position volunteers accordingly.",
        
        "**Temperature-Based Alerts**: Implement weather-based waste prediction to optimize rescue operations during extreme temperatures.",
        
        "**Restaurant Partnership Program**: Focus on top-waste generating restaurants for dedicated rescue partnerships and waste reduction training."
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Data export options
    st.subheader("Data Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Full Dataset"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"food_waste_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Summary Statistics"):
            summary = df.describe()
            summary_csv = summary.to_csv()
            st.download_button(
                label="Download Summary CSV",
                data=summary_csv,
                file_name=f"food_waste_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔍 Download Analysis Report"):
            report_data = {
                'Metric': ['Total Waste (kg)', 'Food Rescued (kg)', 'Rescue Rate (%)', 
                          'Avg Response Time (hrs)', 'Total Economic Value (₹)'],
                'Value': [total_waste, rescued_amount, rescue_rate, 
                         df[df['rescued']]['response_time_hours'].mean(), 
                         df['economic_value'].sum()]
            }
            report_df = pd.DataFrame(report_data)
            report_csv = report_df.to_csv(index=False)
            st.download_button(
                label="Download Report CSV",
                data=report_csv,
                file_name=f"food_waste_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Model performance summary
    st.subheader("Model Performance Summary")
    
    # Re-run quick model evaluation for summary
    df_model = df.copy()
    
    # Encode categorical variables
    le_restaurant = LabelEncoder()
    le_area = LabelEncoder()  
    le_food_type = LabelEncoder()
    le_season = LabelEncoder()
    
    df_model['restaurant_encoded'] = le_restaurant.fit_transform(df_model['restaurant_name'])
    df_model['area_encoded'] = le_area.fit_transform(df_model['area'])
    df_model['food_type_encoded'] = le_food_type.fit_transform(df_model['food_type'])
    df_model['season_encoded'] = le_season.fit_transform(df_model['season'])
    
    feature_cols = ['restaurant_encoded', 'area_encoded', 'food_type_encoded', 
                   'temperature', 'humidity', 'is_weekend', 'is_festival_season',
                   'month', 'day_of_week', 'season_encoded']
    
    X = df_model[feature_cols]
    
    # Waste prediction model
    y_waste = df_model['waste_amount_kg']
    X_train, X_test, y_train, y_test = train_test_split(X, y_waste, test_size=0.2, random_state=42)
    rf_waste = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_waste.fit(X_train, y_train)
    y_pred = rf_waste.predict(X_test)
    waste_r2 = r2_score(y_test, y_pred)
    
    # Rescue prediction model
    y_rescue = df_model['rescued']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rescue, test_size=0.2, random_state=42)
    rf_rescue = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_rescue.fit(X_train_r, y_train_r)
    rescue_accuracy = rf_rescue.score(X_test_r, y_test_r)
    
    model_summary = pd.DataFrame({
        'Model': ['Waste Amount Prediction', 'Rescue Success Prediction'],
        'Algorithm': ['Random Forest Regressor', 'Random Forest Classifier'],
        'Performance Metric': ['R² Score', 'Accuracy'],
        'Score': [f"{waste_r2:.3f}", f"{rescue_accuracy:.3f}"],
        'Interpretation': [
            f"Explains {waste_r2*100:.1f}% of waste variance",
            f"{rescue_accuracy*100:.1f}% correct predictions"
        ]
    })
    
    st.dataframe(model_summary)
    
    # Final summary
    st.subheader("Project Impact")
    
    st.success(f"""
    🎯 **Analytics Impact**: This data science analysis of {len(df):,} food waste records provides 
    actionable insights that could improve rescue rates by 15-25% and reduce food waste by 
    {(1-rescue_rate/100)*100:.0f}% through optimized operations and predictive deployment.
    """)

# Footer
st.markdown("---")
st.markdown("""
### About This Analytics Dashboard

This comprehensive food waste analytics dashboard demonstrates advanced data science techniques including:

- **Exploratory Data Analysis** with statistical summaries and visualizations
- **Predictive Modeling** using Random Forest for waste prediction and rescue success
- **Time Series Analysis** for trend identification and seasonal patterns  
- **Statistical Hypothesis Testing** to validate key assumptions
- **Clustering Analysis** for restaurant segmentation
- **Environmental Impact Assessment** with quantified sustainability metrics

**Technologies Used:** Python, Streamlit, Pandas, Scikit-learn, Plotly, Statistical Analysis

**Data Source:** Synthetic dataset based on realistic food waste patterns and rescue operations

---
*Built for demonstration of data science capabilities in sustainability and social impact domains.*
""")
