import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import datetime
from datetime import timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FoodBridge - AI Analytics Platform",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class FoodBridgeDataScience:
    def __init__(self):
        self.initialize_data()
        self.load_or_train_models()
    
    def initialize_data(self):
        """Generate comprehensive synthetic dataset for food rescue analytics"""
        np.random.seed(42)
        
        # Generate donors data
        self.donors_data = self._generate_donors_data()
        
        # Generate historical donations data
        self.donations_data = self._generate_donations_data()
        
        # Generate volunteer data
        self.volunteers_data = self._generate_volunteers_data()
        
        # Generate external factors
        self.external_factors = self._generate_external_factors()
    
    def _generate_donors_data(self):
        """Generate donor profiles with features for ML"""
        donors = []
        donor_types = ['Restaurant', 'Bakery', 'Grocery Store', 'Catering', 'Hotel', 'Cafe']
        locations = [(12.9716, 77.5946), (13.0827, 80.2707), (11.0168, 76.9558), 
                    (15.2993, 74.1240), (17.3850, 78.4867)]
        
        for i in range(200):  # Reduced for simplicity
            location = locations[i % len(locations)]
            donor_type = np.random.choice(donor_types)
            
            donor = {
                'donor_id': f'D_{i:03d}',
                'name': f'{donor_type}_{i}',
                'type': donor_type,
                'latitude': location[0] + np.random.normal(0, 0.1),
                'longitude': location[1] + np.random.normal(0, 0.1),
                'size_score': np.random.uniform(1, 10),
                'sustainability_score': np.random.uniform(1, 10),
                'avg_daily_footfall': np.random.randint(50, 1000),
                'years_operating': np.random.randint(1, 20),
                'max_donation_capacity': np.random.randint(10, 200)
            }
            donors.append(donor)
        
        return pd.DataFrame(donors)
    
    def _generate_donations_data(self):
        """Generate historical donations with temporal patterns"""
        donations = []
        
        # Generate 1 year of data (reduced for simplicity)
        start_date = datetime.datetime.now() - timedelta(days=365)
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            # More donations on weekends
            day_multiplier = 1.5 if current_date.weekday() >= 5 else 1.0
            
            # Seasonal patterns
            month_multiplier = 1.2 if current_date.month in [11, 12, 1] else 1.0
            
            # Daily donations
            num_donations = int(np.random.poisson(10) * day_multiplier * month_multiplier)
            
            for _ in range(num_donations):
                donor_id = np.random.choice(self.donors_data['donor_id'])
                donor_info = self.donors_data[self.donors_data['donor_id'] == donor_id].iloc[0]
                
                food_categories = ['Prepared Food', 'Bakery Items', 'Fruits', 'Vegetables', 
                                 'Dairy', 'Packaged Food', 'Beverages']
                food_type = np.random.choice(food_categories)
                
                base_quantity = donor_info['size_score'] * np.random.uniform(0.5, 2)
                quantity_kg = max(1, np.random.normal(base_quantity, base_quantity * 0.3))
                
                urgency_map = {'Prepared Food': 4, 'Dairy': 6, 'Bakery Items': 24, 
                              'Fruits': 48, 'Vegetables': 72, 'Packaged Food': 168, 'Beverages': 240}
                hours_to_expire = np.random.normal(urgency_map[food_type], urgency_map[food_type] * 0.3)
                hours_to_expire = max(1, hours_to_expire)
                
                success_probability = min(0.95, 0.5 + 
                                        (donor_info['sustainability_score'] / 20) +
                                        (1 / max(1, hours_to_expire / 24)) * 0.3)
                
                donation = {
                    'donation_id': f'DN_{len(donations):06d}',
                    'donor_id': donor_id,
                    'date': current_date,
                    'food_type': food_type,
                    'quantity_kg': round(quantity_kg, 2),
                    'hours_to_expire': round(hours_to_expire, 1),
                    'pickup_success': np.random.random() < success_probability,
                    'people_fed_estimate': int(quantity_kg * np.random.uniform(2, 4)),
                    'day_of_week': current_date.weekday(),
                    'month': current_date.month,
                    'hour_posted': np.random.randint(6, 22),
                    'weather_condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy']),
                    'latitude': donor_info['latitude'],
                    'longitude': donor_info['longitude']
                }
                donations.append(donation)
        
        return pd.DataFrame(donations)
    
    def _generate_volunteers_data(self):
        """Generate volunteer profiles"""
        volunteers = []
        
        for i in range(50):  # Reduced for simplicity
            location = np.random.choice([(12.9716, 77.5946), (13.0827, 80.2707)])
            
            volunteer = {
                'volunteer_id': f'V_{i:03d}',
                'name': f'Volunteer_{i}',
                'latitude': location[0] + np.random.normal(0, 0.05),
                'longitude': location[1] + np.random.normal(0, 0.05),
                'experience_months': np.random.randint(1, 60),
                'avg_pickups_per_week': np.random.randint(1, 10),
                'success_rate': np.random.uniform(0.7, 0.98)
            }
            volunteers.append(volunteer)
        
        return pd.DataFrame(volunteers)
    
    def _generate_external_factors(self):
        """Generate weather and external factors data"""
        factors = []
        start_date = datetime.datetime.now() - timedelta(days=365)
        
        for day in range(365):
            current_date = start_date + timedelta(days=day)
            
            factor = {
                'date': current_date,
                'temperature': np.random.normal(25, 8),
                'rainfall': max(0, np.random.exponential(2)),
                'humidity': np.random.uniform(40, 90),
                'festival_day': np.random.random() < 0.05,
                'public_holiday': np.random.random() < 0.03
            }
            factors.append(factor)
        
        return pd.DataFrame(factors)
    
    def load_or_train_models(self):
        """Load or train ML models"""
        self.prepare_ml_features()
        self.demand_model = self.train_demand_prediction_model()
        self.success_model = self.train_success_prediction_model()
        self.cluster_model = self.train_clustering_model()
        self.anomaly_model = self.train_anomaly_detection_model()
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        self.donations_data['date_only'] = self.donations_data['date'].dt.date
        self.external_factors['date_only'] = self.external_factors['date'].dt.date
        
        self.ml_data = self.donations_data.merge(
            self.external_factors[['date_only', 'temperature', 'rainfall', 'humidity', 'festival_day', 'public_holiday']],
            on='date_only',
            how='left'
        )
        
        self.ml_data = self.ml_data.merge(
            self.donors_data[['donor_id', 'size_score', 'sustainability_score', 'avg_daily_footfall', 'type']],
            on='donor_id',
            how='left'
        )
    
    def train_demand_prediction_model(self):
        """Train model to predict food demand"""
        daily_data = self.ml_data.groupby('date_only').agg({
            'quantity_kg': 'sum',
            'temperature': 'mean',
            'rainfall': 'mean',
            'humidity': 'mean',
            'festival_day': 'first',
            'public_holiday': 'first'
        }).reset_index()
        
        daily_data['day_of_week'] = pd.to_datetime(daily_data['date_only']).dt.dayofweek
        daily_data['month'] = pd.to_datetime(daily_data['date_only']).dt.month
        
        features = ['temperature', 'rainfall', 'humidity', 'festival_day', 
                   'public_holiday', 'day_of_week', 'month']
        
        X = daily_data[features].fillna(0)
        y = daily_data['quantity_kg']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    
    def train_success_prediction_model(self):
        """Train model to predict pickup success"""
        le_food = LabelEncoder()
        le_type = LabelEncoder()
        
        features_data = self.ml_data.copy()
        features_data['food_type_encoded'] = le_food.fit_transform(features_data['food_type'])
        features_data['type_encoded'] = le_type.fit_transform(features_data['type'])
        
        features = ['quantity_kg', 'hours_to_expire', 'size_score', 'sustainability_score',
                   'temperature', 'rainfall', 'humidity', 'day_of_week', 'hour_posted',
                   'food_type_encoded', 'type_encoded']
        
        X = features_data[features].fillna(0)
        y = features_data['pickup_success'].astype(int)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    
    def train_clustering_model(self):
        """Train clustering model for donor segmentation"""
        donor_features = self.donors_data[['size_score', 'sustainability_score', 
                                          'avg_daily_footfall', 'years_operating']].fillna(0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(donor_features)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        return {'model': kmeans, 'scaler': scaler, 'clusters': clusters}
    
    def train_anomaly_detection_model(self):
        """Train anomaly detection model"""
        features = ['quantity_kg', 'hours_to_expire', 'people_fed_estimate', 'day_of_week', 'hour_posted']
        
        X = self.ml_data[features].fillna(0)
        
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        
        return model

def main():
    st.markdown('<h1 class="main-header">🍲 FoodBridge AI Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Initialize the data science system
    if 'foodbridge_ds' not in st.session_state:
        with st.spinner("Initializing AI models and loading data..."):
            st.session_state.foodbridge_ds = FoodBridgeDataScience()
    
    fb_ds = st.session_state.foodbridge_ds
    
    # Sidebar navigation
    st.sidebar.title("🧠 AI Analytics Dashboard")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Overview & KPIs", "🔮 Predictive Analytics", "👥 Donor Segmentation", 
         "🗺️ Geographic Analysis", "📈 Time Series Analysis", "🔍 Anomaly Detection"]
    )
    
    if page == "📊 Overview & KPIs":
        show_overview_dashboard(fb_ds)
    elif page == "🔮 Predictive Analytics":
        show_predictive_analytics(fb_ds)
    elif page == "👥 Donor Segmentation":
        show_donor_segmentation(fb_ds)
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(fb_ds)
    elif page == "📈 Time Series Analysis":
        show_time_series_analysis(fb_ds)
    elif page == "🔍 Anomaly Detection":
        show_anomaly_detection(fb_ds)

def show_overview_dashboard(fb_ds):
    """Show main KPI dashboard"""
    st.header("📊 Advanced Analytics Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_donations = len(fb_ds.donations_data)
    success_rate = fb_ds.donations_data['pickup_success'].mean()
    total_food_rescued = fb_ds.donations_data['quantity_kg'].sum()
    people_fed = fb_ds.donations_data['people_fed_estimate'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Total Donations</h3>
            <h2>{total_donations:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Success Rate</h3>
            <h2>{success_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🍽️ Food Rescued</h3>
            <h2>{total_food_rescued:,.0f} kg</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 People Fed</h3>
            <h2>{people_fed:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily Food Rescue Trends")
        daily_trends = fb_ds.donations_data.groupby(fb_ds.donations_data['date'].dt.date)['quantity_kg'].sum().reset_index()
        daily_trends.columns = ['Date', 'Quantity']
        
        fig = px.line(daily_trends, x='Date', y='Quantity', title="Daily Food Rescue Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Success Rate by Food Type")
        success_by_type = fb_ds.donations_data.groupby('food_type').agg({
            'pickup_success': 'mean',
            'quantity_kg': 'count'
        }).reset_index()
        success_by_type.columns = ['Food Type', 'Success Rate', 'Count']
        
        fig = px.bar(success_by_type, x='Food Type', y='Success Rate',
                    title="Pickup Success Rate by Food Category",
                    color='Success Rate', color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def show_predictive_analytics(fb_ds):
    """Show predictive analytics dashboard"""
    st.header("🔮 AI-Powered Predictive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Food Demand Prediction")
        
        temp = st.slider("Temperature (°C)", 10, 40, 25)
        rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0)
        humidity = st.slider("Humidity (%)", 40, 90, 70)
        is_festival = st.checkbox("Festival Day")
        is_holiday = st.checkbox("Public Holiday")
        day_of_week = st.selectbox("Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday'])
        month = st.selectbox("Month", range(1, 13))
        
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_num = day_mapping[day_of_week]
        
        features = np.array([[temp, rainfall, humidity, is_festival, is_holiday, day_num, month]])
        prediction = fb_ds.demand_model.predict(features)[0]
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🎯 Predicted Daily Demand</h3>
            <h2>{prediction:.1f} kg</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("✅ Pickup Success Probability")
        
        quantity = st.number_input("Quantity (kg)", 1.0, 100.0, 10.0)
        hours_expire = st.number_input("Hours to Expire", 1.0, 168.0, 24.0)
        donor_size = st.slider("Donor Size Score", 1.0, 10.0, 5.0)
        sustainability = st.slider("Sustainability Score", 1.0, 10.0, 5.0)
        hour_posted = st.slider("Hour Posted", 0, 23, 12)
        
        # Simple encoding
        food_type_val = 1  # Default value
        donor_type_val = 1  # Default value
        
        success_features = np.array([[quantity, hours_expire, donor_size, sustainability,
                                    temp, rainfall, humidity, day_num, hour_posted,
                                    food_type_val, donor_type_val]])
        
        success_prob = fb_ds.success_model.predict(success_features)[0]
        success_prob = max(0, min(1, success_prob))  # Clamp between 0 and 1
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3>📈 Success Probability</h3>
            <h2>{success_prob:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)

def show_donor_segmentation(fb_ds):
    """Show donor segmentation analysis"""
    st.header("👥 AI-Powered Donor Segmentation")
    
    clusters = fb_ds.cluster_model['clusters']
    cluster_data = fb_ds.donors_data.copy()
    cluster_data['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Donor Clusters")
        
        fig = px.scatter(cluster_data, x='size_score', y='sustainability_score',
                        color='Cluster', hover_data=['type', 'avg_daily_footfall'],
                        title="Donor Segmentation: Size vs Sustainability")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Cluster Statistics")
        
        cluster_stats = cluster_data.groupby('Cluster').agg({
            'size_score': 'mean',
            'sustainability_score': 'mean',
            'avg_daily_footfall': 'mean',
            'years_operating': 'mean'
        }).round(2)
        
        st.dataframe(cluster_stats)

def show_geographic_analysis(fb_ds):
    """Show geographic analysis"""
    st.header("🗺️ Geographic Intelligence")
    
    center_lat = fb_ds.donations_data['latitude'].mean()
    center_lon = fb_ds.donations_data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add sample markers
    for i, row in fb_ds.donors_data.head(20).iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"Type: {row['type']}<br>Size: {row['size_score']:.1f}",
            color='blue',
            fillColor='blue',
            fillOpacity=0.6
        ).add_to(m)
    
    st.subheader("📍 Donor Location Map")
    st_folium(m, width=700, height=500)
    
    # Geographic statistics
    st.subheader("📊 Geographic Distribution")
    
    geo_stats = fb_ds.donations_data.groupby('weather_condition').agg({
        'quantity_kg': 'sum',
        'pickup_success': 'mean'
    }).reset_index()
    
    fig = px.bar(geo_stats, x='weather_condition', y='quantity_kg',
                title="Food Donations by Weather Condition")
    st.plotly_chart(fig, use_container_width=True)

def show_time_series_analysis(fb_ds):
    """Show time series analysis"""
    st.header("📈 Time Series Analysis")
    
    ts_data = fb_ds.donations_data.groupby(fb_ds.donations_data['date'].dt.date).agg({
        'quantity_kg': 'sum',
        'pickup_success': 'mean'
    }).reset_index()
    ts_data.columns = ['Date', 'Total_Quantity', 'Success_Rate']
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Daily Food Volume")
        fig = px.line(ts_data, x='Date', y='Total_Quantity', title="Daily Food Rescue Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Success Rate Trend")
        fig = px.line(ts_data, x='Date', y='Success_Rate', title="Daily Success Rate")
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly patterns
    st.subheader("📅 Weekly Patterns")
    fb_ds.donations_data['day_name'] = fb_ds.donations_data['date'].dt.day_name()
    weekly_pattern = fb_ds.donations_data.groupby('day_name')['quantity_kg'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    
    fig = px.bar(x=weekly_pattern.index, y=weekly_pattern.values,
                title="Average Food Volume by Day of Week")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly patterns
    monthly_pattern = fb_ds.donations_data.groupby('month')['quantity_kg'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = px.bar(x=[month_names[i-1] for i in monthly_pattern.index], y=monthly_pattern.values,
                title="Average Food Volume by Month")
    st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(fb_ds):
    """Show anomaly detection analysis"""
    st.header("🔍 AI-Powered Anomaly Detection")
    
    # Detect anomalies
    features = ['quantity_kg', 'hours_to_expire', 'people_fed_estimate', 'day_of_week', 'hour_posted']
    X = fb_ds.ml_data[features].fillna(0)
    
    anomaly_scores = fb_ds.anomaly_model.decision_function(X)
    anomalies = fb_ds.anomaly_model.predict(X)
    
    fb_ds.donations_data['anomaly_score'] = anomaly_scores
    fb_ds.donations_data['is_anomaly'] = anomalies == -1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Anomaly Detection Results")
        
        anomaly_count = fb_ds.donations_data['is_anomaly'].sum()
        anomaly_rate = anomaly_count / len(fb_ds.donations_data) * 100
        
        st.metric("Anomalies Detected", f"{anomaly_count}")
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # Show most anomalous donations
        st.subheader("🔍 Most Unusual Donations")
        if anomaly_count > 0:
            anomalous_donations = fb_ds.donations_data[fb_ds.donations_data['is_anomaly']].head(10)
            display_cols = ['donor_id', 'food_type', 'quantity_kg', 'hours_to_expire', 'pickup_success']
            st.dataframe(anomalous_donations[display_cols])
        else:
            st.info("No significant anomalies detected in current dataset.")
    
    with col2:
        st.subheader("📊 Anomaly Score Distribution")
        
        fig = px.histogram(fb_ds.donations_data, x='anomaly_score', 
                          color='is_anomaly', nbins=50,
                          title="Distribution of Anomaly Scores")
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D anomaly visualization
        fig = px.scatter(fb_ds.donations_data, x='quantity_kg', y='hours_to_expire',
                        color='is_anomaly', title="Anomalies in Quantity vs Expiry Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly patterns
    st.subheader("📈 Anomaly Patterns by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomalies by food type
        anomaly_by_type = fb_ds.donations_data.groupby('food_type')['is_anomaly'].agg(['count', 'sum']).reset_index()
        anomaly_by_type['anomaly_rate'] = anomaly_by_type['sum'] / anomaly_by_type['count']
        
        fig = px.bar(anomaly_by_type, x='food_type', y='anomaly_rate',
                    title="Anomaly Rate by Food Type")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomalies by hour
        anomaly_by_hour = fb_ds.donations_data.groupby('hour_posted')['is_anomaly'].agg(['count', 'sum']).reset_index()
        anomaly_by_hour['anomaly_rate'] = anomaly_by_hour['sum'] / anomaly_by_hour['count']
        
        fig = px.line(anomaly_by_hour, x='hour_posted', y='anomaly_rate',
                     title="Anomaly Rate by Hour Posted")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
