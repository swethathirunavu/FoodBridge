import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import time as tm
import folium
from streamlit_folium import st_folium
import random
from geopy.distance import geodesic
import hashlib

# Set page configuration
st.set_page_config(
    page_title="🍲 FoodBridge - Food Rescue Platform",
    page_icon="🍲",
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
    .volunteer-badge {
        background-color: #673ab7;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'food_donations' not in st.session_state:
    st.session_state.food_donations = []
if 'volunteers' not in st.session_state:
    st.session_state.volunteers = []
if 'organizations' not in st.session_state:
    st.session_state.organizations = []
if 'current_user_type' not in st.session_state:
    st.session_state.current_user_type = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Sample data initialization
def initialize_sample_data():
    if len(st.session_state.food_donations) == 0:
        sample_donations = [
            {
                'id': 'FD001',
                'donor_name': 'Green Valley Restaurant',
                'donor_phone': '+91-9876543210',
                'donor_address': 'Race Course Road, Coimbatore',
                'donor_lat': 11.0168,
                'donor_lng': 76.9558,
                'food_type': 'Cooked Meals',
                'quantity': '50 servings',
                'description': 'Freshly prepared vegetarian meals - rice, dal, vegetables',
                'expiry_time': datetime.now() + timedelta(hours=4),
                'status': 'Available',
                'created_at': datetime.now() - timedelta(hours=1),
                'claimed_by': None,
                'delivered_to': None,
                'special_instructions': 'Keep refrigerated, contains dairy',
                'food_category': 'Vegetarian'
            },
            {
                'id': 'FD002',
                'donor_name': 'City Bakery',
                'donor_phone': '+91-9876543211',
                'donor_address': 'RS Puram, Coimbatore',
                'donor_lat': 11.0096,
                'donor_lng': 76.9750,
                'food_type': 'Bakery Items',
                'quantity': '30 pieces',
                'description': 'Fresh bread, pastries, and cakes from today',
                'expiry_time': datetime.now() + timedelta(hours=12),
                'status': 'Claimed',
                'created_at': datetime.now() - timedelta(hours=3),
                'claimed_by': 'Volunteer001',
                'delivered_to': None,
                'special_instructions': 'Best consumed fresh',
                'food_category': 'Vegetarian'
            },
            {
                'id': 'FD003',
                'donor_name': 'Spice Garden Hotel',
                'donor_phone': '+91-9876543212',
                'donor_address': 'Gandhipuram, Coimbatore',
                'donor_lat': 11.0183,
                'donor_lng': 76.9725,
                'food_type': 'Cooked Meals',
                'quantity': '80 servings',
                'description': 'Mixed vegetarian and non-vegetarian meals',
                'expiry_time': datetime.now() + timedelta(hours=2),
                'status': 'Available',
                'created_at': datetime.now() - timedelta(minutes=30),
                'claimed_by': None,
                'delivered_to': None,
                'special_instructions': 'Urgent pickup required',
                'food_category': 'Mixed'
            }
        ]
        st.session_state.food_donations.extend(sample_donations)
    
    if len(st.session_state.volunteers) == 0:
        sample_volunteers = [
            {
                'id': 'VOL001',
                'name': 'Rajesh Kumar',
                'phone': '+91-9876543220',
                'area': 'Race Course',
                'vehicle': 'Two Wheeler',
                'availability': 'Available',
                'rating': 4.8,
                'completed_deliveries': 25,
                'location_lat': 11.0150,
                'location_lng': 76.9600
            },
            {
                'id': 'VOL002',
                'name': 'Priya Sharma',
                'phone': '+91-9876543221',
                'area': 'RS Puram',
                'vehicle': 'Four Wheeler',
                'availability': 'Busy',
                'rating': 4.9,
                'completed_deliveries': 42,
                'location_lat': 11.0100,
                'location_lng': 76.9750
            }
        ]
        st.session_state.volunteers.extend(sample_volunteers)
    
    if len(st.session_state.organizations) == 0:
        sample_orgs = [
            {
                'id': 'ORG001',
                'name': 'Hope Foundation',
                'contact_person': 'Maria Joseph',
                'phone': '+91-9876543230',
                'address': 'Peelamedu, Coimbatore',
                'type': 'NGO',
                'beneficiaries': 150,
                'location_lat': 11.0296,
                'location_lng': 76.9378,
                'requirements': 'Vegetarian meals preferred, serves elderly and children'
            },
            {
                'id': 'ORG002',
                'name': 'Street Children Shelter',
                'contact_person': 'David Wilson',
                'phone': '+91-9876543231',
                'address': 'Saibaba Colony, Coimbatore',
                'type': 'Shelter Home',
                'beneficiaries': 80,
                'location_lat': 11.0240,
                'location_lng': 76.9350,
                'requirements': 'All types of food accepted, urgent need for breakfast items'
            }
        ]
        st.session_state.organizations.extend(sample_orgs)

# Initialize data
initialize_sample_data()

# App title and description
st.title("🍲 FoodBridge - Connecting Surplus Food with Those in Need")
st.markdown("""
**Reducing Food Waste | Fighting Hunger | Building Community**

FoodBridge is a comprehensive platform that connects restaurants, grocery stores, and individuals with surplus food 
to charitable organizations and volunteers who can distribute it to those in need.
""")

# User authentication simulation
with st.sidebar:
    st.title("🔑 User Portal")
    user_type = st.selectbox("Login as:", ["Select User Type", "Restaurant/Donor", "Volunteer", "NGO/Organization", "Admin"])
    
    if user_type != "Select User Type":
        st.session_state.current_user_type = user_type
        
        if user_type == "Restaurant/Donor":
            st.session_state.current_user = "Green Valley Restaurant"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "Volunteer":
            st.session_state.current_user = "Rajesh Kumar"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "NGO/Organization":
            st.session_state.current_user = "Hope Foundation"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")
        elif user_type == "Admin":
            st.session_state.current_user = "System Admin"
            st.success(f"✅ Logged in as: {st.session_state.current_user}")

# Navigation
if st.session_state.current_user_type:
    if st.session_state.current_user_type == "Restaurant/Donor":
        pages = ["Dashboard", "Donate Food", "My Donations", "Analytics"]
    elif st.session_state.current_user_type == "Volunteer":
        pages = ["Dashboard", "Available Pickups", "My Deliveries", "Profile"]
    elif st.session_state.current_user_type == "NGO/Organization":
        pages = ["Dashboard", "Food Requests", "Received Donations", "Impact Report"]
    elif st.session_state.current_user_type == "Admin":
        pages = ["Dashboard", "All Donations", "Volunteers", "Organizations", "Analytics"]
    else:
        pages = ["Public Dashboard"]
    
    with st.sidebar:
        st.markdown("---")
        selected_page = st.radio("Navigate to:", pages)
else:
    selected_page = "Public Dashboard"

# Helper functions
def calculate_distance(lat1, lng1, lat2, lng2):
    return geodesic((lat1, lng1), (lat2, lng2)).kilometers

def get_urgency_color(expiry_time):
    time_diff = expiry_time - datetime.now()
    if time_diff.total_seconds() < 3600:  # Less than 1 hour
        return "🔴"
    elif time_diff.total_seconds() < 7200:  # Less than 2 hours
        return "🟡"
    else:
        return "🟢"

def format_time_remaining(expiry_time):
    time_diff = expiry_time - datetime.now()
    if time_diff.total_seconds() < 0:
        return "⚠️ EXPIRED"
    hours = int(time_diff.total_seconds() // 3600)
    minutes = int((time_diff.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m remaining"

# Page content based on user type and selection
if selected_page == "Public Dashboard" or selected_page == "Dashboard":
    # Key metrics
    available_donations = len([d for d in st.session_state.food_donations if d['status'] == 'Available'])
    total_served_estimate = sum([int(d['quantity'].split()[0]) if d['quantity'].split()[0].isdigit() else 0 
                               for d in st.session_state.food_donations if d['status'] == 'Delivered'])
    active_volunteers = len([v for v in st.session_state.volunteers if v['availability'] == 'Available'])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🍽️</h3>
            <h2>{available_donations}</h2>
            <p>Food Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥</h3>
            <h2>{total_served_estimate + 234}</h2>
            <p>People Fed Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚗</h3>
            <h2>{active_volunteers}</h2>
            <p>Active Volunteers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌱</h3>
            <h2>{len(st.session_state.food_donations) * 2.5:.1f}kg</h2>
            <p>Food Rescued</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time food availability map
    st.subheader("🗺️ Real-time Food Availability Map")
    
    # Create map centered on Coimbatore
    m = folium.Map(location=[11.0168, 76.9558], zoom_start=12)
    
    # Add markers for available food donations
    for donation in st.session_state.food_donations:
        if donation['status'] == 'Available':
            urgency = get_urgency_color(donation['expiry_time'])
            
            popup_text = f"""
            <b>{donation['donor_name']}</b><br>
            📞 {donation['donor_phone']}<br>
            🍽️ {donation['food_type']}<br>
            📦 {donation['quantity']}<br>
            ⏰ {format_time_remaining(donation['expiry_time'])}<br>
            📝 {donation['description']}
            """
            
            color = 'red' if urgency == '🔴' else 'orange' if urgency == '🟡' else 'green'
            
            folium.Marker(
                location=[donation['donor_lat'], donation['donor_lng']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
            ).add_to(m)
    
    # Add markers for organizations
    for org in st.session_state.organizations:
        popup_text = f"""
        <b>{org['name']}</b><br>
        👤 {org['contact_person']}<br>
        📞 {org['phone']}<br>
        🏢 {org['type']}<br>
        👥 Serves {org['beneficiaries']} people<br>
        📝 {org['requirements']}
        """
        
        folium.Marker(
            location=[org['location_lat'], org['location_lng']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500)
    
    # Urgent pickups alert
    urgent_donations = [d for d in st.session_state.food_donations 
                       if d['status'] == 'Available' and 
                       (d['expiry_time'] - datetime.now()).total_seconds() < 3600]
    
    if urgent_donations:
        st.markdown("### 🚨 Urgent Pickups Required!")
        for donation in urgent_donations:
            st.markdown(f"""
            <div class="alert-urgent">
                <strong>{donation['donor_name']}</strong> - {donation['food_type']} ({donation['quantity']})<br>
                📍 {donation['donor_address']}<br>
                ⏰ <strong>{format_time_remaining(donation['expiry_time'])}</strong><br>
                📝 {donation['special_instructions']}
            </div>
            """, unsafe_allow_html=True)

elif selected_page == "Donate Food":
    st.header("🍽️ Donate Food")
    
    with st.form("food_donation_form"):
        st.subheader("Food Donation Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            donor_name = st.text_input("Restaurant/Organization Name*", value="Green Valley Restaurant")
            donor_phone = st.text_input("Contact Phone*", value="+91-9876543210")
            donor_address = st.text_area("Pickup Address*", value="Race Course Road, Coimbatore")
            
        with col2:
            food_type = st.selectbox("Food Type*", 
                                   ["Cooked Meals", "Raw Ingredients", "Bakery Items", 
                                    "Fruits & Vegetables", "Packaged Food", "Dairy Products"])
            quantity = st.text_input("Quantity*", placeholder="e.g., 50 servings, 10kg, 20 pieces")
            food_category = st.selectbox("Food Category*", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
        
        description = st.text_area("Food Description*", 
                                 placeholder="Describe the food items in detail...")
        
        col3, col4 = st.columns(2)
        with col3:
            pickup_date = st.date_input("Pickup Date*", value=datetime.now().date())
            pickup_time = st.time_input("Pickup Time*", value=time(12, 0))
        
        with col4:
            expiry_date = st.date_input("Food Expiry Date*", value=datetime.now().date())
            expiry_time = st.time_input("Food Expiry Time*", value=time(18, 0))
        
        special_instructions = st.text_area("Special Instructions", 
                                          placeholder="Storage requirements, allergen information, etc.")
        
        # Location selection
        st.subheader("📍 Pickup Location")
        location_option = st.radio("How would you like to set the location?", 
                                 ["Use default location", "Select on map", "Enter coordinates"])
        
        if location_option == "Use default location":
            pickup_lat, pickup_lng = 11.0168, 76.9558
        elif location_option == "Enter coordinates":
            col5, col6 = st.columns(2)
            with col5:
                pickup_lat = st.number_input("Latitude", value=11.0168, format="%.6f")
            with col6:
                pickup_lng = st.number_input("Longitude", value=76.9558, format="%.6f")
        
        submitted = st.form_submit_button("🚀 Submit Food Donation", type="primary")
        
        if submitted:
            if all([donor_name, donor_phone, donor_address, food_type, quantity, description]):
                # Create donation entry
                donation_id = f"FD{len(st.session_state.food_donations) + 1:03d}"
                expiry_datetime = datetime.combine(expiry_date, expiry_time)
                
                new_donation = {
                    'id': donation_id,
                    'donor_name': donor_name,
                    'donor_phone': donor_phone,
                    'donor_address': donor_address,
                    'donor_lat': pickup_lat,
                    'donor_lng': pickup_lng,
                    'food_type': food_type,
                    'quantity': quantity,
                    'description': description,
                    'expiry_time': expiry_datetime,
                    'status': 'Available',
                    'created_at': datetime.now(),
                    'claimed_by': None,
                    'delivered_to': None,
                    'special_instructions': special_instructions,
                    'food_category': food_category
                }
                
                st.session_state.food_donations.append(new_donation)
                
                st.success(f"✅ Food donation submitted successfully! Donation ID: {donation_id}")
                st.balloons()
                
                st.info("""
                📧 **What happens next?**
                1. Your donation is now visible to volunteers and organizations
                2. You'll receive SMS/email notifications when a volunteer claims your donation
                3. The volunteer will contact you to coordinate pickup
                4. You'll get a confirmation when the food is successfully delivered
                """)
            else:
                st.error("Please fill in all required fields marked with *")

elif selected_page == "Available Pickups":
    st.header("🚗 Available Food Pickups")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        area_filter = st.selectbox("Filter by Area", ["All Areas", "Race Course", "RS Puram", "Gandhipuram", "Peelamedu"])
    with col2:
        food_filter = st.selectbox("Filter by Food Type", ["All Types", "Cooked Meals", "Bakery Items", "Raw Ingredients"])
    with col3:
        urgency_filter = st.selectbox("Filter by Urgency", ["All", "Urgent (< 1hr)", "Moderate (1-2hrs)", "Low (> 2hrs)"])
    
    # Get available donations
    available_donations = [d for d in st.session_state.food_donations if d['status'] == 'Available']
    
    if not available_donations:
        st.info("🎉 Great! No food pickups available right now. All food has been claimed!")
    else:
        st.success(f"📋 {len(available_donations)} food donations available for pickup")
        
        for donation in available_donations:
            urgency = get_urgency_color(donation['expiry_time'])
            time_remaining = format_time_remaining(donation['expiry_time'])
            
            # Calculate distance (assuming volunteer is at Race Course)
            volunteer_lat, volunteer_lng = 11.0150, 76.9600
            distance = calculate_distance(volunteer_lat, volunteer_lng, 
                                       donation['donor_lat'], donation['donor_lng'])
            
            with st.container():
                st.markdown(f"""
                <div class="food-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4>{urgency} {donation['donor_name']} - {donation['food_type']}</h4>
                        <span class="status-available">AVAILABLE</span>
                    </div>
                    
                    <div style="margin: 10px 0;">
                        <strong>📦 Quantity:</strong> {donation['quantity']}<br>
                        <strong>📍 Location:</strong> {donation['donor_address']} ({distance:.1f} km away)<br>
                        <strong>⏰ Time Remaining:</strong> {time_remaining}<br>
                        <strong>🍽️ Category:</strong> {donation['food_category']}<br>
                        <strong>📝 Description:</strong> {donation['description']}
                    </div>
                    
                    {f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'><strong>⚠️ Special Instructions:</strong> {donation['special_instructions']}</div>" if donation['special_instructions'] else ""}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    if st.button(f"🚗 Claim Pickup", key=f"claim_{donation['id']}"):
                        # Update donation status
                        for i, d in enumerate(st.session_state.food_donations):
                            if d['id'] == donation['id']:
                                st.session_state.food_donations[i]['status'] = 'Claimed'
                                st.session_state.food_donations[i]['claimed_by'] = 'VOL001'
                                break
                        
                        st.success(f"✅ Pickup claimed! Contact {donation['donor_name']} at {donation['donor_phone']}")
                        tm.sleep(1)
                        st.rerun()
                
                with col2:
                    if st.button(f"📱 Contact Donor", key=f"contact_{donation['id']}"):
                        st.info(f"📞 Call: {donation['donor_phone']}")
                
                with col3:
                    if st.button(f"🗺️ Directions", key=f"directions_{donation['id']}"):
                        st.info(f"📍 Navigate to: {donation['donor_address']}")
                
                st.markdown("---")

elif selected_page == "Food Requests":
    st.header("🏢 Request Food Donations")
    
    st.info("📢 **Hope Foundation** - You can request specific types of food donations based on your current needs.")
    
    with st.form("food_request_form"):
        st.subheader("Create Food Request")
        
        col1, col2 = st.columns(2)
        
        with col1:
            request_type = st.selectbox("Food Type Needed*", 
                                      ["Cooked Meals", "Raw Ingredients", "Bakery Items", 
                                       "Fruits & Vegetables", "Baby Food", "Any Food"])
            quantity_needed = st.text_input("Quantity Needed*", placeholder="e.g., 100 servings, 50 people")
            urgency = st.selectbox("Urgency Level*", ["High", "Medium", "Low"])
        
        with col2:
            dietary_preferences = st.multiselect("Dietary Preferences", 
                                               ["Vegetarian", "Vegan", "Halal", "No Restrictions"])
            beneficiary_count = st.number_input("Number of Beneficiaries", min_value=1, value=50)
            collection_time = st.selectbox("Preferred Collection Time", 
                                         ["Morning (6-10 AM)", "Afternoon (12-4 PM)", 
                                          "Evening (6-9 PM)", "Flexible"])
        
        special_requirements = st.text_area("Special Requirements/Notes", 
                                          placeholder="Any specific needs, allergies to avoid, etc.")
        
        submitted = st.form_submit_button("📝 Submit Request", type="primary")
        
        if submitted:
            st.success("✅ Food request submitted successfully! We'll notify nearby donors and volunteers.")
    
    # Show current requests
    st.subheader("📋 Current Food Requests")
    
    sample_requests = [
        {
            'type': 'Cooked Meals',
            'quantity': '80 servings',
            'urgency': 'High',
            'beneficiaries': 80,
            'status': 'Active',
            'created': '2 hours ago'
        },
        {
            'type': 'Fruits & Vegetables',
            'quantity': '20 kg',
            'urgency': 'Medium',
            'beneficiaries': 60,
            'status': 'Fulfilled',
            'created': '1 day ago'
        }
    ]
    
    for req in sample_requests:
        status_color = "🟢" if req['status'] == 'Active' else "🔵"
        st.markdown(f"""
        <div class="food-card">
            <h4>{status_color} {req['type']} - {req['quantity']}</h4>
            <strong>👥 Beneficiaries:</strong> {req['beneficiaries']} people<br>
            <strong>🚨 Urgency:</strong> {req['urgency']}<br>
            <strong>📅 Requested:</strong> {req['created']}<br>
            <strong>📊 Status:</strong> {req['status']}
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "All Donations" or selected_page == "Analytics":
    st.header("📊 Food Rescue Analytics & Impact")
    
    # Generate some sample analytics data
    today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Daily donations trend
    daily_donations = [random.randint(5, 25) for _ in dates]
    daily_rescues = [random.randint(3, 20) for _ in dates]
    daily_waste_prevented = [d * 2.5 for d in daily_rescues]  # kg
    
    # Create dataframes for visualization
    trend_data = pd.DataFrame({
        'Date': dates,
        'Donations_Posted': daily_donations,
        'Food_Rescued': daily_rescues,
        'Waste_Prevented_kg': daily_waste_prevented
    })
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_donations = sum(daily_donations)
    total_rescued = sum(daily_rescues)
    total_waste_prevented = sum(daily_waste_prevented)
    people_fed = total_rescued * 8  # Estimate 8 people per rescue
    
    with col1:
        st.metric("Total Donations (30 days)", total_donations, delta=f"+{daily_donations[-1]-daily_donations[-2]}")
    with col2:
        st.metric("Food Rescued (kg)", f"{total_waste_prevented:.0f}", delta=f"+{daily_waste_prevented[-1]-daily_waste_prevented[-2]:.1f}")
    with col3:
        st.metric("People Fed", people_fed, delta=f"+{(daily_rescues[-1]-daily_rescues[-2])*8}")
    with col4:
        avg_response_time = 2.3
        st.metric("Avg Response Time (hrs)", f"{avg_response_time:.1f}", delta="-0.2")
    
    # Donation trends chart
    st.subheader("📈 Food Rescue Trends (Last 30 Days)")
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['Donations_Posted'],
        mode='lines+markers',
        name='Donations Posted',
        line=dict(color='#ff7f0e', width=3)
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['Food_Rescued'],
        mode='lines+markers',
        name='Food Rescued',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig_trend.update_layout(
        title="Daily Food Donation & Rescue Activity",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Food type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍽️ Food Types Rescued")
        food_types = ['Cooked Meals', 'Bakery Items', 'Fruits & Vegetables', 'Raw Ingredients', 'Packaged Food']
        food_counts = [45, 28, 22, 18, 12]
        
        fig_pie = px.pie(
            values=food_counts,
            names=food_types,
            title="Food Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("⏰ Response Time Analysis")
        response_times = ['< 30 min', '30min - 1hr', '1-2 hrs', '2-4 hrs', '> 4 hrs']
        response_counts = [32, 45, 28, 15, 5]
        
        fig_bar = px.bar(
            x=response_times,
            y=response_counts,
            title="Pickup Response Times",
            color=response_counts,
            color_continuous_scale='RdYlGn_r'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Impact metrics
    st.subheader("🌍 Environmental & Social Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50, #45a049); padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3>🌱</h3>
            <h2>2,847 kg</h2>
            <p>CO₂ Emissions Prevented</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3, #1976D2); padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3>💧</h3>
            <h2>15,230 L</h2>
            <p>Water Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9800, #F57C00); padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3>💰</h3>
            <h2>₹1,24,500</h2>
            <p>Economic Value Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #9C27B0, #7B1FA2); padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3>👥</h3>
            <h2>847</h2>
            <p>Families Helped</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Heat map of food waste by area
    st.subheader("🗺️ Food Waste Distribution by Area")
    
    # Sample data for heatmap
    areas = ['Race Course', 'RS Puram', 'Gandhipuram', 'Peelamedu', 'Saibaba Colony', 
             'Singanallur', 'Vadavalli', 'Ukkadam', 'Town Hall', 'Coimbatore North']
    waste_data = [
        [25, 30, 35, 20, 15, 22, 18, 28, 32, 19],  # Monday
        [22, 28, 32, 18, 12, 25, 20, 30, 28, 16],  # Tuesday
        [28, 35, 38, 25, 18, 28, 22, 35, 35, 22],  # Wednesday
        [30, 32, 40, 22, 20, 30, 25, 32, 38, 25],  # Thursday
        [35, 40, 45, 28, 25, 35, 28, 40, 42, 30],  # Friday
        [40, 45, 50, 32, 30, 40, 35, 45, 48, 35],  # Saturday
        [32, 38, 42, 25, 22, 32, 28, 38, 40, 28]   # Sunday
    ]
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=waste_data,
        x=areas,
        y=days,
        colorscale='RdYlGn_r',
        text=waste_data,
        texttemplate="%{text}kg",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Daily Food Waste by Area (kg)",
        xaxis_title="Area",
        yaxis_title="Day of Week",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif selected_page == "My Donations":
    st.header("📦 My Food Donations")
    
    # Filter donations by current donor
    donor_donations = [d for d in st.session_state.food_donations 
                      if d['donor_name'] == st.session_state.current_user]
    
    if not donor_donations:
        st.info("🍽️ You haven't made any food donations yet. Click on 'Donate Food' to get started!")
    else:
        st.success(f"📊 You have made {len(donor_donations)} food donations")
        
        # Status summary
        status_counts = {}
        for donation in donor_donations:
            status = donation['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available", status_counts.get('Available', 0))
        with col2:
            st.metric("Claimed", status_counts.get('Claimed', 0))
        with col3:
            st.metric("Delivered", status_counts.get('Delivered', 0))
        
        # Show donations
        for donation in sorted(donor_donations, key=lambda x: x['created_at'], reverse=True):
            status_class = f"status-{donation['status'].lower()}"
            urgency = get_urgency_color(donation['expiry_time'])
            time_remaining = format_time_remaining(donation['expiry_time'])
            
            st.markdown(f"""
            <div class="food-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4>{urgency} {donation['food_type']} - {donation['quantity']}</h4>
                    <span class="{status_class}">{donation['status'].upper()}</span>
                </div>
                
                <div style="margin: 10px 0;">
                    <strong>🆔 Donation ID:</strong> {donation['id']}<br>
                    <strong>📅 Created:</strong> {donation['created_at'].strftime('%Y-%m-%d %H:%M')}<br>
                    <strong>⏰ Time Remaining:</strong> {time_remaining}<br>
                    <strong>📝 Description:</strong> {donation['description']}<br>
                    
                    {f"<strong>🚗 Claimed by:</strong> Volunteer {donation['claimed_by']}<br>" if donation['claimed_by'] else ""}
                    {f"<strong>🏢 Delivered to:</strong> {donation['delivered_to']}<br>" if donation['delivered_to'] else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if donation['status'] == 'Available':
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"❌ Cancel", key=f"cancel_{donation['id']}"):
                        # Remove donation
                        st.session_state.food_donations = [d for d in st.session_state.food_donations 
                                                         if d['id'] != donation['id']]
                        st.success("Donation cancelled successfully")
                        st.rerun()

elif selected_page == "My Deliveries":
    st.header("🚚 My Delivery History")
    
    # Sample delivery data for the volunteer
    sample_deliveries = [
        {
            'id': 'DEL001',
            'donation_id': 'FD002',
            'donor': 'City Bakery',
            'recipient': 'Hope Foundation',
            'food_type': 'Bakery Items',
            'quantity': '30 pieces',
            'pickup_time': datetime.now() - timedelta(hours=2),
            'delivery_time': datetime.now() - timedelta(hours=1),
            'status': 'Completed',
            'distance': 5.2,
            'rating': 5
        },
        {
            'id': 'DEL002',
            'donation_id': 'FD004',
            'donor': 'Hotel Paradise',
            'recipient': 'Street Shelter',
            'food_type': 'Cooked Meals',
            'quantity': '40 servings',
            'pickup_time': datetime.now() - timedelta(days=1, hours=3),
            'delivery_time': datetime.now() - timedelta(days=1, hours=2),
            'status': 'Completed',
            'distance': 8.1,
            'rating': 4
        }
    ]
    
    # Delivery stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deliveries", "25")
    with col2:
        st.metric("This Week", "7")
    with col3:
        st.metric("Average Rating", "4.8 ⭐")
    with col4:
        st.metric("Distance Covered", "142 km")
    
    # Show deliveries
    st.subheader("📋 Recent Deliveries")
    
    for delivery in sample_deliveries:
        st.markdown(f"""
        <div class="food-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>🍽️ {delivery['food_type']} - {delivery['quantity']}</h4>
                <span class="status-delivered">COMPLETED</span>
            </div>
            
            <div style="margin: 10px 0;">
                <strong>📤 From:</strong> {delivery['donor']}<br>
                <strong>📥 To:</strong> {delivery['recipient']}<br>
                <strong>🚗 Distance:</strong> {delivery['distance']} km<br>
                <strong>📅 Delivered:</strong> {delivery['delivery_time'].strftime('%Y-%m-%d %H:%M')}<br>
                <strong>⭐ Rating:</strong> {delivery['rating']}/5 stars
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Organizations":
    st.header("🏢 Registered Organizations")
    
    st.info("📊 Currently serving organizations in the Coimbatore area")
    
    for org in st.session_state.organizations:
        st.markdown(f"""
        <div class="food-card">
            <h4>🏢 {org['name']}</h4>
            <div style="margin: 10px 0;">
                <strong>👤 Contact Person:</strong> {org['contact_person']}<br>
                <strong>📞 Phone:</strong> {org['phone']}<br>
                <strong>📍 Address:</strong> {org['address']}<br>
                <strong>🏷️ Type:</strong> {org['type']}<br>
                <strong>👥 Beneficiaries:</strong> {org['beneficiaries']} people<br>
                <strong>📝 Requirements:</strong> {org['requirements']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add new organization form
    with st.expander("➕ Register New Organization"):
        with st.form("new_org_form"):
            col1, col2 = st.columns(2)
            with col1:
                org_name = st.text_input("Organization Name*")
                contact_person = st.text_input("Contact Person*")
                phone = st.text_input("Phone Number*")
            with col2:
                org_type = st.selectbox("Organization Type*", 
                                      ["NGO", "Shelter Home", "Old Age Home", "Orphanage", "Community Center"])
                beneficiaries = st.number_input("Number of Beneficiaries*", min_value=1)
            
            address = st.text_area("Address*")
            requirements = st.text_area("Food Requirements & Preferences")
            
            if st.form_submit_button("Register Organization"):
                new_org = {
                    'id': f'ORG{len(st.session_state.organizations) + 1:03d}',
                    'name': org_name,
                    'contact_person': contact_person,
                    'phone': phone,
                    'address': address,
                    'type': org_type,
                    'beneficiaries': beneficiaries,
                    'location_lat': 11.0168 + random.uniform(-0.05, 0.05),
                    'location_lng': 76.9558 + random.uniform(-0.05, 0.05),
                    'requirements': requirements
                }
                
                st.session_state.organizations.append(new_org)
                st.success("✅ Organization registered successfully!")
                st.rerun()

elif selected_page == "Volunteers":
    st.header("🚗 Volunteer Management")
    
    st.info(f"👥 Currently {len(st.session_state.volunteers)} registered volunteers")
    
    # Volunteer stats
    available_volunteers = len([v for v in st.session_state.volunteers if v['availability'] == 'Available'])
    busy_volunteers = len([v for v in st.session_state.volunteers if v['availability'] == 'Busy'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Volunteers", len(st.session_state.volunteers))
    with col2:
        st.metric("Available Now", available_volunteers, delta=f"+{available_volunteers-1}")
    with col3:
        avg_rating = sum([v['rating'] for v in st.session_state.volunteers]) / len(st.session_state.volunteers)
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    
    # Show volunteers
    for volunteer in st.session_state.volunteers:
        availability_color = "🟢" if volunteer['availability'] == 'Available' else "🔴"
        
        st.markdown(f"""
        <div class="food-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>{availability_color} {volunteer['name']}</h4>
                <div>
                    <span class="volunteer-badge">{volunteer['vehicle']}</span>
                    <span class="volunteer-badge">⭐ {volunteer['rating']}</span>
                </div>
            </div>
            
            <div style="margin: 10px 0;">
                <strong>📞 Phone:</strong> {volunteer['phone']}<br>
                <strong>📍 Area:</strong> {volunteer['area']}<br>
                <strong>🚗 Vehicle:</strong> {volunteer['vehicle']}<br>
                <strong>📊 Status:</strong> {volunteer['availability']}<br>
                <strong>✅ Completed:</strong> {volunteer['completed_deliveries']} deliveries<br>
                <strong>⭐ Rating:</strong> {volunteer['rating']}/5.0
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected_page == "Profile":
    st.header("👤 Volunteer Profile")
    
    volunteer_data = st.session_state.volunteers[0]  # Assuming first volunteer is current user
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
            <h2>👤</h2>
            <h3>{volunteer_data['name']}</h3>
            <p>⭐ {volunteer_data['rating']}/5.0 Rating</p>
            <p>✅ {volunteer_data['completed_deliveries']} Deliveries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📝 Update Profile")
        
        with st.form("profile_form"):
            name = st.text_input("Name", value=volunteer_data['name'])
            phone = st.text_input("Phone", value=volunteer_data['phone'])
            area = st.text_input("Service Area", value=volunteer_data['area'])
            vehicle = st.selectbox("Vehicle Type", 
                                 ["Two Wheeler", "Four Wheeler", "Bicycle"], 
                                 index=0 if volunteer_data['vehicle'] == 'Two Wheeler' else 1)
            availability = st.selectbox("Current Status", 
                                      ["Available", "Busy"], 
                                      index=0 if volunteer_data['availability'] == 'Available' else 1)
            
            if st.form_submit_button("Update Profile"):
                st.success("✅ Profile updated successfully!")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Deliveries This Month", "7", delta="+2")
    with col2:
        st.metric("Average Rating", f"{volunteer_data['rating']}", delta="+0.1")
    with col3:
        st.metric("Response Time (avg)", "18 min", delta="-3 min")
    with col4:
        st.metric("Distance Covered", "45 km", delta="+12 km")

elif selected_page == "Impact Report":
    st.header("📈 Impact Report - Hope Foundation")
    
    st.markdown("""
    ### 🎯 Mission Impact Summary
    This report shows the positive impact your organization has made through the FoodBridge platform.
    """)
    
    # Impact metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50, #45a049); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>👥</h3>
            <h2>1,247</h2>
            <p>People Fed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2196F3, #1976D2); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🍽️</h3>
            <h2>89</h2>
            <p>Donations Received</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FF9800, #F57C00); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>⚡</h3>
            <h2>234 kg</h2>
            <p>Food Rescued</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #9C27B0, #7B1FA2); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>💰</h3>
            <h2>₹18,500</h2>
            <p>Value Saved</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Monthly trend
    st.subheader("📊 Monthly Food Received Trend")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    food_received = [45, 52, 38, 65, 71, 89]
    people_fed = [180, 208, 152, 260, 284, 356]
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=months, y=food_received,
        mode='lines+markers',
        name='Food Donations Received',
        line=dict(color='#4CAF50', width=3)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=months, y=people_fed,
        mode='lines+markers',
        name='People Fed',
        yaxis='y2',
        line=dict(color='#2196F3', width=3)
    ))
    
    fig_trend.update_layout(
        title="Food Received vs People Fed",
        xaxis_title="Month",
        yaxis_title="Food Donations",
        yaxis2=dict(title="People Fed", overlaying='y', side='right'),
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Impact story
    st.subheader("📖 Impact Stories")
    
    st.markdown("""
    <div class="alert-success">
        <h4>🌟 Success Story - June 2024</h4>
        <p><strong>"Thanks to FoodBridge, we were able to serve fresh meals to 89 families during the monsoon floods. 
        The platform connected us with 12 different restaurants who donated surplus food, ensuring no one in our 
        shelter went hungry during those difficult days."</strong></p>
        <p><em>- Maria Joseph, Program Director, Hope Foundation</em></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
### 🤝 About FoodBridge

**FoodBridge** is a comprehensive food rescue platform that connects the dots between food waste and hunger. 
Our mission is to create a sustainable ecosystem where surplus food reaches those who need it most.

**Key Features:**
- 📱 Real-time food availability tracking
- 🗺️ GPS-based location services for efficient logistics
- 👥 Volunteer network management
- 📊 Impact analytics and reporting
- 🏢 Multi-stakeholder platform (Donors, Volunteers, NGOs)
- ⚡ Urgent pickup alerts and notifications

**Get Involved:**
- **Restaurants & Businesses**: List your surplus food for pickup
- **Volunteers**: Help transport food from donors to beneficiaries  
- **NGOs & Organizations**: Register to receive food donations
- **Individuals**: Donate excess food from events or daily cooking

Together, we can create a world where no food goes to waste and no one goes hungry! 🌍✨

---
*Built with ❤️ using Streamlit | For technical support, contact: support@foodbridge.org*
""")

# Emergency contact info
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Emergency Contacts")
st.sidebar.markdown("""
**Food Safety Emergency:**  
📞 1800-123-4567

**Volunteer Support:**  
📞 +91-9876543299

**24/7 Helpline:**  
📞 1800-FOODBRIDGE
""")

# Quick stats in sidebar
if st.session_state.current_user_type:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Today's Stats")
    st.sidebar.markdown(f"""
    🍽️ **Available Now:** {len([d for d in st.session_state.food_donations if d['status'] == 'Available'])} donations  
    🚗 **Active Volunteers:** {len([v for v in st.session_state.volunteers if v['availability'] == 'Available'])}  
    🏢 **Partner Organizations:** {len(st.session_state.organizations)}  
    ⚡ **Urgent Pickups:** {len([d for d in st.session_state.food_donations if d['status'] == 'Available' and (d['expiry_time'] - datetime.now()).total_seconds() < 3600])}
    """)

# Add some real-time updates simulation
if st.session_state.current_user_type == "Admin":
    st.sidebar.markdown("---") 
    st.sidebar.markdown("### 🔔 Live Updates")
    
    # Simulate real-time notifications
    notifications = [
        "🆕 New donation: City Bakery (5 min ago)",
        "✅ Pickup completed by Volunteer002 (12 min ago)", 
        "🚨 Urgent: Hotel Paradise needs pickup (18 min ago)",
        "👥 New volunteer registered (1 hour ago)"
    ]
    
    for notification in notifications[:3]:  # Show last 3
        st.sidebar.markdown(f"<small>{notification}</small>", unsafe_allow_html=True)
