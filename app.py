import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# Page configuration
st.set_page_config(page_title="Food Product Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
     /* Set the background color for the entire app */
     .stApp {
        background: linear-gradient(to right, #FFD9B3, #FFE5CC, #FFFFFF, #D4F5C1);
     }
     
     /* Main container */
     .main {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
     }
    
     .sidebar-text {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #4CAF50 !important;
        margin-bottom: 20px !important;
     }
    
     .sidebar .stRadio > label {
        font-size: 16px !important;
        color: #2C3E50 !important;
        padding: 8px 0 !important;
     }
    
     .sidebar .stSelectbox > label {
        font-size: 16px !important;
        color: #2C3E50 !important;
        font-weight: 500 !important;
     }
    
     .sidebar .stMultiSelect > label {
        font-size: 16px !important;
        color: #2C3E50 !important;
        font-weight: 500 !important;
     }
    
     .sidebar [data-testid="stMarkdownContainer"] > div {
        padding: 10px 0 !important;
     }
    
     .sidebar [data-testid="stVerticalBlock"] > div {
        padding: 10px 0 !important;
     }
    
     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
     .title-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        border-radius: 20px;
        color: brown;
        box-shadow: 
            0 8px 32px rgba(31, 38, 135, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.18),
            inset 0 0 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
     }
    
     .title-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        
        background: linear-gradient(
            45deg,
            rgba(255, 105, 180, 0.1),
            rgba(255, 223, 0, 0.1),
            rgba(0, 255, 255, 0.1)
        );
        z-index: 0;
        animation: shimmer 8s linear infinite;
     }
    
     .animated-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-align: center;
        font-size: 3.8rem;
        position: relative;
        background: linear-gradient(
            45deg,
            #FF1493,  /* Deep Pink */
            #FF4500,  /* Orange Red */
            #FFD700,  /* Gold */
            #00FA9A,  /* Medium Spring Green */
            #00BFFF   /* Deep Sky Blue */
        );
        background-size: 300% 300%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent; /* Keeps the gradient visible */
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        animation: gradient 8s ease infinite, bounce 2s ease-in-out infinite;
        text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.2); /* Adds more emphasis */
     }

     .subtitle {
        font-family: 'Poppins', sans-serif;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: -0.5rem;
        padding-bottom: 0.5rem;
        animation: fadeIn 1s ease-in;
        position: relative;
     }
    
     .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        animation: fadeInLeft 1s ease-out;
        position: relative;
     }
    
     .logo-container img {
        transition: transform 0.3s ease, filter 0.3s ease;
        filter: drop-shadow(0 0 10px rgba(0,0,0,0.2));
     }
    
     .logo-container img:hover {
        transform: scale(1.08) rotate(2deg);
        filter: drop-shadow(0 0 15px rgba(0,0,0,0.3));
     }
    
     @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
     }
    
     @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
     }
    
     @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
     }
    
     @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
     }
    
     @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
     }
    
     /* Responsive design */
     @media (max-width: 768px) {
        .animated-title {
            font-size: 2.5rem;
            padding: 1rem;
        }
        .subtitle {
            font-size: 1.1rem;
        }
     }
    
     /* Food images gallery */
     .image-gallery {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0;
     }
    
     .image-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        width: 200px;
     }
    
     /* Footer */
     .footer {
        background: linear-gradient(135deg, #D4F5C1, #FFE5CC); /* Updated gradient */
        color: #2C3E50; /* Dark text color */
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
     }
    
     .footer-content {
        position: relative;
        z-index: 1;
     }
    
     .footer h2 {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #2C3E50; /* Dark text color */
     }
    
     .footer-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2.5rem;
        margin: 2rem 0;
     }
    
     .footer-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        transition: transform 0.3s ease;
     }
    
     .footer-section:hover {
        transform: translateY(-5px);
     }
    
     .footer-section h3 {
        color: #2C3E50; /* Dark text color */
        margin-bottom: 1.2rem;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3);
        padding-bottom: 0.5rem;
     }
    
     .footer-section ul {
        list-style: none;
        padding: 0;
     }
    
     .footer-section li {
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
     }
    
     .footer-section li::before {
        content: '✓';
        color: #4CAF50;
        font-weight: bold;
     }
    
     .social-links {
        text-align: center;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
     }
    
     .social-links a {
        display: inline-block;
        color: #2C3E50; /* Dark text color */
        margin: 0 1rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-decoration: none;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
     }
    
     .social-links a:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
     }
    
     .disclaimer {
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
     }
    
     .copyright {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.6);
     }
    
     @media (max-width: 768px) {
        .footer {
            padding: 2rem 1rem;
        }
        .footer-grid {
            gap: 1.5rem;
        }
        .footer h2 {
            font-size: 2rem;
        }
     }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('yes no data.csv')
        data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
        data.fillna("N/A", inplace=True)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found!")
        return None

data = load_data()
if data is None:
    st.stop()

# Title section with enhanced layout
col1, col2 = st.columns([1, 5])

with col1:
    if os.path.exists('logo.ico'):
        st.image('logo.ico', width=200)

with col2:
    st.markdown("""
        <div class="title-container">
            <h1 class="animated-title">What's in your Packaged Food?</h1>
            <p class="subtitle">Discover the truth about your food ingredients</p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown('<p class="sidebar-text">🔍 Search & Filter</p>', unsafe_allow_html=True)
    search_query = st.text_input("Search Products", value="", placeholder="Type product name...")
    st.markdown('<hr style="margin: 20px 0;">', unsafe_allow_html=True)
    
    st.markdown('<p class="sidebar-text">📁 Category</p>', unsafe_allow_html=True)
    category_filter = st.selectbox(
        "",
        options=["All"] + sorted(data["category"].dropna().unique().tolist())
    )
    
    st.markdown('<p class="sidebar-text">🏢 Brand</p>', unsafe_allow_html=True)
    brand_filter = st.multiselect(
        "",
        options=sorted(data["brand"].dropna().unique().tolist())
    )
    
    st.markdown('<p class="sidebar-text">⚠ Product Safety</p>', unsafe_allow_html=True)
    harmful_filter = st.radio(
        "",
        ["All", "Safe", "Potentially Harmful"]
    )

def display_random_images(num_images=3):
    """
    Display random product images from the available image files in the directory.
    
    Args:
        num_images (int): Number of images to display
    """
    # List all image files
    image_files = [f for f in os.listdir() if f.lower().startswith('img') and 
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        st.warning("No product images found in directory.")
        return
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Display images in columns
    cols = st.columns(num_images)
    
    for idx, col in enumerate(cols):
        with col:
            if idx < len(selected_images):
                try:
                    st.image(selected_images[idx],  use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {selected_images[idx]}")
            else:
                st.image("https://placehold.co/200x150/png?text=Food",  use_container_width=True)

display_random_images()

# Filter data
filtered_data = data.copy()
if category_filter != "All":
    filtered_data = filtered_data[filtered_data["category"] == category_filter]
if brand_filter:
    filtered_data = filtered_data[filtered_data["brand"].isin(brand_filter)]
if harmful_filter != "All":
    filtered_data = filtered_data[filtered_data["is_harmful?"].str.lower() == harmful_filter.lower()]

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<p style='color: black;'>Total Products</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: black;'>{len(filtered_data)}</h3>", unsafe_allow_html=True)
with col2:
    st.markdown("<p style='color: black;'>Categories</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: black;'>{len(filtered_data['category'].unique())}</h3>", unsafe_allow_html=True)
with col3:
    st.markdown("<p style='color: black;'>Brands</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: black;'>{len(filtered_data['brand'].unique())}</h3>", unsafe_allow_html=True)

# Product list
st.markdown("<h3 style='color: black;'>📊 Product List</h3>", unsafe_allow_html=True)
st.dataframe(
    filtered_data[['product_name', 'brand', 'category', 'is_harmful?']],
    use_container_width=True
)

# Search functionality
vectorizer = TfidfVectorizer()
vectorizer.fit(data['product_name'].dropna())

def search_product(query, data):
    query_vector = vectorizer.transform([query])
    data_vectors = vectorizer.transform(data['product_name'].dropna())
    similarities = cosine_similarity(query_vector, data_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    return data.iloc[best_match_idx], similarities[best_match_idx]

# Search results
if search_query:
    try:
        result, similarity = search_product(search_query, filtered_data)
        if similarity > 0.5:
            st.markdown(f"<h2 style='color: black;'>Details for {result['product_name']}</h2>", unsafe_allow_html=True)
            st.write(result)

            # Ingredient composition chart
            harmful = pd.to_numeric(result.get('harmful_ingredient_count', 0), errors='coerce')
            total = pd.to_numeric(result.get('total_ingredients', 0), errors='coerce')
            
            if pd.notnull(total) and pd.notnull(harmful) and total > 0:
                harmful = int(harmful)
                total = int(total)
                non_harmful = total - harmful
                fig, ax = plt.subplots()
                ax.pie(
                    [harmful, non_harmful],
                    labels=['Harmful', 'Non-Harmful'],
                    autopct='%1.1f%%',
                    colors=['#E74C3C', '#2ECC71'],
                    startangle=90
                )
                ax.set_title("Ingredient Composition")
                st.pyplot(fig)
            else:
                st.warning("Ingredient composition data is not available for this product.")

            # Nutritional information
            st.markdown("<h2 style='color: black;'>Nutritional Impact and Alternatives</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Nutritional Impact: {result.get('nutritional_impact', 'N/A')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Healthy Alternative: {result.get('healthy_alternatives', 'N/A')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Alternative Description: {result.get('alternative_description', 'N/A')}</p>", unsafe_allow_html=True)

        else:
            st.warning("No close matches found. Try a different search term.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Enhanced Footer
st.markdown("""
    <div class="footer" style="background: linear-gradient(135deg, #A3B7E0, #F2A3A3, #FCE6A4);">
        <h2 style="color: black;">About In-Fact</h2>
        <div class="footer-grid">
            <div class="footer-section">
                <h3 style="color: black;">Our Mission</h3>
                <p style="color: black;">In-Fact empowers consumers with detailed insights about their food products, 
                helping them make informed decisions about their nutrition and health.</p>
            </div>
            <div class="footer-section">
                <h3 style="color: black;">Key Features</h3>
                <ul>
                    <li style="color:black;">✓ Real-time ingredient analysis</li>
                    <li style="color: black;">✓ Health impact assessment</li>
                    <li style="color: black;">✓ Alternative product suggestions</li>
                    <li style="color: black;">✓ Brand comparison tools</li>
                </ul>
            </div>
            <div class="footer-section">
                <h3 style="color: black;">Contact Us</h3>
                <p style="color: black;">📧 Email: infactsap2025@gmail.com</p>
                <p style="color: black;">📱 Phone: +91 XXXXXXXXXX</p>
                <p style="color: black;">📍 PDEA COEM, Pune</p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 2rem;">
            <p style="color: black;">Follow us on social media:</p>
            <div style="margin: 1rem 0;">
                <a href="#" style="color: black; margin: 0 1rem;">Facebook</a>
                <a href="#" style="color: black; margin: 0 1rem;">Twitter</a>
                <a href="#" style="color: black; margin: 0 1rem;">Instagram</a>
            </div>
            <p style="color: black; margin-top: 1rem;">
                &copy; 2025 Team In-Fact Pune. All rights reserved.
            </p>
        </div>
        <div style="text-align: center; margin-top: 2rem;">
            <p style="color: black;">Disclaimer: The information provided in this application is for educational purposes only and should not be considered as medical advice. Always consult with a healthcare professional for dietary recommendations.</p>
        </div>
    </div>
""", unsafe_allow_html=True)
