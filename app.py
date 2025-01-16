import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page setup
st.set_page_config(page_title="Food Product Analysis", layout="wide", initial_sidebar_state="expanded")

# Load the dataset
data_file = 'food_data.csv'
data = pd.read_csv(data_file)

# Normalize column names: Remove spaces and convert to lowercase
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Handle missing 'Is Harmful?' column
if 'is_harmful?' not in data.columns:
    st.warning("'Is Harmful?' column is missing from the dataset. Defaulting to 'No' for all entries.")
    data['is_harmful?'] = 'No'

# Custom styles for light and dark mode
st.markdown("""
    <style>
    /* Light mode styles */
    body {
        background-color: #FFFFFF;
        color: black;
    }
    h1, h2, h3, h4 {
        color: #4CAF50;
    }
    .stTextInput input {
        border: 1px solid #4CAF50;
        color: #000000;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stSidebar .sidebar-content {
        background-color: #F8F9FA;
        color: black;
    }
    .stSidebar .sidebar-header h1 {
        color: #4CAF50;
    }

    /* Dark mode styles */
    body[data-testid="stAppViewContainer"] {
        background-color: #181818;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #A8D08D;
    }
    .stTextInput input {
        border: 1px solid #A8D08D;
        color: white;
    }
    .stButton button {
        background-color: #A8D08D;
        color: black;
    }
    .stSidebar .sidebar-content {
        background-color: #333333;
        color: white;
    }
    .stSidebar .sidebar-header h1 {
        color: #A8D08D;
    }
    .stSidebar .sidebar-header {
        color: #A8D08D;
    }
    
    .footer {
        background-color: #333333;
        padding: 10px;
        text-align: center;
        color: #fff;
        font-size: 14px;
        border-radius: 10px;
        margin-top: 30px;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Sidebar for user input
col1, col2 = st.columns([1, 5])  # Create two columns with ratio 1:5
with col1:
    st.image('logo.ico', width=300) 
with col2:
    st.title("Shows what's in your Food")

st.sidebar.header("🔍 Search and Filter")
query = st.sidebar.text_input("Search for a product by name:", placeholder="Type product name...")
category_filter = st.sidebar.selectbox("Select Category", options=["All"] + data["category"].dropna().unique().tolist())
brand_filter = st.sidebar.multiselect("Select Brand", options=data["brand"].dropna().unique().tolist())
harmful_filter = st.sidebar.radio("Include Harmful Products?", ["All", "Yes", "No"])

# Clean and preprocess data
data.fillna("N/A", inplace=True)
data['harmful_ingredient_count'] = pd.to_numeric(data.get('harmful_ingredient_count', 0), errors='coerce').fillna(0).astype(int)
data['total_ingredients'] = pd.to_numeric(data.get('total_ingredients', 0), errors='coerce').fillna(0).astype(int)

# Filter data based on user input
filtered_data = data.copy()
if category_filter != "All":
    filtered_data = filtered_data[filtered_data["category"] == category_filter]
if brand_filter:
    filtered_data = filtered_data[filtered_data["brand"].isin(brand_filter)]
if harmful_filter != "All":
    filtered_data = filtered_data[filtered_data["is_harmful?"].str.lower() == harmful_filter.lower()]

# Display filtered dataset summary and the table of contents
st.subheader("Filtered Products")
st.dataframe(
    filtered_data[['product_name', 'brand', 'category', 'is_harmful?']].dropna(how='any'),
    width=1000,
    height=500,
)

# NLP Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data['product_name'].dropna())

# Search Functionality
def search_product(query, data):
    query_vector = vectorizer.transform([query])
    data_vectors = vectorizer.transform(data['product_name'].dropna())
    similarities = cosine_similarity(query_vector, data_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    return data.iloc[best_match_idx], similarities[best_match_idx]

# Main content - Search Functionality below the table
st.subheader("Search for a Product")
query = st.text_input("Product Name:", placeholder="Type product name...")
category_filter = st.selectbox("Category", options=["All"] + data["category"].dropna().unique().tolist())
brand_filter = st.multiselect("Brand", options=data["brand"].dropna().unique().tolist())

# Display search results if query
if query:
    try:
        result, similarity = search_product(query, filtered_data)
        if similarity > 0.5:
            st.subheader(f"Details for {result['product_name']}")
            st.write(result)

            # Pie chart: Harmful vs Total Ingredients
            harmful = result['harmful_ingredient_count']
            non_harmful = result['total_ingredients'] - harmful

            if harmful + non_harmful > 0:  # Check to avoid division by zero
                fig, ax = plt.subplots()
                ax.pie(
                    [harmful, non_harmful],
                    labels=['Harmful', 'Non-Harmful'],
                    autopct='%1.1f%%',
                    colors=['red', 'green'],
                    startangle=90,
                )
                ax.set_title("Ingredient Composition")
                st.pyplot(fig)
            else:
                st.warning("No ingredient data available to create a pie chart.")

            # Nutritional Impact
            st.subheader("Nutritional Impact and Alternatives")
            st.write(f"Nutritional Impact: {result.get('nutritional_impact', 'N/A')}")
            st.write(f"Healthy Alternative: {result.get('healthy_alternatives', 'N/A')}")
            st.write(f"Alternative Description: {result.get('alternative_description', 'N/A')}")
        else:
            st.warning("No close matches found. Try a different query or adjust filters.")
            # Show suggestions
            st.subheader("Suggested Products")
            suggestions = filtered_data[filtered_data['product_name'].str.contains(query, case=False, na=False)]
            if not suggestions.empty:
                st.dataframe(suggestions[['product_name', 'brand', 'category']])
            else:
                st.write("No suggestions available.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("""
    <div class="footer">
        <p>📧 Email: <a href='mailto:infactsap2025@gmail.com'>infactsap2025@gmail.com</a></p>
        <p>🌐 Social Media: <a href='#'>Facebook</a> | <a href='#'>Twitter</a> | <a href='#'>Instagram</a></p>
        <p style='color:grey; margin-top: 1rem;'>This is a food analysis web app for informational purposes only. It is not a substitute for professional medical or dietary advice.</p>
        <p style='font-size:12px; color:#808080; margin-top: 0.5rem;'>&copy;2025 Team In-Fact PDEA COEM Pune</p>
    </div>
""", unsafe_allow_html=True)
