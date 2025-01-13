import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data_file = 'food_data.csv'
data = pd.read_csv(data_file)

# Streamlit page setup
st.set_page_config(page_title="Food Product Analysis", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    h1, h2, h3, h4 {
        color: #4CAF50;
    }
    .stTextInput > div > input {
        border: 1px solid #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stSidebar {
        background-color: #F8F9FA;
        padding: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #999999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Sidebar for user input
st.title("🍽️ Food Product Search and Analysis")
st.sidebar.header("🔍 Search and Filter")
query = st.sidebar.text_input("Search for a product by name:", placeholder="Type product name...")
category_filter = st.sidebar.selectbox("Select Category", options=["All"] + data["Category"].unique().tolist())
brand_filter = st.sidebar.multiselect("Select Brand", options=data["Brand"].unique().tolist())
harmful_filter = st.sidebar.radio("Include Harmful Products?", ["All", "Yes", "No"])

# Standardize column names
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Clean and preprocess data
data.fillna("N/A", inplace=True)
data['harmful_ingredient_count'] = data['harmful_ingredient_count'].astype(int)
data['total_ingredients'] = data['total_ingredients'].astype(int)

# Filter data based on user input
filtered_data = data
if category_filter != "All":
    filtered_data = filtered_data[filtered_data["category"] == category_filter]
if brand_filter:
    filtered_data = filtered_data[filtered_data["brand"].isin(brand_filter)]
if harmful_filter != "All":
    filtered_data = filtered_data[filtered_data["is_harmful"].str.lower() == harmful_filter.lower()]

# Display filtered dataset summary
st.sidebar.write(f"Found {len(filtered_data)} products matching your filters.")

# Table of Contents (below the sidebar)
st.subheader("Filtered Products")
st.dataframe(
    filtered_data[['name_of_product', 'brand', 'category', 'is_harmful']],
    width=1000,
    height=500,
)

# NLP Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data['name_of_product'])

# Search Functionality
def search_product(query, data):
    query_vector = vectorizer.transform([query])
    data_vectors = vectorizer.transform(data['name_of_product'])
    similarities = cosine_similarity(query_vector, data_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    return data.iloc[best_match_idx], similarities[best_match_idx]

# Display search results
if query:
    result, similarity = search_product(query, filtered_data)
    if similarity > 0.5:
        st.subheader(f"Details for {result['name_of_product']}")
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
        st.write(f"Nutritional Impact: {result['nutritional_impact']}")
        st.write(f"Healthy Alternative: {result['healthy_alternatives']}")
        st.write(f"Alternative Description: {result['alternative_description']}")
    else:
        st.warning("No close matches found. Try a different query or adjust filters.")
        # Show suggestions
        st.subheader("Suggested Products")
        suggestions = filtered_data[filtered_data['name_of_product'].str.contains(query, case=False, na=False)]
        if not suggestions.empty:
            st.dataframe(suggestions[['name_of_product', 'brand', 'category']])
        else:
            st.write("No suggestions available.")

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Made by InFact Team, PDEA COEM, Pune</div>", unsafe_allow_html=True)
