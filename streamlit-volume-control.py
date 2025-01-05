import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

# Function to load or create mappings
def create_mappings(ratings_df):
    user_ids = ratings_df['user-id'].unique()
    product_ids = ratings_df['product-id'].unique()
    
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    
    product_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}
    index_to_product = {idx: product_id for product_id, idx in product_to_index.items()}
    
    return user_to_index, index_to_user, product_to_index, index_to_product

# Function to get product details
def get_product_details(product_id, products_df):
    product = products_df[products_df['product-id'] == product_id].iloc[0]
    return {
        'Product Name': product.get('product_name', 'N/A'),
        'Category': product.get('category', 'N/A'),
        'Price': product.get('price', 'N/A')
    }

# Function to recommend products
def recommend_products(user_id, model, user_to_index, product_to_index, index_to_product, products_df, num_recommendations=5):
    if user_id not in user_to_index:
        return None
    
    user_idx = user_to_index[user_id]
    product_indices = np.array(list(product_to_index.values()))
    
    # Predict ratings for all products
    user_indices = np.full_like(product_indices, user_idx)
    predictions = model.predict([user_indices, product_indices], verbose=0)
    
    # Get top N recommendations
    top_indices = predictions.flatten().argsort()[-num_recommendations:][::-1]
    
    recommended_products = []
    for idx in top_indices:
        product_id = index_to_product[idx]
        product_details = get_product_details(product_id, products_df)
        recommended_products.append({
            "Product ID": product_id,
            "Predicted Rating": float(predictions.flatten()[idx]),
            **product_details
        })
    
    return recommended_products

# Main app
def main():
    st.title("🛍️ Product Recommendation System")
    st.write("Select a user to get personalized product recommendations!")
    
    # Load data
    try:
        products = pd.read_csv('PRODUCT_NEW (1).csv')
        ratings = pd.read_csv('PRODUCT_NEW_RATINGS (1).csv')
    except FileNotFoundError:
        st.error("Required dataset files not found. Please check the file paths.")
        st.stop()
        
    # Create mappings
    user_to_index, index_to_user, product_to_index, index_to_product = create_mappings(ratings)
    
    # Load model
    try:
        model = load_model('product_recommendation_model.h5')
    except:
        st.error("Model file not found or corrupted. Please check if 'product_recommendation_model.h5' exists.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Settings")
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # User selection
        user_ids = sorted(list(user_to_index.keys()))
        user_id = st.selectbox("Select User ID:", user_ids)
        
        if st.button("Get Recommendations", type="primary"):
            recommendations = recommend_products(
                user_id,
                model,
                user_to_index,
                product_to_index,
                index_to_product,
                products,
                num_recommendations
            )
            
            if recommendations:
                with col2:
                    st.subheader("Recommended Products")
                    
                    # Display recommendations in a nice format
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            #### {i}. {rec['Product Name']}
                            - **Category:** {rec['Category']}
                            - **Price:** ${rec['Price']}
                            - **Predicted Rating:** {rec['Predicted Rating']:.2f} / 5.0
                            ---
                            """)
            else:
                st.error("No recommendations found for this user.")

if __name__ == "__main__":
    main()
