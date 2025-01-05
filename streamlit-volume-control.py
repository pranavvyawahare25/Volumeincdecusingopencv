import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('product_recommendation_model.h5')

# Load datasets (replace with your actual file paths)
products = pd.read_csv('/content/PRODUCT_NEW (1).csv')
ratings = pd.read_csv('/content/PRODUCT_NEW_RATINGS (1).csv')

# Preprocess the data
user_ids = ratings['user-id'].unique()
product_ids = ratings['product-id'].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
product_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}
index_to_product = {idx: product_id for product_id, idx in product_to_index.items()}

# Define a function to recommend products
def recommend_products(user_id, num_recommendations=5):
    if user_id not in user_to_index:
        return f"User ID {user_id} not found!"
    
    user_idx = user_to_index[user_id]
    product_indices = np.array(list(product_to_index.values()))
    
    # Predict ratings for all products for the given user
    user_indices = np.full_like(product_indices, user_idx)
    predictions = model.predict([user_indices, product_indices])
    
    # Get top N recommendations
    top_indices = predictions.flatten().argsort()[-num_recommendations:][::-1]
    recommended_products = [
        {
            "Product ID": index_to_product[idx],
            "Predicted Rating": predictions[idx][0]
        }
        for idx in top_indices
    ]
    return recommended_products

# Streamlit UI
st.title("Product Recommendation System")
st.write("This app demonstrates a simple recommendation system using a trained model.")

# User input for selecting a User ID
user_id = st.selectbox("Select a User ID:", user_ids)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_products(user_id)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.write("Top Recommendations:")
        st.table(pd.DataFrame(recommendations))
