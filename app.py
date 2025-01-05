import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import joblib

# Set page config
st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

@st.cache_resource
def load_models_and_data():
    try:
        # Load the CSV files directly instead of pickled files
        products = pd.read_csv('PRODUCT_NEW (1).csv')
        ratings = pd.read_csv('PRODUCT_NEW_RATINGS (1).csv')
        
        # Create user-item matrix
        user_item_matrix = pd.pivot_table(
            ratings,
            values='rating',
            index='user-id',
            columns='product-id',
            fill_value=0
        )
        
        # Train SVD model
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(user_item_matrix)
        
        svd = TruncatedSVD(n_components=50)
        svd.fit(scaled_data)
        
        return svd, scaler, user_item_matrix, products, ratings
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def get_product_details(product_id, products_df):
    product = products_df[products_df['product-id'] == product_id]
    if not product.empty:
        return {
            'Product Name': product.iloc[0]['product_name'],
            'Category': product.iloc[0]['category'],
            'Price': product.iloc[0]['price']
        }
    return None

def get_recommendations(user_id, svd, scaler, user_item_matrix, products_df, n_recommendations=5):
    try:
        # Get user's row index
        if user_id not in user_item_matrix.index:
            return None
            
        user_idx = user_item_matrix.index.get_loc(user_id)
        
        # Get user's scaled ratings
        user_ratings = scaler.transform(user_item_matrix)[user_idx:user_idx+1]
        
        # Transform user ratings to latent space
        user_latent = svd.transform(user_ratings)
        
        # Transform back to get predicted ratings for all items
        predicted_ratings = svd.inverse_transform(user_latent)
        predicted_ratings = scaler.inverse_transform(predicted_ratings)[0]
        
        # Create dictionary of product_id -> predicted_rating
        predictions = dict(zip(user_item_matrix.columns, predicted_ratings))
        
        # Sort by predicted rating
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = []
        for product_id, pred_rating in sorted_predictions[:n_recommendations]:
            product_details = get_product_details(product_id, products_df)
            if product_details:
                recommendations.append({
                    'Product ID': product_id,
                    **product_details,
                    'Predicted Rating': pred_rating
                })
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

def main():
    st.title("🛍️ Product Recommendation System")
    st.write("Select a user to get personalized product recommendations!")
    
    # Load models and data
    svd, scaler, user_item_matrix, products, ratings = load_models_and_data()
    
    if any(x is None for x in [svd, scaler, user_item_matrix, products, ratings]):
        st.error("Failed to load required data.")
        return
    
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
        user_ids = sorted(ratings['user-id'].unique())
        user_id = st.selectbox("Select User ID:", user_ids)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(
                    user_id,
                    svd,
                    scaler,
                    user_item_matrix,
                    products,
                    num_recommendations
                )
                
                if recommendations:
                    with col2:
                        st.subheader("Recommended Products")
                        
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
                    st.error("Could not generate recommendations for this user.")

if __name__ == "__main__":
    main()
