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
        svd = joblib.load('svd_model.joblib')
        scaler = joblib.load('scaler_model.joblib')
        user_item_matrix = pd.read_pickle('user_item_matrix.pkl')
        products = pd.read_pickle('products.pkl')
        return svd, scaler, user_item_matrix, products
    except Exception as e:
        st.error(f"Error loading models and data: {str(e)}")
        return None, None, None, None

def get_recommendations(user_id, svd, scaler, user_item_matrix, products, n_recommendations=5):
    try:
        # Get user's row index
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
            product = products[products['product-id'] == product_id].iloc[0]
            recommendations.append({
                'Product ID': product_id,
                'Product Name': product.get('product_name', 'N/A'),
                'Category': product.get('category', 'N/A'),
                'Price': product.get('price', 'N/A'),
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
    svd, scaler, user_item_matrix, products = load_models_and_data()
    
    if svd is None or scaler is None or user_item_matrix is None or products is None:
        st.error("Failed to load required models and data.")
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
        user_ids = sorted(user_item_matrix.index.tolist())
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
