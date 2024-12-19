import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model_pipeline = joblib.load("audience_rating_predictor.pkl")
except FileNotFoundError:
    st.error("Model file 'audience_rating_predictor.pkl' not found. Please ensure it's in the same directory as this app.")
    st.stop()

# App title
st.title("Audience Rating Prediction App")

# Subtitle
st.markdown("### Predict the audience rating of a movie based on its details")

# Input fields
st.header("Enter Movie Details:")
genre = st.selectbox("Select Genre", [
    "Action", "Drama", "Comedy", "Thriller", "Horror", "Romance", "Sci-Fi", "Fantasy", "Documentary"
])

runtime_in_minutes = st.number_input(
    "Enter Runtime (in minutes):", min_value=30, max_value=300, value=120, step=1
)

tomatometer_rating = st.slider(
    "Tomatometer Rating (%):", min_value=0, max_value=100, value=50
)

tomatometer_count = st.number_input(
    "Tomatometer Count (Number of Reviews):", min_value=1, max_value=10000, value=50
)

studio_name = st.text_input("Enter Studio Name (e.g., Universal Pictures):", value="Universal Pictures")

# Prediction button
if st.button("Predict Audience Rating"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            "genre": [genre],
            "runtime_in_minutes": [runtime_in_minutes],
            "tomatometer_rating": [tomatometer_rating],
            "tomatometer_count": [tomatometer_count],
            "studio_name": [studio_name]
        })

        # Debug: Display input data
        st.write("Input Data for Prediction:")
        st.write(input_data)

        # Make prediction
        prediction = model_pipeline.predict(input_data)

        # Debug: Display raw prediction result
        st.write("Raw Prediction Output:", prediction)

        # Display the result
        st.success(f"The predicted audience rating is: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.markdown(
    "Developed with Streamlit | Model built with Random Forest Regressor"
)
