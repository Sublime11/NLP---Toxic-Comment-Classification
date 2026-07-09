import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("Sentiment_Analysis_App/best_model.pkl")
tfidf = joblib.load("Sentiment_Analysis_App/tfidf.pkl")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊"
)

st.title("😊 Sentiment Analysis App")

st.write("Enter a review below to predict its sentiment.")

review = st.text_area("Enter your review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        review_vector = tfidf.transform([review])

        prediction = model.predict(review_vector)

        # Assuming 'prediction' is a 2D array and we want the first element of the first row
        # Also, assuming your model predicts string labels like 'Positive', 'Negative', 'Neutral'
        # Adjust the prediction logic based on your model's actual output format
        # For multi-label, you might need to interpret the array of boolean predictions
        # For a simple sentiment, it might be a single class label.
        # Given previous context, it seems to be single label from VADER's sentiment_label.
        # Let's use the first predicted label if it's a list/array of labels.
        # If it's a multi-label classification, this logic might need further refinement
        # to display all predicted labels or the most dominant one.

        # This current Streamlit app structure implies a single class prediction.
        # Let's assume prediction[0] contains the primary sentiment label (e.g., 'Positive', 'Negative').
        # If the model is from 'tox_cols', then prediction would be an array of 0s and 1s.
        # The original `sentiment_label` function for VADER produces 'Positive', 'Negative', 'Neutral'.
        # If `best_model.pkl` is trained on `tox_cols`, it will predict arrays like [0, 0, 1, 0, 0, 0].
        # Re-evaluating the current app structure based on the previous context.
        # The original sentiment analysis used VADER to create 'Positive', 'Negative', 'Neutral'.
        # The best model was trained on `tox_cols` (toxic, severe_toxic, etc.).
        # So, the `prediction` from `model.predict` will be an array of booleans/integers.
        # The current `if/elif/else` structure (Positive, Negative, else) is for VADER-like single sentiment.
        # Let's modify the app logic to reflect the multi-label toxicity prediction.

        # Identify predicted toxic categories
        tox_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        predicted_labels = [col for i, col in enumerate(tox_cols) if prediction[0][i] == 1]

        if predicted_labels:
            st.error(f"Detected toxicity: {', '.join(predicted_labels)}")
        else:
            # If no toxicity is detected, the app could show a neutral or positive message
            # This assumes the model is purely for toxicity detection.
            # If it's also meant to do general sentiment (Positive/Negative/Neutral),
            # we would need to integrate the VADER logic or a separate sentiment model.
            # For now, if no toxicity, we'll state it's clean.
            st.success("No toxic content detected.")

