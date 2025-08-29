import joblib
import streamlit as st
import os

# Path relatif
model = joblib.load(os.path.join("model", "model_sentiment.joblib"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.joblib"))


# Streamlit app
st.title("Analisis Sentimen App Shope - Naive Bayes")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):
    if user_input.strip():
        # Ubah teks jadi fitur
        X = vectorizer.transform([user_input])
        # Prediksi
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.success("Sentimen POSITIF 😀")
        else:
            st.error("Sentimen NEGATIF 😡")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")
