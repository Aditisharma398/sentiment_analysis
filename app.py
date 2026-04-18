import streamlit as st
import pickle

st.title("Sentiment Analyzer 💬")

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except:
    st.error("Model files not found. Run model.py first.")
    st.stop()

text = st.text_area("Enter your text:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        data = vectorizer.transform([text])
        result = model.predict(data)

        if result[0] == "positive":
            st.success("Positive 😊")
        elif result[0] == "negative":
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")

            st.write("Prediction:", result)