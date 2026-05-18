import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("C:\intership project\Fake.csv")
true = pd.read_csv("C:\intership project\True.csv")

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine data
data = pd.concat([fake, true])
data = data.sample(frac=1)

# Prepare data
X = data['text']
y = data['label']

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

# UI
st.title("📰 Fake News Detection App")

st.write("Enter a news article below:")

user_input = st.text_area("News Text")

if st.button("Check News"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.success("✅ This is REAL news")
        else:
            st.error("❌ This is FAKE news")
    else:
        st.warning("Please enter some text")