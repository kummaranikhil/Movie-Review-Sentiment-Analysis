import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# load stopwords
stop_words = set(stopwords.words('english'))

# load trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

# load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# text cleaning function
def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split words
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # join cleaned words
    return " ".join(words)


# sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.write(
    """
    This AI application analyzes movie reviews 
    and predicts whether the sentiment is 
    positive or negative using Natural Language Processing.
    """
)

st.sidebar.subheader("🛠 Technologies Used")

st.sidebar.write(
    """
    - Python
    - Streamlit
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - NLP
    """
)

st.sidebar.subheader("👨‍💻 Developed By")

st.sidebar.write("Nikhil")


# custom CSS styling
st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Text area styling */
    textarea {
        border-radius: 10px !important;
        border: 2px solid #FF4B4B !important;
        padding: 10px !important;
        font-size: 16px !important;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }

    /* Button hover effect */
    div.stButton > button:hover {
        background-color: #ff2e2e;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# banner image
st.image(
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
    use_container_width=True
)


# professional heading
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
        🎬 Movie Review Sentiment Analyzer
    </h1>

    <p style='text-align: center; font-size:18px; color:white;'>
        AI-powered NLP application that predicts whether a movie review is positive or negative.
    </p>
    """,
    unsafe_allow_html=True
)

# user input section
st.subheader("✍️ Enter Your Movie Review")

user_review = st.text_area(
    "",
    placeholder="Type your movie review here...",
    height=180
)

# prediction button
if st.button("🔍 Predict Sentiment"):

    # check empty review
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")

    else:

        # loading spinner
        with st.spinner("Analyzing Review..."):

            # clean review
            cleaned_review = clean_text(user_review)

            # convert to vector
            vector = vectorizer.transform([cleaned_review])

            # prediction
            prediction = model.predict(vector)[0]

            # prediction probabilities
            probability = model.predict_proba(vector)

            # confidence score
            confidence = round(max(probability[0]) * 100, 2)

            # positive result
            if prediction == 1:

                st.success(
                    f"😊 Positive Review\n\nConfidence Score: {confidence}%"
                )

                st.balloons()

            # negative result
            else:

                st.error(
                    f"😡 Negative Review\n\nConfidence Score: {confidence}%"
                )

# footer
st.markdown("---")

st.markdown(
    """
    <p style='text-align:center; color:gray;'>
        Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)