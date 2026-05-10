import streamlit as st
import pickle
import re
import nltk

# download stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords

# page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
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

    # convert text to lowercase
    text = text.lower()

    # remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split text into words
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # join words again
    return " ".join(words)


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📌 Project Information")

st.sidebar.write(
    """
    This AI application analyzes movie reviews 
    and predicts whether the sentiment is 
    positive or negative using NLP and Machine Learning.
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
    - NLTK
    """
)

st.sidebar.subheader("👨‍💻 Developed By")

st.sidebar.write("Nikhil")


# ---------------- CUSTOM CSS ---------------- #

st.markdown(
    """
    <style>

    /* Main App Background */
    .stApp {
        background-color: #0E1117;
    }

    /* Text Area */
    textarea {
        border-radius: 10px !important;
        border: 2px solid #FF4B4B !important;
        padding: 10px !important;
        font-size: 16px !important;
    }

    /* Button Styling */
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

    /* Button Hover Effect */
    div.stButton > button:hover {
        background-color: #ff2e2e;
        color: white;
    }

    /* Metric Styling */
    [data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- BANNER IMAGE ---------------- #

st.image(
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
    use_container_width=True
)


# ---------------- TITLE ---------------- #

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


# ---------------- USER INPUT ---------------- #

st.subheader("✍️ Enter Your Movie Review")

user_input = st.text_area(
    "",
    placeholder="Type your movie review here...",
    height=180
)


# ---------------- PREDICTION ---------------- #

if st.button("🔍 Analyze Sentiment"):

    # empty input validation
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")

    else:

        with st.spinner("Analyzing Review..."):

            # clean text
            cleaned = clean_text(user_input)

            # convert text to TF-IDF vectors
            vector_input = vectorizer.transform([cleaned])

            # predict sentiment
            prediction = model.predict(vector_input)[0]

            # probability/confidence
            probability = model.predict_proba(vector_input)

            confidence = round(max(probability[0]) * 100, 2)

            # positive prediction
            if prediction == 1:

                st.success("😊 Positive Review")

                # progress bar
                st.progress(int(confidence))

                # confidence score card
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence}%"
                )

                # celebration animation
                st.balloons()

            # negative prediction
            else:

                st.error("😡 Negative Review")

                # progress bar
                st.progress(int(confidence))

                # confidence score card
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence}%"
                )


# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown(
    """
    <p style='text-align:center; color:gray;'>
        Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)