import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt

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


# ---------------- TEXT CLEANING FUNCTION ---------------- #

def clean_text(text):

    # convert text to lowercase
    text = text.lower()

    # remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split into words
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
    - Matplotlib
    """
)

st.sidebar.subheader("👨‍💻 Developed By")

st.sidebar.write("Nikhil")


# ---------------- CUSTOM CSS ---------------- #

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Text area */
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

    /* Button hover */
    div.stButton > button:hover {
        background-color: #ff2e2e;
        color: white;
    }

    /* Metric card */
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


# ---------------- SESSION STATE ---------------- #

if 'history' not in st.session_state:
    st.session_state.history = []


# ---------------- USER INPUT ---------------- #

st.subheader("✍️ Enter Your Movie Review")

user_input = st.text_area(
    "",
    placeholder="Type your movie review here...",
    height=180
)


# ---------------- PREDICTION ---------------- #

if st.button("🔍 Analyze Sentiment"):

    # check empty input
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")

    else:

        with st.spinner("Analyzing Review..."):

            # clean text
            cleaned = clean_text(user_input)

            # convert into TF-IDF vectors
            vector_input = vectorizer.transform([cleaned])

            # predict sentiment
            prediction = model.predict(vector_input)[0]

            # probability values
            probability = model.predict_proba(vector_input)

            confidence = round(max(probability[0]) * 100, 2)

            # positive & negative probabilities
            positive_prob = round(probability[0][1] * 100, 2)
            negative_prob = round(probability[0][0] * 100, 2)

            # sentiment label
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"

            # report content
            report = f"""
Movie Review Sentiment Analysis Report

Review:
{user_input}

Predicted Sentiment:
{sentiment}

Confidence Score:
{confidence}%
"""

            # save history
            st.session_state.history.append({
                "review": user_input,
                "sentiment": sentiment,
                "confidence": confidence
            })

            # ---------------- POSITIVE RESULT ---------------- #

            if prediction == 1:

                st.success("😊 Positive Review")

                # progress bar
                st.progress(int(confidence))

                # metric card
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence}%"
                )

                # balloons animation
                st.balloons()

            # ---------------- NEGATIVE RESULT ---------------- #

            else:

                st.error("😡 Negative Review")

                # progress bar
                st.progress(int(confidence))

                # metric card
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence}%"
                )

            # ---------------- BAR CHART ---------------- #

            st.subheader("📊 Sentiment Probability Analysis")

            labels = ['Positive', 'Negative']
            values = [positive_prob, negative_prob]

            fig, ax = plt.subplots()

            ax.bar(labels, values)

            ax.set_ylabel("Probability (%)")
            ax.set_title("Sentiment Probability")

            st.pyplot(fig)

            # ---------------- PIE CHART ---------------- #

            st.subheader("🥧 Sentiment Distribution")

            fig2, ax2 = plt.subplots()

            ax2.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90
            )

            ax2.axis('equal')

            st.pyplot(fig2)

            # ---------------- DOWNLOAD REPORT ---------------- #

            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="sentiment_report.txt",
                mime="text/plain"
            )


# ---------------- REVIEW HISTORY ---------------- #

st.subheader("🕘 Review History")

if len(st.session_state.history) == 0:

    st.info("No reviews analyzed yet.")

else:

    for item in reversed(st.session_state.history):

        st.markdown(
            f"""
            <div style="
                background-color:#262730;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
            ">

            <b>Review:</b> {item['review']} <br><br>

            <b>Sentiment:</b> {item['sentiment']} <br><br>

            <b>Confidence:</b> {item['confidence']}%

            </div>
            """,
            unsafe_allow_html=True
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