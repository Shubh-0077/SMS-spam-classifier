import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered",
)

# -------------------- LOAD MODEL -------------------- #
@st.cache_resource
def load_model():
    tfidf = pickle.load(open("models/vectorizer.pkl", "rb"))
    model = pickle.load(open("models/model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_model()

# -------------------- NLP -------------------- #
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    words = []

    for word in text:
        if word.isalnum():
            words.append(word)

    filtered = []

    for word in words:
        if word not in stopwords.words("english") and word not in string.punctuation:
            filtered.append(word)

    stemmed = []

    for word in filtered:
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)

# -------------------- CSS -------------------- #
st.markdown("""
<style>

.main{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:800;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#8b949e;
    margin-bottom:35px;
    font-size:17px;
}

textarea{
    font-size:17px !important;
}

.stButton{
    display:flex;
    justify-content:center;
    margin-top:20px;
}

.stButton>button{
    width:250px;
    height:55px;
    border-radius:12px;
    border:none;
    background:#4F46E5;
    color:white;
    font-size:18px;
    font-weight:600;
}

.stButton>button:hover{
    background:#4338CA;
}

.result{
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER -------------------- #
st.markdown('<div class="title">📩 Email / SMS Spam Classifier</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Paste your message below and let the AI determine whether it is Spam or Not Spam.</div>',
    unsafe_allow_html=True,
)

# -------------------- INPUT -------------------- #
input_sms = st.text_area(
    "Enter your message",
    placeholder="Type or paste your SMS / Email here...",
    height=180,
)

# -------------------- BUTTON -------------------- #
predict = st.button("🔍 Predict")

# -------------------- PREDICTION -------------------- #
if predict:

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        with st.spinner("Analyzing..."):

            transformed_sms = transform_text(input_sms)

            vector = tfidf.transform([transformed_sms])

            prediction = model.predict(vector)[0]

            confidence = model.predict_proba(vector).max() * 100

        st.divider()

        if prediction == 1:

            st.error("🚨 Spam Message")

        else:

            st.success("✅ Not Spam")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        with st.expander("🔎 Processed Text"):

            st.code(transformed_sms)

# -------------------- FOOTER -------------------- #
st.markdown("---")

st.markdown(
    """
<div class="footer">
Built with ❤️ using Streamlit, Scikit-Learn & NLTK | Shubham Malkar
</div>
""",
    unsafe_allow_html=True,
)
