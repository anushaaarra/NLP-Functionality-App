import streamlit as st
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator
from nltk.corpus import reuters
from nltk import ngrams, FreqDist
import random
import nltk
import sentencepiece
import yake
import heapq
import openai

openai.api_key = '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'
paraphrase_prompt = "Paraphrase this: "
summarise_prompt = "Summarise this:  "
prediction_prompt = "Predict the next word after the following sentence: "

# Function to set a common style for the app

custom_css = """
        <style>
        .stApp {
                background-image: url('https://i.pinimg.com/originals/b1/87/17/b18717768d1f4c87d5ef621915ab3f88.jpg');
                background-size: cover;
            }
        .big-font {
        font-size:300px !important;
        }

        .main {
            
            
            font-family: 'Comic Sans', sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        
    
        </style>
        """
st.set_page_config("NLP Functionality App")


def set_app_style():
    st.markdown(custom_css, unsafe_allow_html=True)


set_app_style()


def extract_keywords_from_text(text, numOfKeywords):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords


def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        sentiment_label = 'Positive'
    elif sentiment_score < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    return sentiment_score, sentiment_label


def similarity_check(text1, text2):
    vectorizer = TfidfVectorizer()
    text1_vector = vectorizer.fit_transform([text1])
    text2_vector = vectorizer.transform([text2])
    similarity_score = cosine_similarity(text1_vector, text2_vector)
    return similarity_score[0][0]


def paraphrase_text(text, api_key):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=paraphrase_prompt + text,
        max_tokens=200
    )
    return response.choices[0].text.strip()


# Example usage
def summarise_text(text, api_key):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=summarise_prompt + text,
        max_tokens=400
    )
    return response.choices[0].text.strip()


# Example usage
def predict_text(text, api_key):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prediction_prompt + text,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def spell_check(text):
    blob = TextBlob(text)
    # Perform spell check and correction
    corrected_text = str(blob.correct())
    return corrected_text


def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text


def main():
    set_app_style()
    st.title("NLP Functionality App")

    function_options = [
        "Sentiment Analysis",
        "Keyword Extraction",
        "Similarity Check",
        "Summarizer",
        "Translator",
        "Text Prediction",
        "Paraphrasing",
        "Spell Check and Corrector",
    ]

    selected_function = st.sidebar.selectbox("Select NLP Function", function_options)

    # for option in function_options:
    #    if st.sidebar.button(option, key=option):
    #       selected_function=option

    st.markdown("---")

    if selected_function == "Sentiment Analysis":
        st.header("Sentiment Analysis", divider='rainbow')
        text = st.text_area("", height=200, placeholder="Enter text for sentiment analysis..")
        if st.button("Analyze"):
            sentiment_score, sentiment_label = perform_sentiment_analysis(text)
            st.write(f"Sentiment Score: {sentiment_score}")
            st.write(f"Sentiment Label: {sentiment_label}")

    elif selected_function == "Keyword Extraction":
        st.subheader("Keyword Extraction", divider='rainbow')
        paragraph = st.text_area("", height=200, placeholder="Enter text for keyword extraction..")
        num_keywords = st.number_input("Number of Keywords", value=5)
        if st.button("Extract Keywords"):
            keywords = extract_keywords_from_text(paragraph, num_keywords)
            st.write("Keywords with Scores:")
            for keyword, score in keywords:
                st.write(f"{keyword} : {score}")

    elif selected_function == "Similarity Check":
        st.subheader("Similarity Check", divider='rainbow')
        text1 = st.text_area("", height=200, placeholder="Enter first text..")
        text2 = st.text_area("", height=200, placeholder="Enter second text..")
        if st.button("Check Similarity"):
            similarity_score = similarity_check(text1, text2)
            st.write(f"Similarity Score: {similarity_score}")

    elif selected_function == "Summarizer":
        st.subheader("Summarizer", divider='rainbow')
        text = st.text_area("", height=200, placeholder="Enter text to summarize..")
        if st.button("Generate Summary"):
            summary = summarise_text(text, openai.api_key)
            st.write("Summary:")
            st.write(summary)


    elif selected_function == "Translator":
        st.subheader("Translator", divider='rainbow')
        text = st.text_area("", height=200, placeholder="Enter text to translate..")
        target_language = st.text_input("Enter Target Language")
        if st.button("Translate"):
            translated_text = translate_text(text, target_language)
            st.write(f"Translated Text: {translated_text}")

    elif selected_function == "Text Prediction":
        st.subheader("Text Prediction", divider='rainbow')
        input_text = st.text_input("Enter Text", value="")
        if st.button("Generate Prediction"):
            predicted_text = predict_text(input_text, openai.api_key)
            st.write(f"Predicted Text: {predicted_text}")

    elif selected_function == "Spell Check and Corrector":
        st.subheader("Spell Check and Corrector", divider='rainbow')
        text = st.text_area("", height=200, placeholder="Enter text to check grammar and spelling..")
        if st.button("Check and Correct"):
            corrected_text = spell_check(text)
            st.write(f"Corrected Text: {corrected_text}")

    elif selected_function == "Paraphrasing":
        st.subheader("Paraphrasing", divider='rainbow')
        text = st.text_area("", height=200, placeholder="Enter text to paraphrase..")
        if st.button("After paraphrasing"):
            paraphrased_text = paraphrase_text(text, openai.api_key)
            st.write(f"Paraphrased Text: {paraphrased_text}")


if __name__ == '__main__':
    main()