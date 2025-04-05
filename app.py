import streamlit as st
import nltk
import os

nltk.download('punkt_tab')  # Only needs to run once

# Set corpus directory path (change if needed)
CORPUS_DIR = "corpus"

@st.cache_resource
def load_data(directory):
    result = []
    for filename in ["positives.txt", "negatives.txt"]:
        with open(os.path.join(directory, filename), encoding='utf-8') as f:
            result.append([
                extract_words(line)
                for line in f.read().splitlines()
            ])
    return result

def extract_words(document):
    return set(
        word.lower() for word in nltk.word_tokenize(document)
        if any(c.isalpha() for c in word)
    )

def generate_features(documents, words, label):
    features = []
    for document in documents:
        features.append(({
            word: (word in document)
            for word in words
        }, label))
    return features

def classify(classifier, document, words):
    document_words = extract_words(document)
    features = {
        word: (word in document_words)
        for word in words
    }
    return classifier.prob_classify(features)

# Streamlit UI
st.title("Naive Bayes Sentiment Analyzer")
st.markdown("This app classifies text as **Positive** or **Negative** using a simple Naive Bayes classifier.")

# Load corpus and train classifier
try:
    positives, negatives = load_data(CORPUS_DIR)

    words = set()
    for document in positives + negatives:
        words.update(document)

    training = []
    training.extend(generate_features(positives, words, "Positive"))
    training.extend(generate_features(negatives, words, "Negative"))

    classifier = nltk.NaiveBayesClassifier.train(training)

    # Input from user
    user_input = st.text_area("Enter a sentence to analyze:", height=150)

    if st.button("Analyze"):
        if user_input.strip():
            result = classify(classifier, user_input, words)
            prediction = result.max()
            prob = result.prob(prediction)
            st.subheader(f"Prediction: {prediction}")
            st.write(f"Confidence: {prob:.4f}")
        else:
            st.warning("Please enter a sentence to analyze.")

except Exception as e:
    st.error(f"Error: {e}")
