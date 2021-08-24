import streamlit as st
import os
import pickle
import nltk # Text libarary
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer


from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()


def identify_tokens2(row):
    text = row
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def lemmatize_list2(row):
    my_list = row
    lemmatized_list = [lemmatizer.lemmatize(word) for word in my_list]
    return (lemmatized_list)

def remove_stops2(row):
    my_list = row
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = identify_tokens2(review)
    review_c = lemmatize_list2(review_c)
    review_c = remove_stops2(review_c)
    review_c = " ".join(review_c)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"


model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_vect = pickle.load(open(vectorizer_name, 'rb'))

def main():
    st.title("Amazon Food Review")
    text = st.text_area("Enter Your Review","Type Here")
    if st.button("Analyze"):
        res = raw_test(text, loaded_model, loaded_vect)
        st.success(res)

if __name__ == '__main__':
	main()
