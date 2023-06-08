# main.py
from fastapi import FastAPI, HTTPException
from nltk.stem import SnowballStemmer
from nltk import bigrams
from nltk.tokenize import word_tokenize
from pydantic import BaseModel


app = FastAPI()

stemmer = SnowballStemmer('english')


class Text(BaseModel):
    text: str


def preprocess_text(text):
    # preprocess
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    bigram_tokens = ['_'.join(bigram) for bigram in bigrams(tokens)]
    return ' '.join(stemmed_tokens + bigram_tokens)


@app.post("/preprocess/")
def preprocess(text: Text):
    preprocessed_text = preprocess_text(text.text)
    return {"preprocessed_text": preprocessed_text}
