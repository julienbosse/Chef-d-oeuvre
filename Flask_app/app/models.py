import numpy as np
import time
import re
from nltk.corpus import stopwords


# Cleaning function

#Remove Urls and HTML links
def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Lower casing
def lower(text):
    low_text= text.lower()
    return low_text

# Number removal
def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove

#Remove stopwords & Punctuations
def remove_stopwords(text,STOPWORDS):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
def remove_punct(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct

def clean_text(text):

    STOPWORDS = set(stopwords.words('english'))

    cleaned = remove_urls(text) # Remove Urls
    cleaned = remove_html(cleaned) # Remove HTML links
    cleaned = remove_punct(cleaned) # Remove Punctuations
    cleaned = lower(cleaned) # Lower casing
    cleaned = remove_num(cleaned) # Remove numbers
    cleaned = remove_stopwords(cleaned,STOPWORDS) # Remove stopwords

    return cleaned

def process_text(text):

    text = text.replace(' ','_')
    text = text.upper()

    return text
