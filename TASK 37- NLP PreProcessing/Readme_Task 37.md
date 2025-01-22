# Task 37 - NLP Preprocessing

This project contains various text preprocessing techniques applied to a dataset. The goal is to clean and prepare text data for Natural Language Processing (NLP) tasks, which involves transforming raw text into a format suitable for modeling.

## Techniques Implemented:

## 1. Lowercasing
Converts all text to lowercase to ensure uniformity and prevent the model from treating the same word differently due to case differences.

```python
df['review'] = df['review'].str.lower()
```

## 2. Removing Punctuation & Special Characters
Removes all punctuation marks to focus solely on the words in the text.

```python
import string
def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))
df['review'] = df['review'].apply(remove_punc)
```

### 3. Removal of HTML Tags
Removes HTML tags from the text to clean up any web-specific formatting.

```python
import re
def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)
df['review'] = df['review'].apply(remove_html_tags)
```

### 4. Removal of URLs
Eliminates URLs from the text as they do not add meaningful content for analysis.

```python
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
df['review'] = df['review'].apply(remove_url)
```

## 5. Handling Spelling Mistakes
Uses `TextBlob` to automatically correct spelling mistakes in the text.

```python
from textblob import TextBlob
text = 'I amm ussing Jupyterr Notebook'
corrected_text = str(TextBlob(text).correct())
```

## 6. Removing Stop Words
Removes common stop words (e.g., "and", "the") that do not contribute much meaning to the text.

```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])
df['review'] = df['review'].apply(remove_stopwords)
```

## 7. Removing Emojis
Removes emojis from the text to focus on the textual content.

```python
import emoji
text_with_emoji = 'Bytewise DataScience fellowship is 🔥'
cleaned_text = emoji.demojize(text_with_emoji)
```

## 8. Tokenization
Breaks down text into smaller, more manageable units, such as words or sentences.

```python
from nltk.tokenize import word_tokenize, sent_tokenize
word_tokens = word_tokenize(text)
sent_tokens = sent_tokenize(text)
```

## 9. Stemming and Lemmatization
Reduces words to their root form (stemming) or dictionary form (lemmatization) to standardize variations of words.

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_word = stemmer.stem("running")

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("running")
lemmatized_word = [token.lemma_ for token in doc]
```

## 10. Removing Numbers
Removes numerical values from the text to focus solely on textual content.

```python
import re
def remove_digits(text):
    return re.sub(r'\d+', '', text)
df['review'] = df['review'].apply(remove_digits)
```

## Dataset
The dataset used is the IMDB movie reviews dataset, which consists of movie reviews and their sentiment labels.

- `review`: Contains the text of the review.
- `sentiment`: Contains the sentiment label (positive/negative).
_________________

