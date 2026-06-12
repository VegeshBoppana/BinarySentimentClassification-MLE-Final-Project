import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({'film', 'movie', 'movies', 'films', 'one', 'br'})

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Step 1 - Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Step 2 - Lowercase
    text = text.lower()
    # Step 3 - Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Step 4 - Tokenize
    tokens = word_tokenize(text)
    # Step 5 - Remove stopwords and short words
    tokens = [t for t in tokens if t not in custom_stopwords and len(t) > 2]
    # Step 6 - Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column='review'):
    print(f"Preprocessing {len(df)} reviews")
    df['cleaned_review'] = df[text_column].apply(preprocess_text)
    print("Preprocessing complete!")
    return df