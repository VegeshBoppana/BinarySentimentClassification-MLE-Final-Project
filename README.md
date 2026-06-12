# Binary Sentiment Classification — IMDB Movie Reviews

## Project Overview
Binary sentiment classification on IMDB movie reviews using NLP and Machine Learning.
- **Task:** Classify movie reviews as positive or negative
- **Dataset:** 50,000 IMDB movie reviews (40,000 train, 10,000 inference)
- **Best Model:** LinearSVC + TF-IDF Bigram
- **Best Accuracy:** 0.9031 (inference)

---

## DS Part

### EDA Conclusions
- Dataset is perfectly balanced — 20,000 positive, 20,000 negative reviews
- No missing values found
- Average review length: 231 words (range: 4 to 2,470 words)
- Review length is identical across both classes — not a useful feature
- 23,278 out of 40,000 reviews contain HTML tags (e.g. `<br />`)
- Bigram analysis confirmed clear vocabulary separation between classes:
    - Positive signals: "well done", "highly recommend", "must see"
    - Negative signals: "waste time", "worst ever", "bad acting"

### Feature Engineering

#### Text Preprocessing Pipeline
1. **HTML tag removal** — 23,278 reviews contained raw HTML
2. **Lowercasing** — normalize case
3. **Special character removal** — remove punctuation and numbers
4. **Tokenization** — split text into individual tokens
5. **Stopword removal** — removed NLTK stopwords + custom words (film, movie, movies, films, one, br)
6. **Lemmatization** — convert words to base dictionary form

#### Stemming vs Lemmatization
- Both were applied and compared
- Lemmatization produces real words → better vocabulary quality
- Stemming reduces vocabulary size (42,767 vs 50,000) but produces non-real words
- Lemmatization chosen for final model

#### Vectorization Comparison
| Approach | Description |
|----------|-------------|
| BoW (CountVectorizer) | Raw word counts |
| TF-IDF Unigrams | Weighted word importance |
| TF-IDF Bigrams | Weighted word + phrase importance |
| TF-IDF Stemmed | TF-IDF on stemmed text |

- TF-IDF consistently outperforms BoW
- Bigrams capture sentiment phrases like "waste time", "highly recommend"
- TF-IDF Bigram is the best vectorization approach

### Model Selection

#### Models Evaluated
| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| Logistic Regression | BoW | 0.8788 |
| Naive Bayes | BoW | 0.8568 |
| LinearSVC | BoW | 0.8568 |
| Logistic Regression | TF-IDF | 0.8901 |
| Naive Bayes | TF-IDF | 0.8641 |
| LinearSVC | TF-IDF | 0.8871 |
| Logistic Regression | TF-IDF Bigram | 0.8955 |
| Naive Bayes | TF-IDF Bigram | 0.8795 |
| **LinearSVC** | **TF-IDF Bigram** | **0.8979** | |**WINNER**|
| Logistic Regression | TF-IDF Stem | 0.8911 |
| Naive Bayes | TF-IDF Stem | 0.8565 |
| LinearSVC | TF-IDF Stem | 0.8875 |

#### Bonus — Dense Vectorization
| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| LinearSVC | GloVe (avg pooling) | 0.7929 |
| Decision Tree | GloVe (avg pooling) | 0.6631 |

#### Best Model — LinearSVC + TF-IDF Bigram
- **Accuracy: 0.8979 (train/test split), 0.9031 (inference)** 
- LinearSVC handles high dimensional sparse matrices excellently
- TF-IDF Bigram captures both individual words and sentiment phrases
- Fast training and inference
- No probability calibration needed

### Performance Evaluation
| Metric | Negative | Positive |
|--------|----------|----------|
| Precision | 0.91 | 0.90 |
| Recall | 0.90 | 0.91 |
| F1-Score | 0.90 | 0.90 |
| **Accuracy** | | **0.9031** |

### Potential Business Applications
- **Review Moderation** — automatically flag negative reviews for follow-up
- **Customer Feedback Analysis** — analyze thousands of reviews instantly
- **Brand Monitoring** — track sentiment of product/service reviews in real time
- **Recommendation Systems** — use sentiment scores to improve recommendations
- **Market Research** — understand customer sentiment trends over time

### Note on Dense Embeddings
Word2Vec and GloVe embeddings were explored as a bonus. Average pooling over 
word vectors resulted in lower accuracy (0.79) compared to TF-IDF (0.90) because:
- Long reviews dilute strong sentiment words during averaging
- TF-IDF rewards rare but discriminative words like "masterpiece" or "unwatchable"
- BERT-based models would likely outperform TF-IDF but require GPU resources

---