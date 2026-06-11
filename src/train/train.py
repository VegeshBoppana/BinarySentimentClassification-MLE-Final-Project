import os
import sys
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Add src to path so that we can import functions from preprocess and evaluate
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess import preprocess_dataframe
from evaluate import evaluate_model, save_metrics, save_confusion_matrix

# Paths
DATA_PATH = 'data/raw/train.csv'
MODEL_DIR = 'outputs/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

def train():
    # Step 1 - Load Data
    print("=== Step 1: Loading Data ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews")

    # Step 2 - Preprocess
    print("\n=== Step 2: Preprocessing ===")
    df = preprocess_dataframe(df, text_column='review')
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Step 3 - Split
    print("\n=== Step 3: Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 4 - Vectorize
    print("\n=== Step 4: Vectorizing ===")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Step 5 - Train
    print("\n=== Step 5: Training LinearSVC ===")
    model = LinearSVC(random_state=42, max_iter=10000, dual='auto')
    model.fit(X_train_vec, y_train)
    print("Training complete!")

    # Step 6 - Evaluate
    print("\n=== Step 6: Evaluating ===")
    y_pred = model.predict(X_test_vec)
    accuracy, report, cm = evaluate_model(y_test, y_pred)

    # Step 7 - Save
    print("\n=== Step 7: Saving ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    save_metrics(accuracy, report, output_dir='outputs', filename='train_metrics.json')
    save_confusion_matrix(cm, output_dir='outputs/figures', filename='train_confusion_matrix.png')

    print("\n=== Training Complete! ===")
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()