import os
import sys
import joblib
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess import preprocess_dataframe
from evaluate import evaluate_model, save_metrics, save_confusion_matrix


INFERENCE_DATA_PATH = 'data/raw/final_project_inference_dataset/final_project_inference_dataset/inference.csv'
MODEL_PATH = 'outputs/models/model.pkl'
VECTORIZER_PATH = 'outputs/models/vectorizer.pkl'
PREDICTIONS_PATH = 'outputs/predictions/inference_predictions.csv'

def run_inference():
    print("=== Step 1: Loading Model and Vectorizer ===")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first!")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Run train.py first!")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully!")

    print("\n=== Step 2: Loading Inference Data ===")
    if not os.path.exists(INFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Inference data not found at {INFERENCE_DATA_PATH}!")
    
    df = pd.read_csv(INFERENCE_DATA_PATH)
    print(f"Loaded {len(df)} reviews")


    print("\n=== Step 3: Preprocessing ===")
    df = preprocess_dataframe(df, text_column='review')


    print("\n=== Step 4: Vectorizing ===")
    X = vectorizer.transform(df['cleaned_review'])
    print(f"Vectorized shape: {X.shape}")


    print("\n=== Step 5: Predicting ===")
    predictions = model.predict(X)
    df['predicted_label'] = predictions
    df['predicted_sentiment'] = df['predicted_label'].map({1: 'positive', 0: 'negative'})
    print("Predictions complete!")


    print("\n=== Step 6: Evaluating ===")
    if 'sentiment' in df.columns:
        df['true_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        accuracy, report, cm = evaluate_model(df['true_label'], df['predicted_label'])
        save_metrics(accuracy, report, output_dir='outputs', filename='inference_metrics.json')
        save_confusion_matrix(cm, output_dir='outputs/figures', filename='inference_confusion_matrix.png')
    else:
        print("No labels found — skipping evaluation")


    print("\n=== Step 7: Saving Predictions ===")
    os.makedirs('outputs/predictions', exist_ok=True)
    df[['review', 'predicted_sentiment']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    print("\n=== Inference Complete! ===")

if __name__ == "__main__":
    run_inference()