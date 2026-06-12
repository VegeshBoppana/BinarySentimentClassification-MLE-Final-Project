import os
import sys
import joblib
import pandas as pd
import mlflow

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess import preprocess_dataframe
from evaluate import evaluate_model, save_metrics, save_confusion_matrix

INFERENCE_DATA_PATH = 'data/raw/inference.csv'
MODEL_PATH = 'outputs/models/model.pkl'
VECTORIZER_PATH = 'outputs/models/vectorizer.pkl'
PREDICTIONS_PATH = 'outputs/predictions/inference_predictions.csv'

def run_inference():
    mlflow.set_experiment("sentiment_inference")
    with mlflow.start_run(run_name="LinearSVC_TFIDFBigram_Inference"):

        # Step 1 - Load Model and Vectorizer
        print("=== Step 1: Loading Model and Vectorizer ===")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first!")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Run train.py first!")

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and Vectorizer loaded successfully!")
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("vectorizer", "TF-IDF Bigram")

        # Step 2 - Load Inference Data
        print("\n=== Step 2: Loading Inference Data ===")
        if not os.path.exists(INFERENCE_DATA_PATH):
            raise FileNotFoundError(f"Inference data not found at {INFERENCE_DATA_PATH}!")

        df = pd.read_csv(INFERENCE_DATA_PATH)
        print(f"Loaded {len(df)} reviews")
        mlflow.log_param("inference_dataset_size", len(df))

        # Step 3 - Preprocess
        print("\n=== Step 3: Preprocessing ===")
        df = preprocess_dataframe(df, text_column='review')

        # Log preprocessed inference data
        os.makedirs('outputs/data', exist_ok=True)
        preprocessed_path = 'outputs/data/preprocessed_inference.csv'
        df[['review', 'cleaned_review']].to_csv(preprocessed_path, index=False)
        mlflow.log_artifact(preprocessed_path, artifact_path="data")

        # Step 4 - Vectorize
        print("\n=== Step 4: Vectorizing ===")
        X = vectorizer.transform(df['cleaned_review'])
        print(f"Vectorized shape: {X.shape}")
        mlflow.log_param("vectorized_shape", str(X.shape))

        # Step 5 - Predict
        print("\n=== Step 5: Predicting ===")
        predictions = model.predict(X)
        df['predicted_label'] = predictions
        df['predicted_sentiment'] = df['predicted_label'].map({1: 'positive', 0: 'negative'})
        print("Predictions complete!")

        # Step 6 - Evaluate
        print("\n=== Step 6: Evaluating ===")
        if 'sentiment' in df.columns:
            df['true_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
            accuracy, report, cm = evaluate_model(df['true_label'], df['predicted_label'])
            mlflow.log_metric("accuracy", round(accuracy, 4))

            save_metrics(accuracy, report, output_dir='outputs', filename='inference_metrics.json')
            mlflow.log_artifact('outputs/inference_metrics.json', artifact_path="metrics")

            save_confusion_matrix(cm, output_dir='outputs/figures', filename='inference_confusion_matrix.png')
            mlflow.log_artifact('outputs/figures/inference_confusion_matrix.png', artifact_path="figures")
        else:
            print("No labels found — skipping evaluation")

        # Step 7 - Save Predictions
        print("\n=== Step 7: Saving Predictions ===")
        os.makedirs('outputs/predictions', exist_ok=True)
        df[['review', 'predicted_sentiment']].to_csv(PREDICTIONS_PATH, index=False)
        mlflow.log_artifact(PREDICTIONS_PATH, artifact_path="predictions")
        print(f"Predictions saved to {PREDICTIONS_PATH}")

        print("\n=== Inference Complete! ===")

if __name__ == "__main__":
    run_inference()