import json
import os
import plotly.figure_factory as ff
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    cm = confusion_matrix(y_true, y_pred)

    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\n=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(cm)
    return accuracy, report, cm

def save_metrics(accuracy, report, output_dir='outputs', filename='metrics.json'):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {'accuracy': round(accuracy, 4),  'classification_report': report}
    metrics_path = os.path.join(output_dir, filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

def save_confusion_matrix(cm, output_dir='outputs/figures', filename='confusion_matrix.png', save_as_png=True):
    os.makedirs(output_dir, exist_ok=True)
    cm_list = cm.tolist()
    fig = ff.create_annotated_heatmap(z=cm_list,
                                      x=['Negative', 'Positive'], y=['Negative', 'Positive'], 
                                      colorscale='Blues', showscale=True)
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted', yaxis_title='Actual'
    )
    fig_path = os.path.join(output_dir, filename)
    fig.write_image(fig_path)
    print(f"Confusion matrix saved to {fig_path}")