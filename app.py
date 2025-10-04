import gradio as gr
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os
from mlp import MLPModule

# Load model + explainer
with open("NeuralHestonRegressor.pkl", "rb") as f:
    model = joblib.load(f)

with open("ModelExplainer.pkl", "rb") as f:
    explainer = joblib.load(f)

# Folder for SHAP plots
PLOT_DIR = "shap_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def predict_and_explain(file):
    # Load CSV
    df = pd.read_csv(file.name)

    # Predictions
    preds = model.predict(df)

    # SHAP values
    shap_values = explainer.shap_values(df)

    # Save SHAP summary plot
    plot_path = os.path.join(PLOT_DIR, f"{uuid.uuid4()}.png")
    plt.figure()
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return preds.tolist(), plot_path

# Define Gradio interface
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.File(label="Upload Features CSV", file_types=[".csv"]),
    outputs=[
        gr.Textbox(label="Predictions"),
        gr.Image(label="SHAP Summary Plot")
    ],
    title="Stock Prediction + SHAP Explainability",
    description="Upload a CSV with features, get predictions and see explainability via SHAP."
)

if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)
