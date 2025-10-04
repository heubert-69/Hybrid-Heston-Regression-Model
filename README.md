## Hybrid Heston Regression Model
This project provides an interactive web app for stock prediction using a Hybrid Heston-inspired ML model with PyTorch + Skorch, and SHAP explainability for model interpretation.
The app is built with Gradio, making it easy to run locally or deploy to Hugging Face Spaces, Docker, or cloud platforms.
--

🚀 Features

📈 Stock prediction using a trained PyTorch model
🔍 Explainability with SHAP → feature importance visualization
📂 Upload a CSV of features (prices, returns, volatility, etc.) to get predictions
🌐 Interactive UI with Gradio (no need to build HTML/CSS manually)
📊 Full model evaluation metrics & trading performance results
☁️ Deployed on HuggingFace Open Spaces

--

📂 Project Structure
```bash
HestonHybridModel
│
├── HybridHestonModel.ipynb
├── ModelExplainer.pkl
├── NeuralHestonRegressor.pkl
├── Visualizations
│   ├── Final Results of Modelling.png
│   ├── Model Explaination.png
│   └── Visualization Phase.png
├── app.py
├── exported_data
│   ├── allocations.csv
│   ├── goog_aapl_msft_amzn.csv
│   └── preds.csv
└── requirements.txt
└── README.md           # Project documentation
```

--

⚙️ Installation
1. Clone the repo
```bash
git clone https://github.com/yourusername/stock-shap-app.git
cd stock-shap-app
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

▶️ Running the App
Local Run:
```bash
python app_gradio.py
```

Then open:
👉 http://127.0.0.1:7860
---

📊 Input Format
Upload a CSV file with your feature set. Example:

AAPL_price	AAPL_ret	AAPL_ma5	AAPL_ma20	AAPL_rv_5	TSLA_price	TSLA_ma5	...
172.31	0.0012	172.00	170.23	0.012	250.12	249.87	...
---
📑 Results & Analysis
🔹 ML Metrics Per Asset
```bash
| Asset | RMSE   | MAE   | R²    | Directional Accuracy |
|-------|--------|-------|-------|-----------------------|
| AAPL  | 0.01896 | 0.01341 | 0.0693 | 55.91% |
| MSFT  | 0.02128 | 0.01523 | 0.0426 | 56.69% |
| GOOG  | 0.01904 | 0.01357 | 0.0525 | 57.42% |
| AMZN  | 0.01718 | 0.01225 | 0.0949 | 56.17% |
| TSLA  | 0.03900 | 0.02821 | 0.0690 | 57.05% |
```
➡️ The model achieves ~53–57% directional accuracy, which is significant for short-term trading.
---

🔹 Trading Performance:
Annual Return: 50.0%
Annual Volatility: 14.8%
Sharpe Ratio: 3.37
Sortino Ratio: 5.13
Max Drawdown: -18.3

➡️ High Sharpe & Sortino ratios indicate strong risk-adjusted returns, with relatively controlled drawdowns.

🔹 SHAP Explainability Insights

Most Important Features:

TSLA 5-day MA & price

AAPL 5-day volatility & returns

MSFT/GOOG/AMZN short-term signals also contribute

Model Behavior:
High short-term momentum (5-day MA ↑) → prediction increases
Low recent returns (negative returns) → prediction decreases
Volatility features affect sensitivity to regime shifts

➡️ The model aligns with financial intuition: momentum + volatility clustering drive short-term predictability.

📷 Example Output
Predictions (JSON list)
SHAP Summary Plot (global feature importance)
--- 

📑 Requirements
See requirements.txt
for full list.

Key dependencies:
gradio
torch
skorch
shap
pandas
matplotlib
scikit-learn
joblib
---

🧠 Model Details:

Architecture: Hybrid Heston-inspired + Linear Regression + MLP (PyTorch/Skorch)
Hyperparameter Tuning: Randomized Grid Search
Evaluation Metrics: RMSE, MAE, R², Directional Accuracy, Sharpe Ratio, Sortino Ratio, Max Drawdown
Explainability: SHAP (global + feature-level impact)

---
## ⚠️ Limitations
- Not financial advice.  
- Past performance does not guarantee future returns.  
- Accuracy varies across assets and market regimes.  

---

## 📜 License
MIT License. Free for research and educational use.

---

👨‍💻 Author

Developed by Heubert-69 — Data Scientist & ML Engineer
If you like this project, ⭐ star the repo and contribute!
