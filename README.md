# 🌟 Perovskite Bandgap Modeling with EQL & Neural Networks

This project leverages the **EQL (Equation Learner) network** to obtain symbolic mathematical expressions as encoders, which guide neural networks in modeling the relationship between **perovskite material compositions and their bandgaps**.  
In addition, we implement **SENN (Symbolically Encoded Neural Network)** and other classical machine learning methods as baselines for comparison, providing insights into both accuracy and interpretability.

---

## 📂 Project Structure


<pre>
.
├── data/                # Dataset: perovskite compositions and bandgap values
├── EQL.py               # Training script for EQL network (produces symbolic encoders)
├── models/              # SENN training code and other ML baseline models
│   ├── senn.py
│   ├── rf.py            # Random Forest
│   ├── svr.py           # Support Vector Regression
│   └── gbdt.py          # Gradient Boosting
├── requirements.txt     # Environment dependencies
└── README.md
</pre>

## ⚙️ Environment Setup

First, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Key dependencies:

numpy, pandas

scikit-learn

torch

DySymNet (for symbolic regression)

🚀 Usage
1. Train the EQL Encoder
python EQL.py

2. Train SENN and Other ML Models

Navigate into the models/ directory and run the corresponding scripts:

python models/senn.py
python models/rf.py
python models/svr.py
python models/gbdt.py

📊 Outputs

Prediction results: stored as .csv files (True vs Predicted bandgap values).

Evaluation metrics: stored as .txt files (R², RMSE, MAE, etc.).

📖 Research Significance

The EQL network generates symbolic expressions that act as encoders, helping neural networks capture underlying physical relations.

SENN and classical ML methods provide baselines to assess model performance in terms of both accuracy and interpretability.

This approach contributes to understanding the link between perovskite compositions and bandgap engineering, offering a data-driven pathway for new material design.
