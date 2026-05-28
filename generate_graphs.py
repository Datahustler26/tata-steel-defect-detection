import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix

sns.set(style='whitegrid')

MODEL_PATH = 'model_lgbm.joblib'
THR_PATH = 'model_threshold.json'
TRAIN_PATH = 'dataset/train.csv'

model = joblib.load(MODEL_PATH)
with open(THR_PATH) as f:
    thr = json.load(f).get('threshold', 0.0)

# Load train
df = pd.read_csv(TRAIN_PATH)
if 'Y' not in df.columns:
    raise SystemExit('No Y in train.csv')
y = df['Y'].values
X = df.drop(columns=['Y'], errors='ignore')
if 'CoilID' in X.columns:
    X = X.drop(columns=['CoilID'])

# Get probabilities
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)[:,1]
else:
    # fallback to decision_function or predict
    try:
        probs = model.decision_function(X)
        probs = (probs - probs.min()) / (probs.max() - probs.min())
    except Exception:
        preds = model.predict(X)
        probs = preds.astype(float)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y, probs)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, color='#1f77b4', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (train)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/pr_curve.png', dpi=150)
plt.close()

# Confusion matrix at saved threshold
preds_thr = (probs >= thr).astype(int)
cm = confusion_matrix(y, preds_thr)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (threshold={thr})')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150)
plt.close()

print('Saved images: images/pr_curve.png, images/confusion_matrix.png')
