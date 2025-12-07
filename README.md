

# Urdu Deepfake Audio Detection & Multi-Label Software Defect Prediction


## 🔬 Overview / Objectives

**Datasets**

* **Part 1 (Audio):** `CSALT/deepfake detection dataset urdu` (loadable with `datasets.load_dataset`).
* **Part 2 (Tabular):** CSV file attached with the assignment (multi-label targets).
* **Part 3:** Integrates models from Part 1 & 2 into a user-facing Streamlit app.

**Tasks**

* Preprocess audio and tabular data.
* Extract MFCC, Mel Spectrogram, Chroma; pad/truncate audio to fixed length.
* Train and compare four models for each task: SVM, Logistic Regression, Single-Layer Perceptron (online), and DNN (≥2 hidden layers).
* Evaluate with the required metrics and compare models on the same test set.
* Build and deploy an interactive Streamlit app for predictions.

---

## 📂 Repository Structure (recommended)

```
urdu-deepfake-audio-detection-and-defect-prediction/
│
├── part1/                          # Part 1: Urdu Deepfake
│   ├── Part1_Urdu_Deepfake.ipynb   # Colab/Jupyter notebook (full pipeline)
│   ├── part1_utils.py              # helper functions (audio, features)
│   ├── train_part1.py              # training script CLI
│   └── inference_part1.py          # inference examples
│
├── part2/                          # Part 2: Multi-label defect prediction
│   ├── Part2_MultiLabel_Defect.ipynb
│   ├── part2_utils.py
│   ├── train_part2.py
│   └── metrics_part2.py
│
├── app/                            # Streamlit app (Part 3)
│   └── streamlit_app.py
│
├── models/                         # saved models after training
│   ├── part1_svm.joblib
│   ├── part1_log.joblib
│   ├── part1_perc.joblib
│   ├── part1_scaler.joblib
│   ├── part1_dnn/                  # keras saved model directory
│   └── part2_*.joblib / part2_dnn/
│
├── data/                           # datasets (optional, small samples)
│   ├── dataset_part2.csv
│   └── README_DATA.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Setup (Colab / Local)

### 1) Clone repo

```bash
git clone <repo-url>
cd urdu-deepfake-audio-detection-and-defect-prediction
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
# or in colab
!pip install librosa numpy pandas scikit-learn tensorflow datasets matplotlib seaborn joblib soundfile
```

---

## 📘 Part 1 — Urdu Deepfake Audio Detection

### A. Loading dataset (Hugging Face)

```python
from datasets import load_dataset
ds = load_dataset("CSALT/deepfake detection dataset urdu")
# ds['train'], ds['test'], etc.
```

### B. Preprocessing & feature extraction

* Resample to 16kHz, fix length to 3s (pad/truncate).
* Extract: MFCC (13), Mel spectrogram (use first 40 averaged bins), Chroma (12).
* Option: produce fixed-length 1D vectors or 2D spectrogram images (for CNNs).

Snippet (helper function included in `part1/part1_utils.py`):

```python
# load_audio, extract_features as provided in notebook
```

### C. Models to train

* SVM (RBF, probability=True)
* Logistic Regression (max_iter=1000)
* Single-Layer Perceptron (sklearn.Perceptron)
* DNN with at least 2 hidden layers (Keras)

### D. Training & saving

* Use stratified train/val/test split (e.g., 70/10/20 or train_test_split with stratify).
* Save sklearn models with `joblib.dump` and Keras model with `model.save()`.

### E. Evaluation metrics

* Accuracy, Precision, Recall, F1-score, ROC-AUC
* Generate confusion matrix & ROC curves for each model
* Compare all models on the same test split; export results to CSV or table.

---

## 📘 Part 2 — Multi-Label Software Defect Prediction

### A. Preprocessing

* Handle missing values (impute mean/median or drop depending on column).
* Feature selection (correlation filter, mutual information, PCA optional).
* Scaling: `StandardScaler`.

### B. Label handling

* Use `sklearn.preprocessing.MultiLabelBinarizer()` to convert label lists to binary matrix.
* Analyze label distribution; report highly imbalanced labels.

### C. Models

* Logistic Regression — OneVsRestClassifier(LogisticRegression)
* SVM — OneVsRestClassifier(SVC, probability=True)
* Perceptron — implement online learning (weight update after each sample)

  * Use sklearn `Perceptron.partial_fit` in a loop to simulate online updates.
* DNN — Keras model with final `sigmoid` activation for multi-label outputs.

**Perceptron (online) example sketch**

```python
from sklearn.linear_model import Perceptron
p = Perceptron()
# Initialize with classes for binary labels: [0,1]
for epoch in range(n_epochs):
    for xi, yi in zip(X_train, Y_train_label):
        p.partial_fit(xi.reshape(1,-1), [yi], classes=[0,1])
```

For multi-label, train one-per-label perceptron in this online fashion.

### D. Training & Hyperparameter tuning

* Split: train / val / test
* Tune `C` for logistic/SVM, learning rate/epochs for perceptron & DNN (GridSearchCV or manual search).
* For expensive models, tune on a validation split not full CV.

### E. Evaluation

* Hamming Loss
* Micro-F1, Macro-F1
* Precision@3, Precision@5 (for each sample, sort predicted probabilities and compute precision among top-k predicted labels)
* Save all evaluation tables & per-class metrics.

**Precision@k implementation sketch**

```python
def precision_at_k(y_true, y_scores, k=3):
    # y_true: binary (n_samples, n_labels)
    # y_scores: predicted probabilities (n_samples, n_labels)
    precisions = []
    for true_row, score_row in zip(y_true, y_scores):
        topk_idx = np.argsort(score_row)[-k:][::-1]
        precisions.append(true_row[topk_idx].sum() / k)
    return np.mean(precisions)
```

---

## 🧪 Part 3 — Interactive Streamlit Application (Deployment)

**Features**

* Upload an audio file → predict bonafide vs deepfake (choose model at runtime).
* Enter/tabular upload → predict multi-label software defects (choose model).
* Show prediction probabilities/confidence, ROC curve, and explanations (optional).

**Run locally:**

```bash
streamlit run app/streamlit_app.py
```

**Deployment**

* Deploy to **Heroku**, **Streamlit Cloud**, or **Hugging Face Spaces** (for simple Streamlit apps).
* Ensure `models/` folder is available or load models from a cloud storage / HF repo.

**Important**: For user-uploaded audio, app should:

* Save the file temporarily (e.g., `/tmp/`), run `extract_features()`, scale using the saved scaler, and call the selected model.

---

## 🧾 Deliverables included in repo

* `part1/Part1_Urdu_Deepfake.ipynb` (complete notebook)
* `part2/Part2_MultiLabel_Defect.ipynb` (complete notebook)
* `app/streamlit_app.py` (user interface)
* `requirements.txt`
* Utility scripts: `part1_utils.py`, `part2_utils.py`
* Example `dataset_part2.csv` (if permitted)
* `README.md` (this file)

---

## ✅ Practical Tips & Common Fixes

* **CSV encoding errors**: If `pd.read_csv()` raises `UnicodeDecodeError`, try `encoding='latin1'` or use `chardet` to detect encoding.
* **Missing files**: Use absolute paths or mount Google Drive in Colab (`drive.mount('/content/drive')`).
* **Large audio dataset**: Use batching when extracting features to avoid memory spike.
* **Reproducibility**: set `random_state=42` for `train_test_split` and seeds for TF/NumPy.
* **GPU for DNN**: enable GPU in Colab (Runtime → Change runtime type → GPU).

---

## 📈 Example Evaluation Commands (quick)

```python
# After training and saving models
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, pred))
print("AUC:", roc_auc_score(y_test, probs))
```

For multi-label:

```python
from sklearn.metrics import hamming_loss, f1_score
print("Hamming Loss:", hamming_loss(Y_test, Y_pred))
print("Micro F1:", f1_score(Y_test, Y_pred, average='micro'))
print("Macro F1:", f1_score(Y_test, Y_pred, average='macro'))
```

---

## ✍️ Author

**Ibrahim Zahid** — AI / ML Student
(Use this name or replace with your details)

---

## 📜 License

Include a license (e.g., MIT) if you want others to reuse the code:

```
MIT License
```

