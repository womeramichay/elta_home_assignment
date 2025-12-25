This project implements an end-to-end machine learning pipeline for predicting Titanic passenger survival using PyTorch.  
It demonstrates a clean separation between **model training** and **model inference**, along with a simple interactive web interface.
## Dataset

- The project is based on the Kaggle Titanic dataset.
- The full Kaggle dataset is not committed to the repository.
- A small sample dataset is provided in `data/sample.csv` for demonstration and review purposes.
- The training script can download the full dataset from Kaggle if a valid `kaggle.json` is configured locally.
- 
### Prerequisites
- Python 3.9+
- pip
- (Optional) Kaggle account and API key if you want to re-download the full dataset
- Linux

Clone the repository:

```bash
git clone https://github.com/womeramichay/elta_home_assignment.git
cd elta_home_assignment
```
<img width="768" height="129" alt="image" src="https://github.com/user-attachments/assets/afed37e6-0ace-4637-b094-ad66463001fd" />

---

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```
---


## Kaggle Credentials (Required for Training) 
The training script (`train.py`) downloads the Titanic dataset directly from Kaggle.
To enable this, you must provide a Kaggle API key (`kaggle.json`).

### How to set up Kaggle credentials

1. Log in to Kaggle and generate an API token:
   - Go to **Kaggle → Account → Settings → Create Legacy API Key**
   - This will download a file called `kaggle.json`
<img width="1022" height="1402" alt="image" src="https://github.com/user-attachments/assets/f94e024f-9dd1-47fd-a156-46d2fcea22d6" />



2. Place the file in the correct location depending on your operating system:
 Run the following commands from the directory where `kaggle.json` was downloaded:


```bash
cp [the path to the download dir]/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json

```



3. Verify that Kaggle is configured correctly:
```bash
kaggle competitions list
```

If the command prints a list of competitions, Kaggle is set up correctly.

### Notes
- The `kaggle.json` file contains private credentials and **must not be committed to Git**.
- If Kaggle credentials are not available, training will fail with a clear error message.
- Inference using the Streamlit app does **not** require Kaggle access.


Once credentials are set up, you can train the model by running:
### Train the Model
Run the training pipeline:

```bash
python3 train.py
```

This will:
- Load and preprocess the Titanic data
- Train a PyTorch neural network with early stopping and grid search
- Evaluate performance on a held-out test set
- Save trained artifacts to the `artifacts/` directory:
  - Trained model
  - Preprocessing pipeline
  - Evaluation metrics

---

### Run the Streamlit App

Start the inference and evaluation app:

```bash
python3 -m streamlit run ds_app.py
```
enter your email adress and he app will open in your browser (usually at `http://localhost:8501`).



## Architecture & Design Choices (Training)

- The data is split into **train / validation / test** with a fixed random seed.  
- The **validation set** is used for model selection and **early stopping**, while the **test set** is used only once for final evaluation.
- Preprocessing is **fit on the training data only** to avoid leakage:
  - Numeric features: imputation + scaling
  - Categorical features: **one-hot encoding**
- A PyTorch **MLP (feed-forward neural network)** is used with:
  - Batch normalization for stable training
  - Dropout and weight decay for regularization
- A **small grid search** is run over a few hyperparameters, optimizing **F1 score**.
- The best model and preprocessing pipeline are saved and reused for inference to ensure reproducibility.







---

## Architecture & Design Choices

- Clear separation of concerns:
  - `train.py` handles data loading, preprocessing, training, evaluation, and artifact creation
  - `ds_app.py` handles inference and visualization only
- **Model persistence**:
  - Preprocessing and model weights are saved and reused without retraining
- **PyTorch model**:
  - Chosen to demonstrate flexibility and production-style workflows
- **Streamlit UI**:
  - Lightweight interface for demonstrating inference and evaluation
- **Reproducibility**:
  - Fixed random seeds and deterministic splits
- **Minimal committed data**:
  - Only a small sample dataset is tracked to keep the repository lightweight

This structure mirrors a realistic machine learning workflow and is designed to be easy to review, run, and extend.


