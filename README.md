# ConfiMed: Multilingual Healthcare Confidentiality Benchmark

**ConfiMed** is a parallel English-Arabic benchmark corpus for fine-grained confidentiality classification in healthcare administration. This repository contains the dataset, annotation guidelines, and the official code for reproducing the deep learning baselines using **XLM-RoBERTa (XLM-R)**.

## 📂 Repository Structure

```
ConfiMed/
├── annotation_tool_and_guidlines/   # PDF guidelines for annotators
├── final_corpus/            # The JSON dataset (FINAL_CORPUS_CONFIMED.json)
├── xlmr_code/               # Python scripts for XLM-R experiments
│   ├── run_xlmr_english_v2.py
│   ├── run_xlmr_arabic_v2.py
│   ├── run_xlmr_zeroshot_v2.py
│   └── run_xlmr_joint_v2.py
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   A GPU is highly recommended (scripts automatically detect CUDA/MPS).

### Setup Environment
1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    The scripts require `transformers`, `torch`, and specifically `accelerate >= 1.1.0` to work with the Hugging Face Trainer.

    ```bash
    pip install -r requirements.txt
    ```

    *If you encounter an `ImportError` related to `accelerate`, run:*
    ```bash
    pip install "accelerate>=1.1.0"
    pip install -U transformers datasets evaluate
    ```

## 🚀 Usage

All scripts are located in the `xlmr_code/` directory. They use **5-Fold Stratified Cross-Validation** and report Accuracy, Macro-F1, and Quadratic Weighted Kappa (QWK).

### 1. English Monolingual Baseline
Trains and evaluates on English data only.
```bash
python xlmr_code/run_xlmr_english_v2.py --input_file final_corpus/FINAL_CORPUS_CONFIMED.json
```

### 2. Arabic Monolingual Baseline
Trains and evaluates on Arabic data only.
```bash
python xlmr_code/run_xlmr_arabic_v2.py --input_file final_corpus/FINAL_CORPUS_CONFIMED.json
```

### 3. Cross-Lingual Zero-Shot Transfer
Trains on one language (Source) and evaluates on the other (Target) without seeing any target data.

**English → Arabic (The "Zero-Shot Surprise"):**
```bash
python xlmr_code/run_xlmr_zeroshot_v2.py \
    --input_file final_corpus/FINAL_CORPUS_CONFIMED.json \
    --train_lang en \
    --test_lang ar
```

**Arabic → English:**
```bash
python xlmr_code/run_xlmr_zeroshot_v2.py \
    --input_file final_corpus/FINAL_CORPUS_CONFIMED.json \
    --train_lang ar \
    --test_lang en
```

### 4. Joint Multilingual Training
Trains on combined English + Arabic data (shuffled) and evaluates on the combined test set.
```bash
python xlmr_code/run_xlmr_joint_v2.py --input_file final_corpus/FINAL_CORPUS_CONFIMED.json
```

## ⚙️ Configuration & Defaults

The scripts use the following optimized hyperparameters (tuned for stability on small datasets):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `--lr` | `2e-5` | Learning rate (lower prevents collapse) |
| `--batch_size` | `8` | Per-device batch size |
| `--epochs` | `5` | Max training epochs |
| `--max_length` | `256` | Max token sequence length |
| `--patience` | `3` | Early stopping patience |
| `--seed` | `42` | Random seed for reproducibility |

**Output:**
Results are saved in `results_xlmr_[experiment_name]_v2/` directories.
*   `final_results.json`: Contains the average metrics across all 5 folds.
*   `results_fold_X.json`: Detailed metrics for each specific fold.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
