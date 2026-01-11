# Unified-Cyber-Threat-Detection

Official implementation of the paper: **"A Unified Framework for Heterogeneous Cyber Threat Detection: Integrating Text and Tabular Data via Semantic Serialization"**.

This repository implements a unified framework processing text (e.g., emails) and tabular data (e.g., logs) using a single RoBERTa model fine-tuned with Low-Rank Adaptation (LoRA).

## Project Structure
* `1_preprocessing/`: Generates splits for T1-T3 (Stratified) and T4 (Time-Stratified). **(Table 1)**
* `2_serialization/`: Converts features for T2-T6 into semantic strings. **(Table 2)**
* `3_main_training/`: Main training loop and prompt configurations. **(Table 3, 4; Table A.7)**
* `4_evaluation/`: Evaluation scripts for performance metrics. **(Table 3, 4, 6; Figure 1)**
* `analysis/`: SHAP interpretability scripts and core calculations. **(Table 5; Figure 2, 3)**
* `baselines/`: Implementation of TF-IDF and XGBoost baselines. **(Table 3, 6)**

## Environment Setup

Choose the installation method that suits your workflow:

**1. Strict Reproduction (Linux)**
Use `environment_linux.yaml` to replicate the exact package builds and system libraries used in the paper.
```bash
conda env create -f environment_linux.yaml
conda activate unified-detection
```

**2. Cross-Platform (Conda)**
Use `environment.yml` for a standard installation. This creates a compatible environment on Linux, Windows, or macOS.
```bash
conda env create -f environment.yml
conda activate unified-detection
```

**3. Standard Python (Pip)**
Use `requirements.txt` for standard virtual environments (e.g., venv). Includes CUDA 12.1 support.
```bash
pip install -r requirements.txt
```

## Datasets

Download datasets from their official sources:

* **Phishing Email (T1):** [Kaggle - Phishing Email Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)
* **Malicious URLs (T2, T3):** [Kaggle - Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
* **Credit Card Fraud (T4):** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **UNSW-NB15 (T5, T6):** [Kaggle - UNSW-NB15](https://www.kaggle.com/datasets/dhoogla/unswnb15)

## Usage

Execute sequentially (configure file paths first):

1. **Preprocessing**: Run notebooks in `1_preprocessing/` to generate splits.
2. **Serialization**: Run `2_serialization/serialization.ipynb` to convert tabular data to text.
3. **Training**: Run `3_main_training/main_training.py` to fine-tune.
4. **Evaluation**: Run `4_evaluation/evaluation.py` for test metrics.

## Performance Summary (AUPRC)

| Task | Dataset | Baseline | Proposed (Unified) |
| --- | --- | --- | --- |
| **T1** | Phishing Email (Text) | 0.9909 | **0.9941** |
| **T2** | Malicious URL (Text) | 0.9913 | **0.9989** |
| **T3** | Malicious URL Multi (Text) | 0.9764 | **0.9956** |
| **T4** | Credit Card Fraud (Tabular) | 0.7918 | 0.7917 |
| **T5** | UNSW-NB15 (Tabular) | 0.9853 | 0.9848 |
| **T6** | UNSW-NB15 Multi (Tabular) | 0.5483 | **0.5910** |

> **Note:** The reported results represent the mean AUPRC across three independent runs.

## Citation

If you use this code, please cite the following paper:

```bibtex
@misc{Huang2026Unified,
  title={A Unified Framework for Heterogeneous Cyber Threat Detection: Integrating Text and Tabular Data via Semantic Serialization},
  author={Huang, Yen-Chin and Wang, Chih-Hung and Fan, Chun-I},
  year={2026},
  howpublished={\url{https://github.com/u04fup/Unified-Cyber-Threat-Detection}},
  note={Preprint}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.