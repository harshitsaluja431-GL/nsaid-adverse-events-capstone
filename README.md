# Serious Adverse Reactions to NSAIDs in Dogs and Cats

**QM 640 Data Analytics Capstone, Walsh College**
Author: Harshit Saluja

This project analyzes U.S. FDA adverse event reports for nonsteroidal anti-inflammatory drugs (NSAIDs), the painkillers most commonly prescribed to dogs and cats. It identifies which report characteristics are associated with a report being classified as **serious**, and builds an explainable machine learning model that predicts seriousness for new reports.

> **Important:** the data contains only animals that had a suspected reaction. Results describe patterns among reported cases. They do not estimate how often reactions occur in all treated animals, and they do not prove that a drug caused a reaction.

---

## Research questions

| RQ | Question | Method |
|----|----------|--------|
| RQ1 | Among reported adverse events involving meloxicam, is the proportion classified as serious significantly different between cats and dogs? | Chi-square test |
| RQ2 | Among reported adverse events in dogs, does the proportion of serious outcomes differ between grapiprant and carprofen? | Chi-square test |
| RQ3 | How are age and body weight associated with the odds of a serious report, after accounting for species and NSAID? | Multiple logistic regression |
| RQ4 | Can a machine learning model predict serious reports better than a logistic regression baseline? | Logistic regression vs. random forest vs. XGBoost (ROC-AUC, DeLong test) |

All tests use α = 0.05. The minimum required sample size is **5,250 reports** (set by RQ4); about **31,700** eligible reports are available, so all of them are used.

---

## Data source

- **Dataset:** openFDA Animal & Veterinary Adverse Event Reports, published by the U.S. FDA Center for Veterinary Medicine
- **Documentation:** https://open.fda.gov/apis/animalandveterinary/event/
- **API endpoint:** https://api.fda.gov/animalandveterinary/event.json
- **Scope:** dog and cat reports received 2015–2025 listing exactly one of six NSAIDs: carprofen, meloxicam, grapiprant, deracoxib, firocoxib, robenacoxib
- **Target variable:** `serious_ae` (FDA regulatory definition of a serious adverse drug experience, 21 C.F.R. § 514.3)

The data is public, free, and contains no personal information about owners or reporters. It is downloaded directly from the FDA; no Kaggle or synthetic data is used.

Raw data files are **not stored in this repository** because of their size. The scripts download them directly from the FDA, so every result can be reproduced from the original source.

---

## Repository structure
*Planned structure; files are added as each stage is completed.*
```
nsaid-adverse-events-capstone/
├── README.md                        This file
├── requirements.txt                 Pinned Python package versions 
├── .gitignore                       Excludes raw data and temporary files
├── data/
│   ├── raw/                         Downloaded openFDA JSON (not committed)
│   └── processed/                   Cleaned analysis table (CSV)
├── src/
│   ├── verify_fda_data.py           Report counts and field completeness checks
│   └── download_and_clean.py        Download, flatten, filter, and convert units
├── notebooks/
│   ├── 01_eda.ipynb                 Exploratory data analysis
│   ├── 02_hypothesis_tests.ipynb    RQ1–RQ3 tests and effect sizes
│   └── 03_models.ipynb              RQ4 models, evaluation, and SHAP
└── reports/
    └── figures/                     Charts used in the reports
```

---

## How to run

**Requirements:** Python 3.11 and an internet connection.

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshitsaluja431-GL/nsaid-adverse-events-capstone.git
   cd nsaid-adverse-events-capstone
   ```

2.  **Create a virtual environment and install packages** *(requirements file added in the interim stage)*
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify the data** *(script added in the interim stage)* (report counts, serious proportions, completeness, and minimum sample sizes)
   ```bash
   python src/verify_fda_data.py
   ```
   This takes about a minute and saves `verification_summary.csv` and `sample_records.json`.

4. **Download and clean the full dataset** *(available from the interim stage)*
   ```bash
   python src/download_and_clean.py
   ```

5. **Run the notebooks in order** *(added in the interim and final stages)*: `01_eda.ipynb` → `02_hypothesis_tests.ipynb` → `03_models.ipynb`.

All random processes use a fixed seed (`42`) so results are reproducible.

**Optional:** a free openFDA API key raises the daily request limit. Request one at https://open.fda.gov/apis/authentication/ and paste it into `API_KEY` at the top of the scripts.

---

## Project status

| Stage | Submission | Status |
|-------|------------|--------|
| Topic, data verification, synopsis | Synopsis (Oct 13) | Complete |
| Data download, cleaning, EDA | Interim draft (Oct 20) | In progress |
| RQ1–RQ3 hypothesis tests, baseline model | Interim report (Oct 27) | Planned |
| RQ4 models, DeLong test, SHAP | Final report draft (Nov 3) | Planned |
| Presentation | Nov 29 | Planned |
| Final report | Dec 13 | Planned |

This README is updated as each stage is completed.



## Data citation

U.S. Food and Drug Administration. (2026). *openFDA animal and veterinary adverse event reports* [Data set]. https://open.fda.gov/apis/animalandveterinary/event/
