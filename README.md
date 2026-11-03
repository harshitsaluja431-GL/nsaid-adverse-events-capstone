# Classifying and Understanding Serious Adverse Reactions to NSAIDs in Dogs and Cats

**QM 640 Data Analytics Capstone, Walsh College**
Author: Harshit Saluja · Mentor: Pralhad Teggi

This project analyzes U.S. FDA adverse event reports for nonsteroidal anti-inflammatory drugs (NSAIDs), the painkillers most commonly prescribed to dogs and cats. It identifies which report characteristics are associated with a report being classified as **serious**, and compares machine learning models that classify seriousness among submitted reports.

> **Important:** the data contains only animals that had a suspected reaction. Results describe patterns among reported cases. They do not estimate how often reactions occur in all treated animals, and they do not prove that a drug caused a reaction.

---

## Repository contents

| File | What it is |
|------|------------|
| `Capstone_Analysis_Complete.ipynb` | **All analysis code**, in order: download, cleaning, EDA, RQ1–RQ4, model comparison, DeLong test, SHAP, assumption checks, sensitivity analysis, and the final-report additions (SHAP direction, spline comparison, adjustment for newer reports, confusion matrix) |
| `data/nsaid_reports_clean.csv` | The cleaned dataset: 29,219 reports, one row per report |
| `data/cleaning_log.csv` | Number of reports remaining after each cleaning step |
| `figures/` | Charts used in the reports |
| `results/` | Result tables (model comparison, DeLong test, future-report comparison, SHAP importance and direction, adjustment for newer reports) |

---

## Research questions

| RQ | Question | Method |
|----|----------|--------|
| RQ1 | Among reported adverse events involving meloxicam, is the proportion classified as serious significantly different between cats and dogs? | Chi-square test |
| RQ2 | Among reported adverse events in dogs, does the proportion of serious outcomes differ between grapiprant and carprofen? | Chi-square test |
| RQ3 | How are age and body weight associated with the odds of a serious report, after accounting for species and NSAID? | Multiple logistic regression |
| RQ4 | Can a machine learning model classify serious reports better than a logistic regression baseline? | Logistic regression vs. random forest vs. XGBoost; DeLong test |

---

## Data source

- **Dataset:** openFDA Animal & Veterinary Adverse Event Reports, U.S. FDA Center for Veterinary Medicine
- **Documentation:** https://open.fda.gov/apis/animalandveterinary/event/
- **Scope:** dog and cat reports received 2015–2025 listing exactly one of six NSAIDs (carprofen, meloxicam, grapiprant, deracoxib, firocoxib, robenacoxib) and describing a single animal
- **Target variable:** `serious` (FDA regulatory definition of a serious adverse drug experience, 21 C.F.R. § 514.3)

The data is public and contains no personal information. No Kaggle or synthetic data is used.

---

## How to run

1. Open `Capstone_Analysis_Complete.ipynb` in Google Colab (**File → Upload notebook**).
2. **To reproduce the reported results exactly** (recommended): click the cell **"Optional: reproduce the reported results from the saved cleaned data"**, run it, then choose **Runtime → Run after**. This loads `data/nsaid_reports_clean.csv` from this repository instead of downloading new data.
3. **To rebuild everything from the FDA source:** choose **Runtime → Run all**. The FDA adds new reports over time, so a fresh download may differ slightly from the reported results.

All random steps use a fixed seed (42), and the data is sorted by report ID before the train/test split. Running all cells takes about 5–10 minutes on a standard Colab CPU.

**Packages:** pandas, numpy, scipy, statsmodels, scikit-learn, xgboost, shap, matplotlib (all available in Google Colab).

---

## Project status

| Stage | Submission | Status |
|-------|------------|--------|
| Topic, data verification, synopsis | Synopsis (Oct 13) | Complete |
| Data cleaning, EDA, RQ1–RQ4 analysis | Interim report (Oct 20) | Complete |
| SHAP interpretation, model adjustment for newer reports, recommendations | Final report draft (Nov 10) | Complete |
| Revisions after mentor feedback | Final report (Dec 13) | Pending |

---

## Data citation

U.S. Food and Drug Administration. (2026). *openFDA animal and veterinary adverse event reports* [Data set]. https://open.fda.gov/apis/animalandveterinary/event/
