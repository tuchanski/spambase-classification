# Spambase Classification

A data science project exploring email spam classification with the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase). Developed for the Computer Science degree at **PUCPR (2026-2)** by **Guilherme Tuchanski Rocha**, under **Prof. Rayson Laroca**.

The analysis lives in [`main.ipynb`](main.ipynb). It currently covers data validation, cleaning, descriptive statistics, and univariate analysis. Classification models will be developed in a later stage.

## Dataset

Spambase contains **4,601 emails**, represented by **57 numerical features**:

| Feature group | Count | Meaning |
| --- | ---: | --- |
| Word frequencies | 48 | Percentage of words in an email matching a given token. |
| Character frequencies | 6 | Percentage of characters matching a given symbol. |
| Capitalization | 3 | Average and longest uppercase run lengths, and total uppercase letters. |

The target column, `Class`, uses **0 for non-spam** and **1 for spam**. The notebook works with extracted numerical features; it does not process raw email text.

Data is downloaded through `ucimlrepo` using dataset ID `94`, so an internet connection is required when loading it.

## Current progress

- **Data cleaning:** validates numeric values, missing and infinite values, target labels, frequency ranges, and capitalization consistency.
- **Statistical description:** reports dataset dimensions, feature types, class balance, and a majority-class accuracy reference.
- **Univariate analysis:** examines 20 predictors and the target, with variable definitions, selection justifications, plots, and interpretations in English.
- **Still to do:** multivariate analysis, improved final visualizations, project reflection, and model training and evaluation.

## Analysis decisions and initial findings

The cleaning step removes **391 repeated rows**, leaving **4,210 observations**: **2,531 non-spam (60.12%)** and **1,679 spam (39.88%)**. Removing repeated rows is an analysis choice: identical extracted features do not prove that the original emails were identical. Three groups with identical predictors but conflicting labels are retained; these groups should stay together in future train/test partitions.

Zero frequencies are retained because they represent the absence of a word or character. Extreme values are also retained after validation, since a large value alone is not evidence of an error.

The univariate section covers commercial vocabulary, reader and contextual vocabulary, punctuation, and capitalization. It reports means, sample standard deviations, quartiles, ranges, zero percentages, skewness, and **excess kurtosis**, whose Gaussian reference is zero.

All 20 selected predictors are right-skewed on their original scales, and many word frequencies have zero medians. Histograms use **`log1p(value)`** to compress long tails while preserving zeros and all observations. This affects only the visualization; statistics are calculated on the original values, and the cleaned data remains unchanged.

Always predicting non-spam would achieve **60.12% accuracy on the cleaned dataset**. This is a descriptive baseline, not a trained model result. The current analysis does not establish which features predict spam; that requires comparisons with the target and subsequent model evaluation.

## Run the notebook

### Google Colab

[Open the notebook in Google Colab](https://colab.research.google.com/github/tuchanski/spambase-classification/blob/main/main.ipynb) and run the cells from top to bottom. The notebook includes a cell to install `ucimlrepo`.

### Locally

From the repository directory, create and activate a Python 3 virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate` instead.

Install the packages used by the current notebook and launch JupyterLab:

```bash
python -m pip install pandas numpy matplotlib ucimlrepo jupyterlab
python -m jupyterlab main.ipynb
```

Run the cells in order so later sections use the cleaned `df`. Dependencies are currently unpinned. Before exporting or submitting the report, restart the kernel and run all cells to refresh the saved outputs.

## Dataset reference

Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase*. UCI Machine Learning Repository. [https://doi.org/10.24432/C53G6X](https://doi.org/10.24432/C53G6X).
