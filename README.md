# multilayer-perceptron

42 project: implement a multilayer perceptron from scratch to classify breast
cancer tumors (malignant/benign) from the Wisconsin Diagnostic Breast Cancer
(WDBC) dataset. See `en.subject.pdf` for the full subject.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Clean the raw dataset (adds headers, sanity-checks values, writes
`data/data_clean.csv`):

```sh
python scripts/clean_data.py [data/data.csv] [data/data_clean.csv]
```

Explore the cleaned dataset (prints a summary, saves graphs to `outputs/`):

```sh
python scripts/explore_data.py [data/data_clean.csv] [--outdir outputs/]
```
