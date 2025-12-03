# Sentiment Analysis – COM6115 Text Processing

A coursework project investigating Bayesian and rule-based sentiment classification

## Overview

This project implements sentiment analysis techniques to classify text on two datasets (Rotten Tomatoes movie reviews and Nokia phone reviews).  

All code is written in Python 3 and is fully reproducible with the provided datasets and lexicons.

## Features

- Naïve Bayes classifier
- Dictionary-based classifier
- Rule-Implemented Dictionary classifier (VADER)
- Error analysis

## Running the Project

```bash
# Clone the repository
git clone https://github.com/TanukiDong/Sentiment-Analysis.git
cd SentimentAnalysis

# Install dependencies
uv sync

. .venv/bin/activate

# Run Sentiment Analysis
python src/Sentiment.py

# Run Error Analysis
python src/ErrorAnalysis.py
```

## Structure

```bash
Sentiment-Analysis/
├── README.md
├── data
│   ├── negative-words.txt        # Negative lexicon (Hu & Liu, 2004)
│   ├── nokia-neg.txt             # Nokia negative reviews
│   ├── nokia-pos.txt             # Nokia positive reviews
│   ├── positive-words.txt        # Positive lexicon (Hu & Liu, 2004)
│   ├── rt-polarity.neg           # Rotten Tomatoes negative reviews
│   └── rt-polarity.pos           # Rotten Tomatoes positive reviews
│
├── error                         # Misclassification logs for analysis
│   ├── common_errors.txt
│   ├── nb_fn.txt                 # Naïve Bayes false negatives
│   ├── nb_fp.txt                 # Naïve Bayes false positives
│   ├── nb_only_errors.txt
│   ├── rd_fn.txt                 # Rule-based false negatives
│   ├── rd_fp.txt                 # Rule-based false positives
│   └── rd_only_errors.txt
│
├── pyproject.toml                # Project metadata & dependencies
├── uv.lock                       # UV environment lockfile
│
├── src
│   ├── ErrorAnalysis.py          # Helper scripts for Step 6 analysis
│   ├── Sentiment.py              # Main implementation (NB, dictionary, improved rules)
│   └── utils.py                  # Evaluation metrics + rule-based engine

```