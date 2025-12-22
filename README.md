# Sentiment Analysis – COM6115 Text Processing

A coursework project investigating Bayesian and rule-based sentiment classification

## Overview

This project implements sentiment analysis techniques to classify text on two datasets (Rotten Tomatoes movie reviews and Nokia phone reviews).  

All code is written in Python 3 and is fully reproducible with the provided datasets and lexicons.

## Classifier

- Naïve Bayes classifier
- Dictionary-based classifier
- Rule-Implemented Dictionary classifier (VADER)

## Dataset

- Rotton Tomatoes Movies Review
- Nokia Phone Review

## Running the Project

```bash
# Clone the repository
git clone https://github.com/TanukiDong/Sentiment-Analysis.git
cd Sentiment-Analysis

# Install dependencies
uv sync

# Run Sentiment Analysis
uv run src/Sentiment.py

# Run Error Analysis
uv run src/ErrorAnalysis.py
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
│   ├── nb_only_errors.txt        # Combination of nb_fn.txt and nb_fp.txt   
│   ├── rd_fn.txt                 # Rule-based false negatives
│   ├── rd_fp.txt                 # Rule-based false positives
│   └── rd_only_errors.txt        # Combination of rd_fn.txt and rd_fp.txt
│
├── pyproject.toml                # Project metadata & dependencies
├── uv.lock                       # UV environment lockfile
│
├── src
│   ├── ErrorAnalysis.py          # Helper scripts for Step 6 analysis
│   ├── Sentiment.py              # Main implementation (NB, dictionary, improved rules)
│   └── utils.py                  # Evaluation metrics + rule-based engine

```

## Results

Top 5 positive and negative words selected by Naïve Bayes.
|  #  | Positive  | Negative  |
|:---:|:---------:|:---------:|
|  1  | ½         | generic   |
|  2  | ï         | unfunny   |
|  3  | engrossing| waste     |
|  4  | inventive | mediocre  |
|  5  | riveting  | routine   |

Top sentiment words that are found and not found in the sentiment dictionary

| In Dictionary (Positive) | In Dictionary (Negative) | Not in Dictionary (Positive) | Not in Dictionary (Negative) |
|:------------------------:|:------------------------:|:----------------------------:|:----------------------------:|
| Mesmerizing              | Boring                   | Touching                     | Harvard                      |
| Captivating              | Offensive                | Transcends                   | Clichés                      |
| Masterpiece              | Lifeless                 | Martha                       | Tuxedo                       |
| Breathtaking             | Lame                     | Answers                      | Arts                         |

The error distribution across the Naïve Bayes and Dictionary-based classifiers. There are 100 common errors.

| Metric         | Naïve Bayes | Dictionary-based |
|:--------------:|:-----------:|:----------------:|
| Total Errors   | 234         | 358              |
| Unique Errors  | 134         | 258              |



## Sources

This project use te lexicons from:
- Hu & Liu (2004) Opinion Lexicon  
Provided in ```positive-words.txt``` and ```negative-words.txt```