# Sentiment Analysis – COM6115 Text Processing

A coursework project investigating Bayesian and rule-based sentiment classification

## Overview

This project implements sentiment analysis techniques to classify text on two datasets (Rotten Tomatoes movie reviews and Nokia phone reviews).  

All code is written in Python 3 and is fully reproducible with the provided datasets and lexicons.

## Classifier

- Naïve Bayes classifier
- Dictionary-based classifier (baseline)
- Improved rule-based dictionary classifier (VADER-inspired)

## Dataset

- Rotten Tomatoes Movies Reviews
- Nokia Phone Reviews

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

### Top 5 positive and negative words selected by Naïve Bayes.

|  #  | Positive  | Negative  |
|:---:|:---------:|:---------:|
|  1  | ½         | generic   |
|  2  | ï         | unfunny   |
|  3  | engrossing| waste     |
|  4  | inventive | mediocre  |
|  5  | riveting  | routine   |

Due to the nature of the Naïve Bayes algorithm, it does not understand the context and semantic meaning of the text, so some tokens such as “½” and ï are detected as sentiment words.


### Top sentiment words that are found and not found in the sentiment dictionary

| In Dictionary (Positive) | In Dictionary (Negative) | Not in Dictionary (Positive) | Not in Dictionary (Negative) |
|:------------------------:|:------------------------:|:----------------------------:|:----------------------------:|
| Mesmerizing              | Boring                   | Touching                     | Harvard                      |
| Captivating              | Offensive                | Transcends                   | Clichés                      |
| Masterpiece              | Lifeless                 | Martha                       | Tuxedo                       |
| Breathtaking             | Lame                     | Answers                      | Arts                         |

Some predictive words are missing from the sentiment lexicon. However, many of these words do not carry sentiment meaning, such as proper nouns or domain-specific terms.

### Comparison between Naïve Bayes, simple dictionary, and improved dictionary

| Dataset | Classifier | Accuracy | Macro F1 |
|:------:|:----------:|:--------:|:--------:|
| Rotten Tomatoes | Naïve Bayes | 0.78 | 0.78 |
| Rotten Tomatoes | Dictionary-based | 0.63 | 0.63 |
| Rotten Tomatoes | Rule-based Dictionary | **0.66** | **0.67** |
| Nokia Reviews | Naïve Bayes | 0.57 | 0.55 |
| Nokia Reviews | Dictionary-based | 0.80 | 0.77 |
| Nokia Reviews | Rule-based Dictionary | **0.82** | **0.81** |

Naïve Bayes performs better on movie reviews but fails on phone reviews, as it is trained on the movie dataset. Dictionary-based approaches perform worse on movie reviews but generalise better to other domains, as seen in the Nokia product reviews.

## Sources

This project use the lexicons from:
- Hu & Liu (2004) Opinion Lexicon  
Provided in ```positive-words.txt``` and ```negative-words.txt```