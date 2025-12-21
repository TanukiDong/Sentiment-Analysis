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

<style>
  table {
    width: 75%;
    border-collapse: collapse;
  }

  th, td {
    text-align: center;
    border: 1px solid black;
    padding: 5px;
  }
</style>

Top 5 positive and negative words selected by Naïve Bayes.

<table>
  <thead>
    <tr>
      <th></th>
      <th align="center">Positive</th>
      <th align="center">Negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td align="center">½</td>
      <td align="center">generic</td>
    </tr>
    <tr>
      <td>2</td>
      <td align="center">ï</td>
      <td align="center">unfunny</td>
    </tr>
    <tr>
      <td>3</td>
      <td align="center">engrossing</td>
      <td align="center">waste</td>
    </tr>
    <tr>
      <td>4</td>
      <td align="center">inventive</td>
      <td align="center">mediocre</td>
    </tr>
    <tr>
      <td>5</td>
      <td align="center">riveting</td>
      <td align="center">routine</td>
    </tr>
  </tbody>
</table>

Top sentiment words that are found and not found in the sentiment dictionary

<table>
  <thead>
    <tr>
      <th colspan="2">Word Found in Dictionary</th>
      <th colspan="2">Word not Found in Dictionary</th>
    </tr>
    <tr>
      <th>Positive</th>
      <th>Negative</th>
      <th>Positive</th>
      <th>Negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mesmerizing</td>
      <td>Boring</td>
      <td>Touching</td>
      <td>Harvard</td>
    </tr>
    <tr>
      <td>Captivating</td>
      <td>Offensive</td>
      <td>Transcends</td>
      <td>Clichés</td>
    </tr>
    <tr>
      <td>Masterpiece</td>
      <td>Lifeless</td>
      <td>Martha</td>
      <td>Tuxedo</td>
    </tr>
    <tr>
      <td>Breathtaking</td>
      <td>Lame</td>
      <td>Answers</td>
      <td>Arts</td>
    </tr>
  </tbody>
</table>





The error distribution across the Naïve Bayes and Dictionary-based classifiers

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="center">Naïve Bayes</th>
      <th align="center">Dictionary-based</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Total Errors</b></td>
      <td align="center">234</td>
      <td align="center">358</td>
    </tr>
    <tr>
      <td><b>Unique Errors</b></td>
      <td align="center">134</td>
      <td align="center">258</td>
    </tr>
    <tr>
      <td><b>Common Errors</b></td>
      <td colspan="2" align="center">100</td>
    </tr>
  </tbody>
</table>

## Sources

This project use te lexicons from:
- Hu & Liu (2004) Opinion Lexicon  
Provided in ```positive-words.txt``` and ```negative-words.txt```