#!/usr/bin/env python
import re, random, math, collections, itertools

from utils import eval, calculate_valence, predict_sentiment

PRINT_ERRORS=0

random.seed(80956)

#------------- Function Definitions ---------------------

def readFiles(
        sentimentDictionary: dict[str, int],
        sentencesTrain: dict[str, str],
        sentencesTest: dict[str, str],
        sentencesNokia: dict[str, str]
        ) -> None:
    """
    Load sentiment lexicons and datasets, including Rotten Tomatoes movie reviews
    and Nokia phone review datasets.

    The Rotten Tomatoes dataset is randomly split into training (90%) and test (10%) sets. 
    Nokia reviews are stored separately for cross-domain evaluation.

    Parameters
    ----------
    sentimentDictionary : dict
        Dictionary mapping words to sentiment values:
        +1 for positive words and -1 for negative words.

    sentencesTrain : dict
        Dictionary mapping training sentences to sentiment labels

    sentencesTest : dict
        Dictionary mapping test sentences to sentiment labels

    sentencesNokia : dict
        Dictionary mapping Nokia review sentences to sentiment labels.

    Returns
    -------
    None
        Data is loaded into sentimentDictionary, sentencesTrain, sentencesTest, and sentencesNokia.
    """

    #reading pre-labeled input and splitting into lines
    posSentences = open('data/rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('data/rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('data/nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('data/nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    posDictionary = open('data/positive-words.txt', 'r', encoding="ISO-8859-1")
    # Omit comments
    posWordList = [line.strip() for line in posDictionary if not line.startswith(';')]

    negDictionary = open('data/negative-words.txt', 'r', encoding="ISO-8859-1")
    # Omit comments
    negWordList = [line.strip() for line in negDictionary if not line.startswith(';')]

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"
        
    # create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(
        sentencesTrain: dict[str, str],
        pWordPos: dict[str, float],
        pWordNeg: dict[str, float],
        pWord: dict[str, float]
        ) -> None:
    """
    Train a Naïve Bayes classifier using Rotten Tomatoes review training data.
    It stores the conditional probabilities of words given sentiment classes
    and the overall word probabilities in the provided dictionaries.

    Parameters
    ----------
    sentencesTrain : dict
        Dictionary mapping sentences to sentiment labels

    pWordPos : dict
        Dictionary to store probabilities of positive words P(word | positive).

    pWordNeg : dict
        Dictionary to store probabilities of negative words P(word | negative).
 
    pWord : dict
        Dictionary to store probabilities of words P(word).

    Returns
    -------
    None
        Probability dictionaries are saved in pWordPos, pWordNeg, and pWord.
    """
    freqPositive = {}
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# Implement Naïve Bayes Algorithm
def testBayes(
        sentencesTest: dict[str, str],
        dataName: str,
        pWordPos: dict[str, float],
        pWordNeg: dict[str, float],
        pWord: dict[str, float],
        pPos: float
        ) -> None:
    """
    Apply a Naïve Bayes classifier to a dataset (either training or testing) and evaluate performance.
    Calculates the posterior probabilities of sentiment classes for each sentence based on word probabilities and class priors.
    Performance metrics including accuracy, precision, recall, and F1 score are printed using the eval() function.

    Parameters
    ----------
    sentencesTest : dict
        Dictionary mapping test sentences to sentiment labels

    dataName : str
        Name of the dataset to use for display purposes.

    pWordPos : dict
        Dictionary of probability of word given it is positive P(word | positive).

    pWordNeg : dict
        Dictionary of probability of word given it is negative P(word | negative).

    pWord : dict
        Dictionary of probability of word P(word).

    pPos : float
        Fraction of positive reviews in the training dataset P(positive).

    Returns
    -------
    None
        Evaluation results are printed to stdout.
    """
    print("Naive Bayes classification")
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    # Print errors to file nb_fn.txt
                    with open('error/nb_fn.txt', 'a') as file:
                        print ("ERROR (neg classed as pos %0.2f):" %prob + sentence, file=file)

        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    # Print errors to file nb_fp.txt
                    with open('error/nb_fp.txt', 'a') as file:
                        print ("ERROR (pos classed as neg %0.2f):" %prob + sentence, file=file)

    eval(
            dataName,
            correct,
            total,
            totalpos,
            totalneg,
            correctpos,
            totalpospred,
            correctneg,
            totalnegpred,
        )

# A simple classifier that uses a sentiment dictionary to classify a sentence.
def testDictionary(
        sentencesTest: dict[str, str],
        dataName: str,
        sentimentDictionary: dict[str, int],
        threshold: float
        ) -> None:
    """
    Classify sentences using a dictionary-based sentiment classifier.

    Each sentence is scored by summing sentiment values of words found in the sentiment dictionary.
    The sentence is classified as positive if the score exceeds a given threshold.

    Parameters
    ----------
    sentencesTest : dict
        Dictionary mapping test sentences to sentiment labels

    dataName : str
        Name of the dataset to use for display purposes.

    sentimentDictionary : dict
        Dictionary mapping words to sentiment values (+1 or -1).

    threshold : float
        Decision threshold for classification.

    Returns
    -------
    None
        Evaluation results are printed to stdout.
    """
    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
                score+=sentimentDictionary[word]

        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1

        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1

    eval(
            dataName,
            correct,
            total,
            totalpos,
            totalneg,
            correctpos,
            totalpospred,
            correctneg,
            totalnegpred,
        )

# Print out n most useful predictors
def mostUseful(
        pWordPos: dict[str, float],
        pWordNeg: dict[str, float],
        pWord: dict[str, float],
        n: int
        ) -> tuple[list[str], list[str]]:
    """
    Identify the n most predictive words for sentiment classification.
    Words are ranked by their probability of appearing in positive or negative sentiment sentences.

    Parameters
    ----------
    pWordPos : dict
        Dictionary to store probabilities of positive words P(word | positive).

    pWordNeg : dict
        Dictionary to store probabilities of negative words P(word | negative).
 
    pWord : dict
        Dictionary to store probabilities of words P(word).

    n : int
        Number of words to return.

    Returns
    -------
    head : list
        List of n words most predicted to occur in a negative sentiment sentence.

    tail : list
        List of n words most predicted to occur in a positive sentiment sentence.
    """
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]

    # Sort tail in descending order
    tail.reverse()

    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)

    # Return most useful words
    return head, tail


# Count number of words in dictionary
def countDictionaryWords(
        words: list[str],
        sentimentDictionary: dict[str, int],
        label: str = ""
        ) -> None:
    """
    Count how many words in a given list appear in the sentiment dictionary.
    Use to check how many of the most useful words are present in the sentiment dictionary.

    Parameters
    ----------
    words : list of str
        List of words to be checked against the sentiment dictionary.

    sentimentDictionary : dict
        Dictionary mapping words to sentiment values (+1 or -1).

    label : str, optional
        Sentiment label used for display purposes.
        Positive or Negative. Default is an empty string.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    matched_words = [w for w in words if w in sentimentDictionary]
    print(f"{label} words matched {len(matched_words)} out of {len(words)} words ({(len(matched_words)/len(words))*100:.0f}%)")
    print(matched_words)


# Implement rules to dictionary classifier
def testImprovedDict(
        sentencesTest: dict[str, str],
        dataName: str,
        sentimentDictionary: dict[str, int],
        threshold: float
        ) -> None:
    """
    Apply an improved rule-based sentiment classifier.

    This classifier implements VADER-styled heuristic rules including:
    - Punctuation
    - Contrastive Conjunctions "but"
    - Degree Modifiers
    - Negations
    - Exclamation
    Details on these rules can be found in the calculate_valence() function in utils.py.

    Parameters
    ----------
    sentencesTest : dict
        Dictionary mapping test sentences to sentiment labels.

    dataName : str
        Name of the dataset to use for display purposes.

    sentimentDictionary : dict
        Dictionary mapping words to sentiment values (+1 or -1).

    threshold : float
        Decision threshold for classification.

    Returns
    -------
    None
        Evaluation results are printed to stdout.
    """
    # Declare classifier
    print("\033[94mImproved Dictionary-based classification\033[0m")

    # Initialize values
    total=len(sentencesTest)
    pos_actual = sum([1 for s in sentencesTest.values() if s == "positive"])
    neg_actual = total - pos_actual
    pos_pred=0
    neg_pred=0
    pos_correct=0
    neg_correct=0

    for sentence, sentiment_actual in sentencesTest.items():

        # Find all words, including hyphens and punctuations
        Words = re.findall(r"[\w'-]+|[.,!?;:]", sentence)

        # "but" in sentence
        if "but" in Words:
            # Find index of "but"
            i_b = Words.index("but")

            # Split sentence into left and right of "but"
            left = Words[:i_b]
            right = Words[i_b+1:]

            # Diminish left valence, intensify right valence
            score = calculate_valence(left, sentimentDictionary) * 0.5 + calculate_valence(right, sentimentDictionary) * 1.5
            
        # No "but" in sentence
        else:
            score = calculate_valence(Words, sentimentDictionary)

        # Predicted sentiment
        sentiment_predicted = predict_sentiment(score, threshold)

        # Predicted "positive"
        if sentiment_predicted == "positive":
            pos_pred +=1
            # Prediction is correct
            if sentiment_actual == "positive":
                pos_correct +=1
            # Prediction is incorrect → False Positive
            else:
                if PRINT_ERRORS:
                    # Print errors to file rd_fp.txt
                    with open('error/rd_fp.txt', 'a') as file:
                        print(f"ERROR (neg classed as pos {score:.2f}):{sentence}", file=file)

        # Predicted "negative"
        else:
            neg_pred +=1
            # Prediction is correct
            if sentiment_actual == "negative":
                neg_correct +=1
            # Prediction is incorrect → False Negative
            else:
                if PRINT_ERRORS:
                    # Print errors to file rd_fn.txt
                    with open('error/rd_fn.txt', 'a') as file:
                        print(f"ERROR (pos classed as neg {score:.2f}):{sentence}", file=file)

        total_correct = pos_correct + neg_correct

    eval(
            dataName,
            total_correct,
            total,
            pos_actual,
            neg_actual,
            pos_correct,
            pos_pred,
            neg_correct,
            neg_pred,
        )
    
#---------- Main Script --------------------------
if __name__ == "__main__":

    sentimentDictionary={}
    sentencesTrain={}
    sentencesTest={}
    sentencesNokia={}

    # Initialise datasets and dictionaries
    readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

    pWordPos={} # p(W|Positive)
    pWordNeg={} # p(W|Negative)
    pWord={}    # p(W) 

    # Build Conditional probabilities using training data
    trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

    # Run Naïve Bayes classifier on datasets
    testBayes(sentencesTrain,  "Films (Train Data, Naïve Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
    testBayes(sentencesTest,  "Films  (Test Data, Naïve Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
    testBayes(sentencesNokia, "Nokia   (All Data,  Naïve Bayes)\t", pWordPos, pWordNeg, pWord,0.7)

    # Run sentiment dictionary based classifier on datasets
    testDictionary(sentencesTrain,  "Films (Train Data, non Rule-Based)\t", sentimentDictionary, 1)
    testDictionary(sentencesTest,  "Films  (Test Data, non Rule-Based)\t",  sentimentDictionary, 1)
    testDictionary(sentencesNokia, "Nokia   (All Data, non Rule-Based)\t",  sentimentDictionary, 1)

    # Run improved sentiment dictionary based classifier on datasets
    testImprovedDict(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1.0)
    testImprovedDict(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1.0)
    testImprovedDict(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1.0)

    # Print most useful words
    neg_useful, pos_useful = mostUseful(pWordPos, pWordNeg, pWord, 100)

    print()
    countDictionaryWords(neg_useful, sentimentDictionary, label="Negative")
    print()
    countDictionaryWords(pos_useful, sentimentDictionary, label="Positive")