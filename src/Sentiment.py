#!/usr/bin/env python
import re, random, math, collections, itertools

from utils import eval, calculate_valence, predict_sentiment

PRINT_ERRORS=0

random.seed(80956)

#------------- Function Definitions ---------------------

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

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
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
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

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

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

# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;

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

# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

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

# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;

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
def mostUseful(pWordPos, pWordNeg, pWord, n):
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
def countDictionaryWords(words, sentimentDictionary, Label=""):

    matched_words = [w for w in words if w in sentimentDictionary]
    print(f"{Label} words matched {len(matched_words)} out of {len(words)} words ({(len(matched_words)/len(words))*100:.0f}%)")
    print(matched_words)


# Implement rules to dictionary classifier
def testImprovedDict(sentencesTest, dataName, sentimentDictionary, threshold):

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


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



# run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, non Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, non Rule-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia   (All Data, non Rule-Based)\t",  sentimentDictionary, 1)

# run improved sentiment dictionary based classifier on datasets
testImprovedDict(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1.0)
testImprovedDict(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1.0)
testImprovedDict(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1.0)

# print most useful words
neg_useful, pos_useful = mostUseful(pWordPos, pWordNeg, pWord, 100)

print()
countDictionaryWords(neg_useful, sentimentDictionary, Label="Negative")
print()
countDictionaryWords(pos_useful, sentimentDictionary, Label="Positive")