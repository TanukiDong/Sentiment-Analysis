import re

# Sentiment words and values from VADER Sentiment Analysis 
# Hutto, C.J. & Gilbert, E.E. (2014)
M_INCR = 1.333
M_DECR = 0.667

TERMINAL_PUNCTUATION = [".",";",":"] # not include , ! ?

NEGATION = [
    "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
    "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
    "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
    "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
    "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
    "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
    "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
    "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"
    ]

MODIFIER = {
    "absolutely": M_INCR, "amazingly": M_INCR, "awfully": M_INCR,
    "completely": M_INCR, "considerable": M_INCR, "considerably": M_INCR,
    "decidedly": M_INCR, "deeply": M_INCR, "effing": M_INCR, "enormous": M_INCR, "enormously": M_INCR,
    "entirely": M_INCR, "especially": M_INCR, "exceptional": M_INCR, "exceptionally": M_INCR,
    "extreme": M_INCR, "extremely": M_INCR,
    "fabulously": M_INCR, "flipping": M_INCR, "flippin": M_INCR, "frackin": M_INCR, "fracking": M_INCR,
    "fricking": M_INCR, "frickin": M_INCR, "frigging": M_INCR, "friggin": M_INCR, "fully": M_INCR,
    "fuckin": M_INCR, "fucking": M_INCR, "fuggin": M_INCR, "fugging": M_INCR,
    "greatly": M_INCR, "hella": M_INCR, "highly": M_INCR, "hugely": M_INCR,
    "incredible": M_INCR, "incredibly": M_INCR, "intensely": M_INCR,
    "major": M_INCR, "majorly": M_INCR, "more": M_INCR, "most": M_INCR, "particularly": M_INCR,
    "purely": M_INCR, "quite": M_INCR, "really": M_INCR, "remarkably": M_INCR,
    "so": M_INCR, "substantially": M_INCR,
    "thoroughly": M_INCR, "total": M_INCR, "totally": M_INCR, "tremendous": M_INCR, "tremendously": M_INCR,
    "uber": M_INCR, "unbelievably": M_INCR, "unusually": M_INCR, "utter": M_INCR, "utterly": M_INCR,
    "very": M_INCR,
    "almost": M_DECR, "barely": M_DECR, "hardly": M_DECR, "just enough": M_DECR,
    "kind of": M_DECR, "kinda": M_DECR, "kindof": M_DECR, "kind-of": M_DECR,
    "less": M_DECR, "little": M_DECR, "marginal": M_DECR, "marginally": M_DECR,
    "occasional": M_DECR, "occasionally": M_DECR, "partly": M_DECR,
    "scarce": M_DECR, "scarcely": M_DECR, "slight": M_DECR, "slightly": M_DECR, "somewhat": M_DECR,
    "sort of": M_DECR, "sorta": M_DECR, "sortof": M_DECR, "sort-of": M_DECR
    }

# Evaluate classifier
def eval(
        dataName,
        correct,
        total,
        pos_actual,
        neg_actual,
        pos_correct,
        pos_pred,
        neg_correct,
        neg_pred
):

    # 1 accuracy
    accuracy = correct / total

    # 2 precision and recall for the positive class
    precision_pos = pos_correct / pos_pred
    recall_pos = pos_correct / pos_actual

    # 3 precision and recall for the negative class
    precision_neg = neg_correct / neg_pred
    recall_neg = neg_correct / neg_actual

    # 4 F1 score
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)
    f1_macro = (f1_pos + f1_neg) / 2

    print(f"""\
        ------------------------------------------------------------
                    {dataName}
        ------------------------------------------------------------
        Positive
            Precision : {precision_pos:.3f}     Recall : {recall_pos:.3f}    F1 : {f1_pos:.3f}
        Negative
            Precision : {precision_neg:.3f}     Recall : {recall_neg:.3f}    F1 : {f1_neg:.3f}
        Accuracy : {accuracy:.3f}
        Macro F1 : {f1_macro:.3f}
""")

def calculate_valence(texts, sentimentDictionary):

    # Initialize variables
    score = 0
    valence = []
    mod_multiplier = 1.0
    mod_step = 0
    neg_multiplier = 1.0
    neg_step = 0

    # If a terminal punctuation exist, separate into sentence chunks
    if any(punc in TERMINAL_PUNCTUATION for punc in texts):

        # Find index(es) of terminal punctuations
        i_ts = [i for i, val in enumerate(texts) if val in TERMINAL_PUNCTUATION]

        # Split text by terminal punctuations
        text_chunks = []
        start = 0
        for i_t in i_ts:
            text_chunks.append(texts[start:i_t])
            start = i_t + 1
        text_chunks.append(texts[start:])
        
        # Remove empty element (result of period and ellipsis)
        text_chunks = [chunk for chunk in text_chunks if chunk]

        # Calculate valence for each chunk and sum (continue below)
        for chunk in text_chunks:
            score += calculate_valence(chunk, sentimentDictionary)
        return score

    # Texts does not contain "but" or terminal punctuation anymore
    
    # Create valence by matching word to sentiment dictionary
    for word in texts:

        # Add sentiment value
        # If there is a modifier word
        if word in MODIFIER:
            mod_multiplier = MODIFIER[word]
            mod_step = 0
            valence.append(0)

        # If there is a negative word
        elif word in NEGATION:
            neg_multiplier = -0.75
            neg_step = 0
            valence.append(0)

        # If there is an exclamation mark
        elif word == "!":
            valence.append(0)
            # Increase valence of word before "!" by 50%
            if len(valence) >= 2:
                valence[-2] *= 1.5

        # word is a normal word and has sentiment value
        elif word in sentimentDictionary:
            value = sentimentDictionary[word] * mod_multiplier * neg_multiplier
            valence.append(value)

        # word is a normal word and doesn't have sentiment value
        else:
            valence.append(0)

        # Update steps for modifier and negation
        # Reset modifier and negation after 5 words
        if mod_multiplier != 1.0:
            mod_step +=1
            if mod_step == 5:
                mod_multiplier = 1.0
                mod_step = 0
        if neg_multiplier != 1.0:
            neg_step +=1
            if neg_step == 5:
                neg_multiplier = 1.0
                neg_step = 0

        # Adjust modifier and negation value
        # Intensifier : multiplier > 1.0
        if mod_multiplier > 1.0 and mod_step > 1:
            # Decrease by 5%
            # 1.33 1.26 1.20 1.14 1.08 1
            mod_multiplier *= 0.95

        # Diminisher : multiplier < 1.0
        elif mod_multiplier < 1.0 and mod_step > 1:
            # Increase by 10%
            # 0.67 0.74 0.81 0.89 0.98 1
            mod_multiplier *= 1.1

        # Negation : multiplier < 1.0
        if neg_multiplier < 1.0 and neg_step > 1:
            # Decrease by 5%
            # -0.75 -0.67 -0.61 -0.55 -0.49 1
            neg_multiplier *= 0.95

    return sum(valence)

def predict_sentiment(score, threshold):
    if score >= threshold:
        return "positive"
    else:
        return "negative"