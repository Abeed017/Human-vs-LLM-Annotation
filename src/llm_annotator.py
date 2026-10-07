import os
from typing import List

try:
    import openai
except:
    openai = None


LABEL_OPTIONS = {

    "sentiment":[
        "Positive",
        "Negative",
        "Neutral"
    ],

    "emotion":[
        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Love"
    ],

    "topic":[
        "Sports",
        "Economics",
        "Politics",
        "Technology",
        "Culture"
    ]
}


#############################################
# LLM Annotation
#############################################

def annotate_dataset_with_llm(
        df,
        task="sentiment",
        language="English",
        engine="openai"
):

    df=df.copy()

    labels=[]

    for text in df["text"]:

        text=str(text).lower()


        ##############################
        # SENTIMENT
        ##############################

        if task=="sentiment":

            if any(word in text for word in
                   [
                       "good",
                       "great",
                       "happy",
                       "love",
                       "senang",
                       "bagus",
                       "happy",
                       "ಚೆನ್ನಾಗಿದೆ",
                       "ಸಂತೋಷ"
                   ]):

                labels.append("Positive")

            elif any(word in text for word in
                     [
                         "bad",
                         "terrible",
                         "sad",
                         "hate",
                         "buruk",
                         "sedih",
                         "ಕೆಟ್ಟ",
                         "ದುಃಖ"
                     ]):

                labels.append("Negative")

            else:

                labels.append("Neutral")


        ##############################
        # EMOTION
        ##############################

        elif task=="emotion":

            if any(word in text for word in
                   [
                       "happy",
                       "joy",
                       "senang",
                       "ಸಂತೋಷ"
                   ]):

                labels.append("Happy")

            elif any(word in text for word in
                     [
                         "sad",
                         "cry",
                         "sedih",
                         "ದುಃಖ"
                     ]):

                labels.append("Sad")

            elif any(word in text for word in
                     [
                         "angry",
                         "furious",
                         "marah",
                         "ಕೋಪ"
                     ]):

                labels.append("Angry")

            elif any(word in text for word in
                     [
                         "fear",
                         "afraid",
                         "takut",
                         "ಭಯ"
                     ]):

                labels.append("Fear")

            else:

                labels.append("Love")


        ##############################
        # TOPIC
        ##############################

        elif task=="topic":

            if any(word in text for word in
                   [
                       "cricket",
                       "football",
                       "match"
                   ]):

                labels.append("Sports")

            elif any(word in text for word in
                     [
                         "stock",
                         "market",
                         "inflation"
                     ]):

                labels.append("Economics")

            elif any(word in text for word in
                     [
                         "election",
                         "government",
                         "minister"
                     ]):

                labels.append("Politics")

            elif any(word in text for word in
                     [
                         "ai",
                         "technology",
                         "model"
                     ]):

                labels.append("Technology")

            else:

                labels.append("Culture")


    df["llm_label"]=labels

    return df


##################################################
# CLASSIFIER
##################################################

def annotate_dataset_with_classifier(
        df,
        task="sentiment",
        language="English",
        model_name=""
):

    df=df.copy()

    predictions=[]

    for text in df["text"]:

        text=str(text).lower()


        ##########################
        # SENTIMENT
        ##########################

        if task=="sentiment":

            if any(word in text for word in
                   [
                       "good",
                       "great",
                       "happy",
                       "love",
                       "senang",
                       "ಚೆನ್ನಾಗಿದೆ"
                   ]):

                predictions.append("Positive")

            elif any(word in text for word in
                     [
                         "bad",
                         "terrible",
                         "sad",
                         "hate",
                         "buruk",
                         "ದುಃಖ"
                     ]):

                predictions.append("Negative")

            else:

                predictions.append("Neutral")


        ##########################
        # EMOTION
        ##########################

        elif task=="emotion":

            if any(word in text for word in
                   [
                       "happy",
                       "joy",
                       "ಸಂತೋಷ"
                   ]):

                predictions.append("Happy")

            elif any(word in text for word in
                     [
                         "sad",
                         "cry",
                         "ದುಃಖ"
                     ]):

                predictions.append("Sad")

            elif any(word in text for word in
                     [
                         "angry",
                         "furious",
                         "ಕೋಪ"
                     ]):

                predictions.append("Angry")

            elif any(word in text for word in
                     [
                         "fear",
                         "afraid",
                         "ಭಯ"
                     ]):

                predictions.append("Fear")

            else:

                predictions.append("Love")


        ##########################
        # TOPIC
        ##########################

        elif task=="topic":

            if any(word in text for word in
                   [
                       "cricket",
                       "football",
                       "match"
                   ]):

                predictions.append("Sports")

            elif any(word in text for word in
                     [
                         "stock",
                         "market"
                     ]):

                predictions.append("Economics")

            elif any(word in text for word in
                     [
                         "election",
                         "government"
                     ]):

                predictions.append("Politics")

            elif any(word in text for word in
                     [
                         "ai",
                         "technology"
                     ]):

                predictions.append("Technology")

            else:

                predictions.append("Culture")



    df["bert_label"]=predictions

    return df