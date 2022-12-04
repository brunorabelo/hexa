import re
from datetime import datetime

import spacy


def get_stopwords_list(path):
    # opening the file in read mode
    my_file = open(path, "r")

    # reading the file
    data = my_file.read()

    # replacing end splitting the text
    # when newline ('\n') is seen.
    data_into_list = data.split("\n")
    my_file.close()
    return data_into_list


def preprocess(
        df,
        count_abreviations=False,
        create_keywords=False,
        lower=False,
        clean_text=False
):
    df["url_count"] = df["urls"].apply(lambda s: s[1:-1].count("\'") / 2)
    df["text_len"] = df["text"].apply(lambda s: len(s))
    df["hashtags_count"] = df["hashtags"].apply(lambda s: s[1:-1].count("\'") / 2)
    df["day"] = df["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t / 1000).day)
    df["hour"] = df["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t / 1000).hour)

    if count_abreviations:
        df["abrev"] = df["text"].apply(lambda s: count_abreviations_func(s))
    if lower:
        df['text'] = df['text'].str.lower()
    # create keywords
    if create_keywords:
        df["Macron"] = df["text"].apply(lambda s: ("macron" in s.lower().split()))
        df["Zemmour"] = df["text"].apply(lambda s: ("zemmour" in s.lower().split()))
        df["Melenchon"] = df["text"].apply(lambda s: ("melenchon" in s.replace("é", "e").lower().split()))
        df["rt"] = df["text"].apply(lambda s: ("rt" in s.lower().split()))
    if clean_text:
        stopwords = get_stopwords_list('data/stopwords-fr.txt')
        df['text'] = df['text'].apply(
            lambda s: clean_text_func(s, lower=lower, stopwords=stopwords))
    return df


def count_abreviations_func(text, all_vocabs=None):
    if not all_vocabs:
        return
    all_words = text.split(' ')
    abbrevitations = [x for x in all_words if 0 <= len(x) <= 3 and x.isalpha() and x not in all_vocabs]
    return len(abbrevitations) / len(all_words)


def clean_text_func(text, lower=True, stem=False, stopwords=[]):
    """Clean raw text."""
    # Lower
    if lower:
        text = text.lower()

        # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

        # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        # to use: python3 -m spacy download fr_core_news_md
        # https://stackoverflow.com/questions/13131139/lemmatize-french-text
        nlp = spacy.load('fr_core_news_md')
        text = " ".join([token.lemma_ for token in nlp(text)])

    return text
