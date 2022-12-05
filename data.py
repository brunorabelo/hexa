import re
from datetime import datetime

from spacy.lang.fr.stop_words import STOP_WORDS
import spacy
from pylexique import Lexique383
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer


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


def feature_abbrev(df):
    LEXIQUE = Lexique383()
    lexique = set(LEXIQUE.lexique.keys())
    df["abrev"] = df["text"].apply(lambda s: count_abreviations_func(s, lexique))
    return df


def feature_delete_stop_words(df, column="text_without_stopwords"):
    stop_words = set(STOP_WORDS)
    deselect_stop_words = ['n\'', 'ne', 'pas', 'plus', 'personne', 'aucun', 'ni', 'aucune', 'rien']
    for w in deselect_stop_words:
        if w in stop_words:
            stop_words.remove(w)
        else:
            continue

    df[column] = df['text_arr'].apply(
        lambda s: " ".join([w for w in s if not ((w in stop_words) or (len(w) == 1))]))
    return df


def feature_words_arr(df):
    df['text_arr'] = df['text'].apply(lambda s: re.sub("\W", " ", s).split())
    return df


def feature_sent_analysis(df, column='text'):
    tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    df["polarity"] = df[column].apply(lambda s: tb(s).sentiment[0])
    df["subjectivity"] = df[column].apply(lambda s: tb(s).sentiment[1])
    return df


def preprocess(
        df,
        create_keywords=False
):
    df["url_count"] = df["urls"].apply(lambda s: s[1:-1].count("\'") / 2)
    df["text_len"] = df["text"].apply(lambda s: len(s))
    df["hashtags_count"] = df["hashtags"].apply(lambda s: s[1:-1].count("\'") / 2)
    df["day"] = df["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t / 1000).day)
    df["hour"] = df["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t / 1000).hour)

    # create keywords
    if create_keywords:
        df["Macron"] = df["text"].apply(lambda s: ("macron" in s.lower().split()))
        df["Zemmour"] = df["text"].apply(lambda s: ("zemmour" in s.lower().split()))
        df["Melenchon"] = df["text"].apply(lambda s: ("melenchon" in s.replace("é", "e").lower().split()))
        df["rt"] = df["text"].apply(lambda s: ("rt" in s.lower().split()))
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
