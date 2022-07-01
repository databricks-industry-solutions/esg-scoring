import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import nltk


def clean_line(line):
    # removing header number
    line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
    # removing trailing spaces
    line = line.strip()
    # words may be split between lines, ensure we link them back together
    line = re.sub(r'\s?-\s?', '-', line)
    # remove space prior to punctuation
    line = re.sub(r'\s?([,:;\.])', r'\1', line)
    # ESG contains a lot of figures that are not relevant to grammatical structure
    line = re.sub(r'\d{5,}', r' ', line)
    # remove mentions of URLs
    line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                  r' ', line)
    # remove multiple spaces
    line = re.sub(r'\s+', ' ', line)
    # remove multiple dot
    line = re.sub(r'\.+', '.', line)
    # split paragraphs into well-defined sentences using nltk
    return line


def extract_statements(text):
    # remove non ASCII characters
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    lines = []
    prev = ""
    for line in text.split('\n'):
        # aggregate consecutive lines where text may be broken down
        # only if next line starts with a space or previous does not end with a dot.
        if line.startswith(' ') or not prev.endswith('.'):
            prev = prev + ' ' + line
        else:
            # new paragraph
            lines.append(prev)
            prev = line

    # don't forget left-over paragraph
    lines.append(prev)

    # clean paragraphs from extra space, unwanted characters, urls, etc.
    # best effort clean up, consider a more versatile cleaner
    sentences = []
    for line in lines:
        line = clean_line(line)
        try:
          for part in nltk.sent_tokenize(line):
              sentences.append(str(part).strip().lower())
        except:
          pass
    return sentences


def load_nltk(path):
    import nltk
    nltk.data.path.append("{}/wordnet".format(path))
    nltk.data.path.append("{}/punkt".format(path))
    nltk.data.path.append("{}/omw".format(path))


def load_spacy(path):
    import spacy
    return spacy.load(path)


def clean_org_name(text):
    import re
    text = text.lower()
    name = []
    stop_words = ['group', 'inc', 'ltd', 'ag', 'plc', 'limited', 'sa', 'holdings']
    for t in re.split('\\W', text):
        if len(t) > 0 and t not in stop_words:
            name.append(t)
    if len(name) > 0:
        return ' '.join(name).strip()
    else:
        return ''


def extract_organizations(text, nlp):
    doc = nlp(text)
    orgs = [X.text for X in doc.ents if X.label_ == 'ORG']
    return [clean_org_name(org) for org in orgs]


def tokenize(sentence):
    import re
    return [t for t in re.split('\\W', sentence.lower()) if len(t) > 0]


def lemmatize_text(text):
    results = []
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for token in tokenize(text):
        stem = stemmer.stem(lemmatizer.lemmatize(token))
        matcher = re.match('\\w+', stem)
        if matcher:
            part = matcher.group(0)
            if len(part) > 3:
                results.append(part)
    return ' '.join(results)


def get_stopwords(stop_words, organisations, path):
    load_nltk(path)
    org_stop_words = []
    for organisation in organisations:
        for t in re.split('\\W', organisation):
            lemma = lemmatize_text(t)
            if len(lemma) > 0:
                org_stop_words.append(lemma)
    stop_words = stop_words + org_stop_words
    return stop_words
