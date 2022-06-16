import unittest

import spacy
import nltk
import os

from utils.nlp_utils import *


class UtilTest(unittest.TestCase):

    def test_org_clean(self):
        cleansed = clean_org_name('Hello World Limited')
        self.assertEqual('hello world', cleansed)

    def test_extract_organizations(self):
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')
        text = "Sentence: Sundar Pichai, the CEO of Google Inc. is walking in the streets of California."
        orgs = extract_organizations(text, nlp)
        self.assertEqual(1, len(orgs))
        self.assertEqual('google', orgs[0])

    def test_lemmatize_text(self):
        nltk.download('wordnet')
        nltk.download('punkt')
        text = "Sentence: Sundar Pichai, the CEO of Google Inc. is walking in the streets of California."
        self.assertEqual("sentenc sundar pichai googl walk street california", lemmatize_text(text))


if __name__ == '__main__':
    unittest.main()
