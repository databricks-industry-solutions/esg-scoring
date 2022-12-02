import unittest

import spacy

from utils.nlp_utils import *
from utils.scraper_utils import *


class UtilTest(unittest.TestCase):

    def setUp(self):
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        spacy.cli.download("en_core_web_sm")

    def test_org_clean(self):
        stop_words = ['group', 'inc', 'ltd', 'ag', 'plc', 'limited', 'sa', 'holdings']
        for stop_word in stop_words:
            cleansed = clean_org_name('Hello World {}'.format(stop_word))
            self.assertEqual('hello world', cleansed)
        cleansed = clean_org_name('Hello World Antoine')
        self.assertEqual('hello world antoine', cleansed)

    def test_extract_organizations(self):
        nlp = spacy.load('en_core_web_sm')
        text = "Sundar Pichai, the CEO of Google Inc. is walking down the streets of San Fransisco."
        orgs = extract_organizations(text, nlp)
        self.assertEqual(1, len(orgs))
        self.assertEqual('google', orgs[0])

    def test_lemmatize_text(self):
        text = "Sundar Pichai, the CEO of Google Inc. is walking down the streets of San Fransisco."
        self.assertEqual("sundar pichai googl walk down street fransisco", lemmatize_text(text))

    def test_clean_sentences(self):
        text = "Sundar Pichai, the CEO of Google Inc. http://google.com is walking down the streets of San Fransisco."
        expected = "Sundar Pichai, the CEO of Google Inc. is walking down the streets of San Fransisco."
        self.assertEqual(expected, clean_line(text))

    def test_extract_statements(self):
        text = """Our approach to environmental and social issues is becoming increasingly integrated in the work we do across our business and is subject to the governance and oversight of our management and Board structures. Reflecting this trend, we have taken the decision to integrate our ESG reporting into this year’s Annual Report.
Our approach to environmental and social issues is grounded in the work we do every day, right across our business, supported in turn by the governance and oversight of our management and Board structures.
In line with our commitment to give our shareholders a ‘Say on Climate’, we are giving shareholders an opportunity to vote to endorse our climate strategy, targets and progress at our 2022 Annual General Meeting"""
        sentences = extract_statements(text)
        self.assertEqual(4, len(sentences))
        self.assertEqual('our approach to environmental and social issues is becoming increasingly integrated in the work we do across our business and is subject to the governance and oversight of our management and board structures.', sentences[0])
        self.assertEqual('reflecting this trend, we have taken the decision to integrate our esg reporting into this years annual report.', sentences[1])
        self.assertEqual('our approach to environmental and social issues is grounded in the work we do every day, right across our business, supported in turn by the governance and oversight of our management and board structures.', sentences[2])
        self.assertEqual('in line with our commitment to give our shareholders a say on climate, we are giving shareholders an opportunity to vote to endorse our climate strategy, targets and progress at our 2022 annual general meeting', sentences[3])

    def test_count_organizations(self):
        # testing connectivity to responsibility reports.com
        orgs = get_organizations(4)
        self.assertTrue(len(orgs) > 0)
        self.assertTrue('barclays' in orgs)

    def test_org_details(self):
        # testing connectivity to responsibility reports.com
        details = get_organization_details('barclays')
        self.assertEqual(details[1], 'BARC')

    def test_pdf_download(self):
        # testing connectivity to responsibility reports.com
        url = 'https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/g/NYSE_GS_2020.pdf'
        statements = extract_statements(download_csr(url))
        expected = 'goldman sachs 2020 sustainability reportintegrating sustainability with purpose across our business the future'
        self.assertTrue(statements[0].startswith(expected))


if __name__ == '__main__':
    unittest.main()
