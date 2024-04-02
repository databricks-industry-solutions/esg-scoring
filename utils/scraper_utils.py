from bs4 import BeautifulSoup
import requests
from pathlib import Path
from io import BytesIO

def get_organizations(sector):
    index_url = "https://www.responsibilityreports.com/Companies?sect={}".format(sector)
    response = requests.get(index_url)
    soup = BeautifulSoup(response.text, features="html.parser")
    csr_entries = [link.get('href') for link in soup.findAll('a')]
    organizations = [ele.split("/")[-1] for ele in csr_entries if ele.startswith('/Company/')]
    return organizations


def get_organization_details(organization):
    # use beautiful soup to parse company page on responsibilityreports.com
    company_url = "https://www.responsibilityreports.com/Company/" + organization
    response = requests.get(company_url)
    soup = BeautifulSoup(response.text, features="html.parser")
    try:
        # page contains useful information such as company legal name and ticker
        name = soup.find('h1').text
        ticker = soup.find('span', {"class": "ticker_name"}).text
        csr_url = ""
        # also contains the link to their most recent disclosures
        for link in soup.findAll('a'):
            data = link.get('href')
            if data.split('.')[-1] == 'pdf':
                csr_url = 'https://www.responsibilityreports.com' + data
                break
        return [name, ticker, csr_url]
    except:
        # a lot of things could go wrong here, simply ignore that record
        return [None, None, None]


def download_csr(url, path):
    response = requests.get(url)
    filename = Path(path)
    filename.write_bytes(response.content)