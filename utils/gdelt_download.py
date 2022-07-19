from datetime import datetime
from datetime import timedelta
from datetime import date
from zipfile import ZipFile
import gzip
import io
import os
import urllib.request


def download_content(url, file_name):
    with urllib.request.urlopen(url) as dl_file:
        input_zip = ZipFile(io.BytesIO(dl_file.read()), "r")
        name = input_zip.namelist()[0]
        with gzip.open(file_name, 'wb') as f:
            f.write(input_zip.read(name))

def download_to_dbfs(url, dst_path):
    file_name = '{}.gz'.format(url.split('/')[-1][:-4])
    # write to a mounted directory
    download_content(url, '/dbfs/{}/{}'.format(dst_path, file_name))
    
def download(min_date, max_date, dst_path):
    master_url = 'http://data.gdeltproject.org/gdeltv2/masterfilelist.txt'
    master_file = urllib.request.urlopen(master_url)
    to_download = []
    for line in master_file:
        decoded_line = line.decode("utf-8")
        if 'gkg.csv.zip' in decoded_line:
            a = decoded_line.split(' ')
            file_url = a[2].strip()
            file_dte = datetime_object = datetime.strptime(file_url.split('/')[-1].split('.')[0], '%Y%m%d%H%M%S')
            if (file_dte > min_date and file_dte <= max_date):
                to_download.append(file_url)

    print("{} file(s) to download from {} to {}".format(len(to_download), min_date, max_date))
    n = len(to_download)
    for i, url in enumerate(to_download):
        download_to_dbfs(url, dst_path)
        print("{}/{} [{}]".format(i + 1, n, url))





