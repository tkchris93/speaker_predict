import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from unidecode import unidecode
import numpy as np
from tqdm import tqdm

URL_HEAD = 'http://www.lds.org'

df = pd.read_pickle("top10.pdpkl")

talks = []
for i in xrange(df.shape[0]):
    soup = BeautifulSoup(requests.get(URL_HEAD + df.url.iloc[i]).text, 'html.parser')
    print df.name_trimmed.iloc[i]

    top = soup.find('ul', class_='pages-nav__list')

    list_of_additional_urls = [a['href'] for a in top.find_all('a', href=True)[:-1]]

    soups = [soup]
    for j, url in enumerate(list_of_additional_urls):
        print "  Getting soup {}/{}...".format(j+1, len(list_of_additional_urls))
        soups.append(BeautifulSoup(requests.get(URL_HEAD + url).text, 'html.parser'))
        #print "  Sleeping 2-3 sec..."
        time.sleep(2 + np.random.random())

    for sp in soups:
        all_links_soup = sp.find_all('a', class_='lumen-tile__link', href=True)
        for link_soup in all_links_soup:
            talks.append((df.name_trimmed.iloc[i], unidecode(link_soup.text.strip().split('\n')[0]), URL_HEAD + link_soup['href']))

pd.DataFrame(talks, columns=['speaker', 'talk', 'url']).to_pickle("talk_urls.pdpkl")
