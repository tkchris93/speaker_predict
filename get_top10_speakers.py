from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

with open("all_speakers.htm", 'rb') as f:
    lines = f.readlines()
soup = BeautifulSoup("".join(lines), 'html.parser')
out = []
for div in soup.find_all(class_='lumen-tile__title'):
    out.append((div.text.strip(), div.a['href']))
df = pd.DataFrame(out, columns=['name', 'url'])

def parse(s):
    p = re.compile('\([0-9]+\)')
    try: m = p.search(s).group(0)[1:-1]
    except Exception: m = 1
    return m

df['count'] = df.name.apply(parse).astype(int)
df['name_trimmed'] = df.name.apply(lambda s : s[:s.find('(')].strip())
df = df.sort_values('count', ascending=False)
df = df.drop_duplicates('name_trimmed')
df.head(10).to_pickle("top10.pdpkl")
