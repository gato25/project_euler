import requests
from bs4 import BeautifulSoup

resp = requests.get('https://projecteuler.net/problem=1')
soup = BeautifulSoup(resp.text, 'html.parser')

h1_tag = soup.find('h2').text.strip()

print(h1_tag)
        


