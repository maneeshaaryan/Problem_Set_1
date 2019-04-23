import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import time
import datetime
import requests

def creepy_crawly(initial_page_no, max_pages):
    page = initial_page_no

    df = pd.DataFrame()
    while page <= max_pages:

        url = "https://boardgamegeek.com/browse/boardgame/page/" + str(page)
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, "html.parser")
        #print(plain_text)

        for eachGame in soup.findAll('tr', {'id' : 'row_'}):
            soup1 = BeautifulSoup(str(eachGame), "html.parser")
            #rankBlob = soup1.find('td', {'class': 'collection_rank'})
            #rank = rankBlob.find('a').get("name")
            #print(soup1)
            #print("title")
            titleBlob = soup1.find('td', {'class': 'collection_objectname'})
            titleSoup = BeautifulSoup(str(titleBlob), "html.parser")
            title = titleSoup.find('a').get_text()
            ratingBlob = soup1.findAll('td', {'class': 'collection_bggrating'})
            rating = []

            priceBlob = soup1.find('td',{'class':'collection_shop'})
            priceSoup = BeautifulSoup(str(priceBlob),"html.parser")
            pricelist = soup.find('div',{'class':'aad'})
            pricelistSoup = BeautifulSoup(str(pricelist),"html.parser")
            pricelistSoupdiv = pricelistSoup.findAll('div')
            #print(soup)
            listprice = []


            for eachRating in ratingBlob:
                rating.append(eachRating.get_text())
            #print (title + ' '+ rating[0]+ ' '+ rating[1]+ ' '+ rating[2])

            GeekRating = rating[0]
            AvgRating = rating[1]
            NumVoters = rating[2]

            GeekRating_c = GeekRating.strip()
            AvgRating_c = AvgRating.strip()
            NumVoters_c = NumVoters.strip()


            df = df.append({
            #'Rank': rank,
            'Name':title,
            'Geek_Rating': GeekRating_c,
            'Average_Rating' :AvgRating_c,
            'Number_of_Voters' :NumVoters_c
            }, ignore_index=True)

        page += 1

        df.to_csv("parsed_results/boardgamegeek_dataset3.csv")

creepy_crawly(1,1066)