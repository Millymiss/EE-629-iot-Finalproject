#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Bowen Li
"""

from bs4 import BeautifulSoup
import re
import time
import requests

def getName(feature):
    """
    <span id="productTitle" class="a-size-large">LG Electronics 24LJ4540-WU 24-Inch 720p LED TV (2017 Model)</span>
    """
    name='NA'
    nameChunk=feature.find('span',{'id':re.compile('productTitle')})
    if nameChunk: 
        name=nameChunk.text.strip()
    return name

def getBrand(feature):
    """
    <a id="bylineInfo" class="a-link-normal" href="/LG/b/ref=bl_dp_s_web_2529790011?ie=UTF8&amp;node=2529790011&amp;field-lbr_brands_browse-bin=LG">LG</a>
    """
    brand='NA'
    brandChunk=feature.find('a',{'id':re.compile('bylineInfo')})
    if brandChunk: brand=brandChunk.text
    return brand

def getPrice(feature):
    """
    <span id="priceblock_ourprice" class="a-size-medium a-color-price">$116.99</span>
    """
    price = 0
    priceChunk = feature.find('span',{'id':re.compile('priceblock_ourprice')})
    if priceChunk: price = priceChunk.text
    return price

def getPrime(feature):
    """
    <i class="a-icon a-icon-prime"><span class="a-icon-alt">Free Shipping for Prime Members</span></i>
    """
    prime = 0
    isprime=feature.find('i',{'class':re.compile('a-icon a-icon-prime')})
    if isprime:
        prime = 1
    return prime

def getStars(feature):
    """
    <i class="a-icon a-icon-star a-star-3-5"><span class="a-icon-alt">3.7 out of 5 stars</span></i>
    """
    stars='NA'
    star=feature.find('i',{'class':re.compile('a-icon a-icon-star')})
    if star: stars=star.text.strip()
    return stars

def getReview(feature):
    """
    <span id="acrCustomerReviewText" class="a-size-base">80 customer reviews</span>
    """
    review = '0 customer reviews'
    reviewChunk = feature.find('span',{'id':re.compile('acrCustomerReviewText')})
    if reviewChunk: review = reviewChunk.text
    return review

def getQuestion(feature):
    """
   <a id="askATFLink" class="a-link-normal askATFLink" href="#Ask">
      <span class="a-size-base">
        104 answered questions
      </span>
    </a>
    """
    question = '0 answered questions'
    questionChunk = feature.find('a',{'id':re.compile('askATFLink')})
    if questionChunk:
        question = questionChunk.text.strip()
    return question

def getStock(feature):
    """
    <span class="a-size-medium a-color-success">In Stock.</span>
    """
    stock = 'NA'
    isStock = feature.find('span',{'class':re.compile('a-size-medium a-color-success')})
    if isStock:
        stock = "InStock"
    return stock

def getDelivery(feature):
    """
    <div id="merchant-info" class="a-section a-spacing-mini">Ships from and sold by Amazon.com.
            <span class="">
            
            </span>
        </div>
    """
    delivery='NA'
    deliveryChunk=feature.find('div',{'id':re.compile('merchant-info')})
    if deliveryChunk:
        if deliveryChunk.text.strip() == 'Ships from and sold by Amazon.com.':
            delivery='Amazon.com'
    return delivery

def run(url):

    pageNum=1 # number of pages to collect
	
    for p in range(1,pageNum+1): # for each page 

        print ('page',p)
        html=None

        if p==1: pageLink=url # url for page 1 为了防止输入1的时候显示的不是第一页
        else: pageLink=url+'?page='+str(p)+'&sort=' # make the page url
		
        for i in range(5): # try 5 times
            try:
                #use the browser to access the url
                response=requests.get(pageLink,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                html=response.content # get the html
                break # we got the file, break the loop
            except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
                print ('failed attempt',i)
                time.sleep(2) # wait 2 secs
				
		
        if not html:continue # couldnt get the page, ignore
        
        soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') # parse the html  prepare the page to search

        features=soup.findAll('div', {'id':re.compile('dp-container')}) # get all the review divs
        

        
        for feature in features:
            name = getName(feature)
            price = getPrice(feature)
            review = getReview(feature)
            prime = getPrime(feature)
            stock = getStock(feature)
            brand = getBrand(feature)
            stars = getStars(feature)
            question = getQuestion(feature)
            delivery = getDelivery(feature)
            #print(name, price, review, prime, stock, brand, stars, question, delivery, '\n')
            
            f.write(name)
            f.write('\t')
            f.write(brand)
            f.write('\t')
            f.write(str(price))
            f.write('\t')
            f.write(str(prime))
            f.write('\t')
            f.write(stars)
            f.write('\t')
            f.write(review)
            f.write('\t')
            f.write(stock)
            f.write('\t')
            f.write(delivery)
            f.write('\t')
            f.write(question)
            f.write('\t')
            f.write('TVs')
            f.write('\t')
            f.write('1')
            f.write('\n')
            
            #f.close()
            
            
def getProductUrl(URL):
    headers = {'Host': 'www.amazon.com',
               'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:52.0) Gecko/20100101 Firefox/52.0',
               'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Language':'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
               'Accept-Encoding':'gzip, deflate, br',
               'Connection':'keep-alive'
              }
            
    session = requests.session()
    get_html = session.get(URL, headers=headers)
    con = BeautifulSoup(get_html.text, 'lxml')
    
    urls = []
    
    contents = con.find_all('span', class_="aok-inline-block zg-item")
    
   #for i in range(0, 50) for not best seller
    for i in range(0, 10):
        u = contents[i].find('a', class_='a-link-normal').get('href')
        url = 'https://www.amazon.com' + u
        if not url:
            s = 'we only have' + (i / 4).str() + 'products'
            print(s)
            break
    
        urls.append(url)
        
    return urls

if __name__ == '__main__':
    #URL = []
    urls = []
    f = open('demo.txt','a')
    f.write('Name'+'\t'+'Made by'+'\t'+'Price'+'\t'+'Prime or not'+'\t'+'Stars'+'\t'+'# of Reviews'+'\t'+'Stock'+'\t'+'Delivery option'+'\t'+'Questions'+'\t'+'Category'+'\t'+'Label')
    f.write('\n')
    """
    # for not best seller
    u = ['https://www.amazon.com/Best-Sellers-Electronics-LED-LCD-TVs/zgbs/electronics/6459737011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Laptop-Computers/zgbs/electronics/13896609011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-All-One-Computers/zgbs/electronics/13896603011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Home-Theater-Systems/zgbs/electronics/281056/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Computers-Tablets/zgbs/electronics/13896617011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Computer-Printers/zgbs/electronics/172635/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Digital-Cameras/zgbs/electronics/281052/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Unlocked-Cell-Phones/zgbs/electronics/2407749011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Portable-Bluetooth-Speakers/zgbs/electronics/7073956011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Electronics-Smartwatches/zgbs/electronics/7939901011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2',
         'https://www.amazon.com/Best-Sellers-Womens-Fashion/zgbs/fashion/7147440011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Mens-Fashion/zgbs/fashion/7147441011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Girls-Fashion/zgbs/fashion/7147442011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Boys-Fashion/zgbs/fashion/7147443011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Baby-Clothing-Shoes/zgbs/fashion/7147444011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Novelty-More/zgbs/fashion/7147445011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Luggage-Travel-Gear/zgbs/fashion/9479199011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Uniforms-Work-Safety/zgbs/fashion/7586144011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2,
         'https://www.amazon.com/Best-Sellers-Costumes-Accessories/zgbs/fashion/7586165011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2, 
         'https://www.amazon.com/Best-Sellers-Shoe-Jewelry-Watch-Accessories/zgbs/fashion/7586146011/ref=zg_bs_pg_2?_encoding=UTF8&pg=2 Shoe     
         ]
    """
    # for best seller
    u = ['https://www.amazon.com/gp/bestsellers/electronics/6459737011/ref=sr_bs_0_6459737011_1',
         'https://www.amazon.com/Best-Sellers-Electronics-Camera-Photo/zgbs/electronics/502394/ref=zg_bs_nav_e_1_e',
         'https://www.amazon.com/Best-Sellers-Electronics-Car/zgbs/electronics/1077068/ref=zg_bs_nav_e_1_e']
   
    for url in u:
        urls += getProductUrl(url)

    for i in range(0, len(urls)):
        run(urls[i])
    f.close()
    
    

    
    
