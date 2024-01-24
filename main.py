import base64
import functions_framework
import base64
import html
import os

import functions_framework
import re
from datetime import datetime
import time
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import io
from google.cloud import storage
# nltk.download('stopwords')
# nltk.download('punkt')

rss_feeds = {
    'Blog': {
        'tds': ['https://towardsdatascience.com/feed'],
        'mlm': ['http://feeds.feedburner.com/MachineLearningMastery'],
        'gai': ['http://googleaiblog.blogspot.com/atom.xml']
    },
}
exc_list = ["model", "data", "data science", "science", "using", "photo", "image", "author", "updated", "python", "language", "computer"
            "dataframes", "dataframe", "science", "artificial", "intelligence", "ai", "world", "machine", "learning", "ml"
            "google", "research", "task", "help"
            ]
today_date = datetime.today().strftime('%Y%m%d')
bucket_name = 'story-store'
func_start_time = time.time()
# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def daily_story_record(cloud_event):
    print("Starting daily story fetch...")
    if cloud_event.data:
        print("Starting process: ", base64.b64decode(cloud_event.data["message"]["data"]))
    start_time = time.time()
    soups = fetch_article_soups()
    data = process_article_soups(soups)
    do_text_cleaning(data)
    data.to_csv(today_date + '.csv', index=False)
    filepath = today_date+'.csv'
    upload_blob(data, filepath)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time = end_time - func_start_time
    print(f"Daily Story Record Time: {elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")

def fetch_article_soups():
    start_time = time.time()
    feed_type = 'Blog'
    rss_urls = rss_feeds[feed_type]
    soup_list = []
    for key, subcategory_urls in rss_urls.items():
        for rss_url in subcategory_urls:
            soup = fetch_soup(rss_url)
            if soup:
                soup_list.append((key, soup))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"RSS Fetch Time: {elapsed_time} seconds")
    return soup_list

def fetch_soup(rss_url):
    response = requests.get(rss_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')
        return soup
    else:
        print(f"Failed to fetch RSS feed from {rss_url}. Status code: {response.status_code}")
        return None

def process_article_soups(soup_list):
    article_list = []
    for souple in soup_list:
        article_list += get_articles(souple[0], souple[1])

    print("\nNumber of articles found: ", len(article_list), '\n')

    df = pd.DataFrame(article_list)
    return df

def do_text_cleaning(df):
    start_time = time.time()
    df['Combined_Text'] = df['Title'] + ' ' + df['Body']
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
        return tokens
    df['Processed_Text'] = df['Combined_Text'].apply(preprocess_text)
    df['Processed_Title'] = df['Title'].apply(preprocess_text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Text Clean/Tokenizing Time: {elapsed_time} seconds")

def get_articles(key, soup):
    alist=[]
    if key not in ['gai']:
        articles = soup.findAll('item')
        for a in articles:
            title = a.find('title').text if a.find('title') else ''
            link = a.find('link').text if a.find('link') else ''
            pub_date_tag = a.find('pubDate') or a.find('pubdate')
            # published_date = parse_date(pub_date_tag, today_date)
            published_date = pub_date_tag.text
            body_tag = get_body(key, a)
            body_text = body_tag.get_text(strip=True) if body_tag else ''
            acbody = clean_text(body_text[:800])
            title = clean_text(title)
            acbody = '' if len(acbody) < 100 else acbody
            article = {'Title': title, 'Body': acbody, 'Link': link, 'Date': published_date}
            alist.append(article)

    if key in ['gai']:
        articles = soup.findAll('entry')
        for a in articles:
            title = a.find('title').text if a.find('title') else ''
            link = a.find('link').text if a.find('link') and a.find('link').text!='' else a.find('id').text if a.find('id') else ''
            pub_date_tag = a.find('published')
            published_date = pub_date_tag.text
            body_tag = get_body(key, a)
            body_text = body_tag.get_text(strip=True) if body_tag else ''
            acbody = clean_text(body_text[:800])
            title = clean_text(title)
            acbody = '' if len(acbody) < 100 else acbody
            article = {'Title': title, 'Body': acbody, 'Link': link, 'Date': published_date}
            alist.append(article)

    return alist

def get_body(key, a):
    # Handles TDS, MLM, Google AI
    b_tag = 'body' if a.find('body') else 'content:encoded' if a.find('content:encoded') else None

    if not b_tag:
        if key not in ['gai'] and a.find('description'):
            html_body = a.find('description').text
            unescaped_html = html.unescape(html_body)
            body = BeautifulSoup(unescaped_html, 'html.parser')
            x = body.findAll('p', 'medium-feed-snippet')
            if not len(x):
                x = body.findAll('p')
            b = x[0] if len(x) else None
        if key in ['gai']:
            if a.find('content'):
                html_body = a.find('content').text
                unescaped_html = html.unescape(html_body)
                body = BeautifulSoup(unescaped_html, 'html.parser')
                x = body.findAll('p')
                b = x[0] if len(x) else None
    else:
        b = a.find(b_tag)
    return b

def clean_text(text):
    root = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(root, 'nltk_data')
    # os.chdir(download_dir)
    nltk.data.path.append(download_dir)
    
    text = text.lower()
    text.replace("\u00A0", " ").replace('.','').replace(',','').replace(':',' ').replace('\'','').replace("..."," ").replace("  "," ").strip()
    pattern = r'<\/[^>]+>$'
    match = re.search(pattern, text)
    if match:
        content_after_last_tag = match.group()
        text = content_after_last_tag
    else:
        text = re.sub(r'<.*?>', '', text)
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    for keyword in exc_list:
        if keyword in text:
            text = text.replace(keyword, '')
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def upload_blob(df, filename):
    print(df.head())
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob_data = bucket.blob(f'{filename}')
        blob_data.upload_from_filename(filename)
    except:
        print("Could not upload to bucket.")
    os.remove(filename)