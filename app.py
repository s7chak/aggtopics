from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import base64
import ops

app = Flask(__name__)

rss_feeds = {
    'Blog': {
        'tds': ['https://towardsdatascience.com/feed'],
        'mlm': ['http://feeds.feedburner.com/MachineLearningMastery'],
        'gai': ['http://googleaiblog.blogspot.com/atom.xml']
    },
}
exc_map = {'Blog' : ["model", "data", "data science", "science", "using", "photo", "image", "author", "updated", "python", "language", "computer"
            "dataframes", "dataframe", "science", "artificial", "intelligence", "ai", "world", "machine", "learning", "ml"
            "google", "research", "task", "help"
            ]
           }
bucket_name = 'story-store'
func_start_time = time.time()
today_date = datetime.today().strftime('%Y%m%d')

@app.route('/fetchstory')
def fetch_story():
    story_type = request.args.get('type')
    print("Starting daily story fetch...", story_type)
    start_time = time.time()
    soups = ops.fetch_article_soups(rss_feeds[story_type])
    exc_list = exc_map[story_type]
    data = ops.process_article_soups(soups, exc_list)
    ops.do_text_cleaning(data)
    data.to_csv(today_date + '.csv', index=False)
    filepath = today_date + '.csv'
    ops.upload_blob(data, filepath, bucket_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time = end_time - func_start_time
    print(f"Daily Story Record Time: {elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")
    print(f"Story fetch ended for type:",story_type)
    return jsonify({'API':"Topicverse", 'call': "fetchstory:"+story_type, "status": 'Complete'})


@app.route('/')
def home():
    print('Reached API home')
    return jsonify({'API':"Topicverse"})

if __name__ == '__main__':
	app.run(host="0.0.0.0")
