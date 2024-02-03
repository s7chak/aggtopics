from flask import Flask, request, jsonify, json
from datetime import datetime
import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import base64
import ops
import sys


bucket_name = 'a-storyverse'
func_start_time = time.time()
today_date = datetime.today().strftime('%Y%m%d')
json_file_path='sources.json'

with open(json_file_path, 'r') as json_file:
    sources = json.load(json_file)
with open(json_file_path, 'r') as json_file:
    exc_map = json.load(json_file)

# @app.route('/fetchstory', methods=['POST'])
def fetch_story(type):
    story_type = type
    print("Starting daily story fetch: ", story_type)
    start_time = time.time()
    try:
        soups = ops.fetch_article_soups(sources, [story_type])
        exc_list = exc_map[story_type]
        data = ops.process_article_soups(soups, exc_list)
        ops.do_text_preprocessing(data)
        data.to_csv(today_date + '.csv', index=False)
        filepath = today_date + '.csv'
        ops.upload_blob(filepath, bucket_name)
        # ops.save_doc(data, filepath) # local
        os.remove(today_date + '.csv')
    except:
        print(str(sys.exc_info()))
        # return jsonify({'API': "Topicverse", 'call': "fetchstory:" + story_type, "status": 'Failure'})
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time = end_time - func_start_time
    print(f"Daily Story Record Time: {elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")
    print(f"Story fetch ended for type:",story_type)
    # return jsonify({'API':"Topicverse", 'call': "fetchstory:"+story_type, "status": 'Complete'})


fetch_story('blog')



# @app.route('/')
# def home():
#     print('Reached API home')
#     return jsonify({"API":"Topicverse", "Version": '1.0'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
