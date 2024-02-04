import os
import sys
import time
from datetime import datetime

import google.appengine.api
from flask import Flask, request, jsonify, json
from google.appengine.api import mail

import ops

app = Flask(__name__)
app.wsgi_app = google.appengine.api.wrap_wsgi_app(app.wsgi_app)

bucket_name = 'a-storyverse'
func_start_time = time.time()
today_date = datetime.today().strftime('%Y%m%d')
json_file_path='sources.json'
with open(json_file_path, 'r') as json_file:
    sources = json.load(json_file)
    print('~~~Sources loaded~~~')
with open(json_file_path, 'r') as json_file:
    exc_map = json.load(json_file)
    print('~~~Exclusions loaded~~~')

# try:
#     root = os.path.dirname(os.path.abspath(__file__))
#     download_dir = os.path.join(root, 'nltk_data')
#     nltk.data.load(
#         os.path.join(download_dir, 'tokenizers/punkt/english.pickle')
#     )
#     os.environ['NLTK_DATA'] = download_dir
#     print('~~~NLTK loaded~~~')
# except:
#     print("NLTK Load failure.", str(sys.exc_info()))

@app.route('/fetchstory', methods=['POST'])
def fetch_story():
    request_data = request.get_json()
    story_type = request_data.get('type').lower()
    # story_type = request.args.get('type').lower()
    print("Starting daily story fetch: ", story_type)
    start_time = time.time()
    try:

        soups = ops.fetch_article_soups(sources, [story_type])
        exc_list = exc_map[story_type]
        data = ops.process_article_soups(soups, exc_list)
        ops.do_text_preprocessing(data)
        # data.to_csv(today_date + '.csv', index=False)
        filepath = today_date + '.csv'
        ops.upload_blob(data, filepath, story_type, bucket_name)
        # ops.save_doc(data, filepath) # local
        os.remove(today_date + '.csv')
    except:
        print(str(sys.exc_info()))
        return jsonify({'API': "Topicverse", 'call': "fetchstory:" + story_type, "status": 'Failure'})
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time = end_time - func_start_time
    print(f"Daily Story Record Time: {elapsed_time} seconds")
    print(f"Run Time: {total_elapsed_time} seconds")
    print(f"Story fetch ended for type:",story_type)
    return jsonify({'API':"Topicverse", 'call': "fetchstory:"+story_type, "status": 'Complete'})

@app.route('/emailtest', methods=['POST'])
def email_test():
    try:
        projectid = 'topicverse'
        sender_address = f"summary@[{projectid}].appspotmail.com"
        message = mail.EmailMessage(
            sender=sender_address,
            subject="Test Email")

        message.to = "Subh <subhayuchakr@gmail.com>"
        message.body = """
        Test email.
        """
        message.send()
    except:
        print('Email failed: ', str(sys.exc_info()))
        return jsonify({'API':"Topicverse", 'call': "emailtest:", "status": 'Failure'})
    print('Email sent')
    return jsonify({'API':"Topicverse", 'call': "emailtest:", "status": 'Complete'})


@app.route('/', methods=['POST'])
def test():
    print('Reached API test')
    return jsonify({"API":"Topicverse", "Version": '1.0'})


@app.route('/param', methods=['POST'])
def params():
    print('Reached API test params', str(request.args))
    return jsonify({"API":"Topicverse", "Version": '1.0'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
