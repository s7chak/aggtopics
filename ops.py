import ast
import base64
import copy
import html
import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
import pandas as pd
# import pyLDAvis
import re
import requests
import requests
import time
import time
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime
from datetime import datetime
from flask import Flask, request
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from io import BytesIO
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from pathlib import Path
# from pyLDAvis import gensim
from wordcloud import WordCloud
from google.cloud import storage
import ops


def fetch_article_soups(rss_urls):
    start_time = time.time()
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


def process_article_soups(soup_list, exc_list):
    article_list = []
    start_time = time.time()
    for souple in soup_list:
        article_list += get_articles(souple[0], souple[1], exc_list)

    print("\nNumber of articles found: ", len(article_list), '\n')

    df = pd.DataFrame(article_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Article Processing Time: {elapsed_time} seconds")
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


def get_articles(key, soup, exc_list):
    alist = []
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
            acbody = clean_text(body_text[:800], exc_list)
            title = clean_text(title, exc_list)
            acbody = '' if len(acbody) < 100 else acbody
            article = {'Title': title, 'Body': acbody, 'Link': link, 'Date': published_date}
            alist.append(article)

    if key in ['gai']:
        articles = soup.findAll('entry')
        for a in articles:
            title = a.find('title').text if a.find('title') else ''
            link = a.find('link').text if a.find('link') and a.find('link').text != '' else a.find('id').text if a.find(
                'id') else ''
            pub_date_tag = a.find('published')
            published_date = pub_date_tag.text
            body_tag = get_body(key, a)
            body_text = body_tag.get_text(strip=True) if body_tag else ''
            acbody = clean_text(body_text[:800], exc_list)
            title = clean_text(title, exc_list)
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


def clean_text(text, exc_list):
    root = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(root, 'nltk_data')
    # os.chdir(download_dir)
    nltk.data.path.append(download_dir)

    text = text.lower()
    text.replace("\u00A0", " ").replace('.', '').replace(',', '').replace(':', ' ').replace('\'', '').replace("...",
                                                                                                              " ").replace(
        "  ", " ").strip()
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


def upload_blob(df, filename, bucket_name):
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob_data = bucket.blob(f'{filename}')
        blob_data.upload_from_filename(filename)
    except:
        print("Could not upload to bucket.")
    os.remove(filename)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=1200, height=900, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img_bytes_io = BytesIO()
    wordcloud.to_image().save(img_bytes_io, format='PNG')
    img_bytes = img_bytes_io.getvalue()
    plt.close()
    return img_bytes

def find_top20words(df, field):
    try:
        df[field] = df[field].apply(ast.literal_eval)
    except:
        return {}
    all_terms = [term for sublist in df[field].tolist() for term in sublist]
    term_counts = Counter(all_terms)
    top_20_terms = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True)[:20])
    return top_20_terms

def do_wordclouds(data, fields):
    df = copy.deepcopy(data)
    res = {'top20': {}, 'wc': {}}
    for field in fields:
        # Top20terms
        top_20_terms = find_top20words(df, field)
        clean_text = ' '.join([term for sublist in df[field].tolist() for term in sublist])
        ttwc = generate_wordcloud(clean_text)
        res['top20'][field] = top_20_terms
        res['wc'][field] = ttwc
    return res


# def do_lda_html(data, field):
#     field = 'Processed_Text'
#     processed_titles = data[field].apply(eval)
#     dictionary = Dictionary(processed_titles)
#     corpus = [dictionary.doc2bow(title) for title in processed_titles]
#     coherence_values = []
#     model_list = []
#     for num_topics in range(1, round(len(processed_titles)/5)):
#         lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
#         model_list.append(lda_model)
#         coherencemodel = CoherenceModel(model=lda_model, texts=data[field].apply(eval).to_list(), dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     optimal_num_topics = coherence_values.index(max(coherence_values)) + 1
#     optimal_lda_model = LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary)

#     print(f"Optimal Number of Topics: {optimal_num_topics}")
#     for topic_num in range(optimal_num_topics):
#         print(f"Topic {topic_num + 1}: {optimal_lda_model.print_topic(topic_num)}")

#     if optimal_num_topics > 1:
#         prepared_data = pyLDAvis.gensim.prepare(optimal_lda_model, corpus, dictionary)
#         html_string = pyLDAvis.prepared_data_to_html(prepared_data)
#         html_path = Path("output/lda_viz.html")
#         pyLDAvis.save_html(prepared_data, str(html_path))
#         return html_string
#     return None


def do_dmm_analysis(dictionary, texts):
    group_topics = 10
    gsdmm = MovieGroupProcess(K=group_topics, alpha=0.1, beta=0.3, n_iters=group_topics)
    y = gsdmm.fit(texts, len(dictionary))

    doc_count = np.array(gsdmm.cluster_doc_count)
    print('Number of documents per topic :', doc_count)

    # Topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-group_topics:][::-1]
    print('Most important clusters (by number of docs inside):', top_index)

    # define function to get top words per topic
    def top_words(cluster_word_distribution, top_cluster, values):
        for cluster in top_cluster:
            sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
            print("\nCluster %s : %s" % (cluster, sort_dicts))

    # get top words in topics
    top_words(gsdmm.cluster_word_distribution, top_index, 20)

    cluster_word_distribution = gsdmm.cluster_word_distribution

    topic_num = 0
    # Select topic you want to output as dictionary (using topic_number)
    topic_dict = sorted(cluster_word_distribution[topic_num].items(), key=lambda k: k[1], reverse=True)  # [:values]

    # Generate a word cloud image
    wordcloud = WordCloud(background_color='#fcf2ed',
                          width=1000,
                          height=600,
                          colormap='flag').generate_from_frequencies(topic_dict)

    # Print to screen
    fig, ax = plt.subplots(figsize=[20, 10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off");
    # Save to disk
    plt.savefig('dmm_summary_cloud.png')

def do_topicwizard_analysis(dictionary, texts):
    min_topics = 1

    vectorizer = CountVectorizer(min_df=min_topics, max_df=5)

    # Creating a Dirichlet Multinomial Mixture Model with 30 components
    dmm = DMM(n_components=5, n_iterations=100, alpha=0.1, beta=0.1)

    # Creating topic pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("dmm", dmm),
    ])
    full_string = texts[0]
    pipeline.fit(full_string)
    topicwizard.visualize(pipeline=pipeline, corpus=full_string)

def do_lda_analysis(df, corpus, dictionary, texts):
    coherence_values = []
    model_list = []
    for num_topics in range(2, 4):
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=df['Processed_Text'], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"LDA Time: {elapsed_time} seconds")

    optimal_num_topics = coherence_values.index(max(coherence_values)) + 2  # Adding 2 because we started the loop from 2
    optimal_lda_model = models.LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary)

    print(f"Optimal Number of Topics: {optimal_num_topics}")
    for topic_num in range(optimal_num_topics):
        print(f"Topic {topic_num + 1}: {optimal_lda_model.print_topic(topic_num)}")

    # topic_assignments = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    # # Topic Summarization
    # topic_summaries = [lda_model.print_topic(topic_num) for topic_num in range(lda_model.num_topics)]
    #
    # # Quality Evaluation
    # coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    # coherence_score = coherence_model.get_coherence()
    # perplexity_score = lda_model.log_perplexity(corpus)

    prepared_data = pyLDAvis.gensim.prepare(optimal_lda_model, corpus, dictionary)
    # pyLDAvis.display(prepared_data)
    # pyLDAvis.save_html(prepared_data, image_path+'topic_cluster.html')
    html_content = pyLDAvis.prepared_data_to_html(prepared_data)
    report_collection['data']['pyldavis_html'] = html_content

