#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import pymongo
import sys
# import kafka
import spacy
import datetime
import logging
import re

class NewsArticle:

    def __init__(self, url, lang, publishing_date, nlp_processing_date, extracttime):
        self.url = url
        self.publishing_date = publishing_date
        self.lang = lang
        self.nlp_processing_date = nlp_processing_date
        self.extracttime = extracttime

class SpacyModel:

    def __init__(self, model_name):
        self.nlp = spacy.load(model_name)

def getCollection(host, port, database, collection):
    mongoClient = client = pymongo.MongoClient(host, port)
    return mongoClient[database][collection]

def get_spacy_model_name(language):
    if language == 'de':
        spacy_model_name = 'de_core_news_sm'
    elif language == 'en':
        spacy_model_name = 'en_core_web_sm'
    else:
        spacy_model_name = 'de_core_news_sm'
    return spacy_model_name

def get_entites_with_sentence_ind(spacy_model, article_body, article_id):
    doc = spacy_model.nlp(article_body)
    entities = []
    for ind, sent in enumerate(doc.sents):
        for ent in sent.ents:
            entities.append({'sentence_index': ind, 'entity_name': ent.text, 'entity_type': ent.label_, 'article_id': article_id})
    return entities

def parse_date_format(date_string, crawltime):
    # Python versions < 3.7 only have very limited functionality for parsing
    # timezones. Both 'Z' (an alias for +0000) and timezone specifiers
    # containing a colon are not supported.
    date_string = re.sub(r'Z$', r'+0000', date_string)
    date_string = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', date_string)

    # Try parsing date field from String to datetime format. If parsing does not work use crawltime.
    try:
        # Format 2019-04-15
        return datetime.datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        pass

    try:
        # Format 2019-04-15T17:15:38+02:00 or 2019-05-23T11:13:00Z
        return datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        pass

    try:
        # Format 2019-05-22T03:33:14.929+02:00 or 2019-05-22T03:33:14.929Z
        return datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError:
        pass

    # Log not parseable date format
    logging.info('Format not processable: {}'.format(date_string))
    return crawltime

if __name__ == "__main__":
    logging_filename = os.path.dirname(os.path.abspath(__file__)) + '/unprocessed_date_formats.log'
    logging.basicConfig(filename=logging_filename, level=logging.INFO)

    if len(sys.argv) == 1:
        htmlFolder = os.getcwd()  # default: current working directory
    elif len(sys.argv) > 2:
        print("Too many command line arguments, [0] or [1] accepted.")
        sys.exit()
    elif len(sys.argv) == 2:
        htmlFolder = sys.argv[1]

    article_extract_collection = getCollection('localhost', 27019, 'LiveNews', 'extract')
    article_processed_collection = getCollection('localhost', 27017, 'tvg', 'article')
    entity_collection = getCollection('localhost', 27017, 'tvg', 'entity')

    # get the KafkaConsumer
    # consumer = kafka.KafkaConsumer('news-extract-available',
    #                     group_id='pipeline-extension-group',
    #                     bootstrap_servers=['localhost:9092'])

    # listen to kafka messages
    # for message in consumer:
        # url = message.key.decode('utf-8')

    # Process latest article using extracttime from the last processed article collection
    latest_record = article_processed_collection.find().sort("extracttime", pymongo.DESCENDING).limit(1)
    try:
        last_processed_article = latest_record.next()
        d = last_processed_article['extracttime']
    except:
        print("No data found!")
        d = datetime.datetime(2019, 5, 29)

    for post in article_extract_collection.find({"extracttime": {"$gte": d}}).sort("extracttime", pymongo.ASCENDING).limit(5):
        processedElement = article_processed_collection.find_one({"url":post['_id']})
        if processedElement == None:
            try:
                publishing_date = parse_date_format(post['date'], post['crawltime'])
            except:
                # date field does not exist
                publishing_date = post['crawltime']

            article = NewsArticle(post['_id'], post['lang'], publishing_date, datetime.datetime.now(), post['extracttime'])
            article_primary_key = article_processed_collection.insert_one(article.__dict__)
            try:
                spacy_model_name = get_spacy_model_name(article.lang)
                spacy_model = SpacyModel(spacy_model_name)
                entites_with_sentence_ind = get_entites_with_sentence_ind(spacy_model, post['body'], article_primary_key.inserted_id)
                entity_collection.insert_many(entites_with_sentence_ind)
            except Exception as e:
                article_processed_collection.delete_one({'_id': article_primary_key.inserted_id})
                print(e)
