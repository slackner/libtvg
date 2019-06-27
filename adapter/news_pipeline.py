#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import pymongo
import sys
import spacy
import datetime
import logging
import re
import json
import argparse

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
    mongoClient = pymongo.MongoClient(host, port)
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

    parser = argparse.ArgumentParser(description="News Pipeline")
    parser.add_argument("config", help="Path to a configuration file")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    source = config['source']

    if 'article_extract_host' not in source:
        raise RuntimeError("No host for article_extract specified")
    if 'article_extract_port' not in source:
        raise RuntimeError("No port for article_extract specified")
    if 'article_extract_database' not in source:
        raise RuntimeError("No database for article_extract specified")
    if 'article_extract_collection' not in source:
        raise RuntimeError("No collection for article_extract specified")
    if 'article_and_entity_host' not in source:
        raise RuntimeError("No host for article_processed and entity specified")
    if 'article_and_entity_port' not in source:
        raise RuntimeError("No port for article_processed and entity specified")
    if 'article_and_entity_database' not in source:
        raise RuntimeError("No database for article and entity specified")
    if 'article_processed_collection' not in source:
        raise RuntimeError("article_processed_collection not specified")
    if 'entity_collection' not in source:
        raise RuntimeError("entity_collection not specified")

    article_extract_collection = getCollection(source['article_extract_host'], int(source['article_extract_port']), source['article_extract_database'], source['article_extract_collection'])
    article_processed_collection = getCollection(source['article_and_entity_host'], int(source['article_and_entity_port']), source['article_and_entity_database'], source['article_processed_collection'])
    entity_collection = getCollection(source['article_and_entity_host'], int(source['article_and_entity_port']), source['article_and_entity_database'], source['entity_collection'])

    logging_filename = os.path.dirname(os.path.abspath(__file__)) + '/unprocessed_date_formats.log'
    logging.basicConfig(filename=logging_filename, level=logging.INFO)

    # Process latest article using extracttime from the last processed article collection
    latest_record = article_processed_collection.find().sort("extracttime", pymongo.DESCENDING).limit(1)
    try:
        last_processed_article = latest_record.next()
        d = last_processed_article['extracttime']
    except:
        print("No data found!")
        # Use manually set datetime
        d = datetime.datetime.strptime(source['default_date'], '%Y-%m-%d')
        # d = datetime.datetime(2019, 5, 29)

    for post in article_extract_collection.find({"extracttime": {"$gte": d}}).sort("extracttime", pymongo.ASCENDING):
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
