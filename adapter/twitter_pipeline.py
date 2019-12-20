#!/usr/bin/env python3
import elasticsearch_dsl
import elasticsearch
import configparser
import cachetools
import functools
import traceback
import datetime
import mockupdb
import getpass
import struct
import spacy
import time
import bson
import sys
import os

nlp = spacy.load("de_core_news_sm", disable=["parser"])
article_cache = cachetools.TTLCache(maxsize=10240, ttl=1800)

# Convert integer value to bson.ObjectId
def int_to_bson_oid(value):
    if (value >> 96) != 0:
        raise RuntimeError("cannot encode integer as OID")
    binary = struct.pack(">IQ", value >> 64, value & ((1 << 64) - 1))
    return bson.ObjectId(binary)

# Convert bson.ObjectId to integer value
def bson_oid_to_int(oid):
    if not isinstance(oid, bson.ObjectId):
        raise RuntimeError("object is not an OID")
    high, low = struct.unpack(">IQ", oid.binary)
    return (high << 64) | low

# Convert datetime object to unix timestamp
def datetime_to_unix(date):
    diff = date - datetime.datetime.utcfromtimestamp(0)
    return (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds // 1000)

# Convert date string to datetime object
def string_to_datetime(string):
    result = datetime.datetime.strptime(string, "%a %b %d %H:%M:%S %z %Y")
    return result.astimezone(tz=None)

# Convert MongoDB condition to ElasticSearch
def mongodb_condition_to_es(key, value):
    if not isinstance(value, dict):
        if isinstance(value, bson.ObjectId):
            value = bson_oid_to_int(value)
        if isinstance(value, datetime.datetime):
            value = datetime_to_unix(value)
            args = {key: {"gte": value, "lte": value, 'format': 'epoch_millis'}}
        else:
            args = {key: {"gte": value, "lte": value}}
        return elasticsearch_dsl.Q("range", **args)

    if len(value) != 1:
        raise NotImplementedError

    op, value = list(value.items())[0]
    if op in ['$gte', '$lte', '$gt', '$lt']:
        if isinstance(value, bson.ObjectId):
            value = bson_oid_to_int(value)
        if isinstance(value, datetime.datetime):
            value = datetime_to_unix(value)
            args = {key: {op[1:]: value, 'format': 'epoch_millis'}}
        else:
            args = {key: {op[1:]: value}}
        return elasticsearch_dsl.Q("range", **args)

    raise NotImplementedError

# Convert MongoDB query to ElasticSearch
def mongodb_query_to_es(query):
    result = []

    for key, value in query.items():
        if   key == 'id':
            result.append(mongodb_condition_to_es("id", value))
        elif key == 'time':
            result.append(mongodb_condition_to_es("created_at", value))
        elif key == '$or':
            subresult = [mongodb_query_to_es(sub) for sub in value]
            result.append(functools.reduce((lambda x, y: x | y), subresult))
        else:
            raise NotImplementedError("cannot translate %s=%s" % (key, value))

    result = functools.reduce((lambda x, y: x & y), result)
    return result

# Convert MongoDB sort order to ElasticSearch
def mongodb_sort_to_es(sort):
    result = []
    for key, value in sort.items():
        if   (key, value) == ("id", 1):
            result.append("id")
        elif (key, value) == ("id", -1):
            result.append("-id")
        elif (key, value) == ("time", 1):
            result.append("created_at")
        elif (key, value) == ("time", -1):
            result.append("-created_at")
        else:
            raise NotImplementedError
    return result

def process_article(hit):
    global nlp

    # FIXME: Take tags into account.
    # FIXME: Take terms into account.
    # FIXME: Take author into account.
    # FIXME: Take other referenced users into account.

    doc = nlp(hit['text'])

    entities = []
    for entity in doc.ents:
        entities.append({'sentence': 1, 'text': entity.text})

    return entities

def handle_request(request):
    global article_cache
    global es

    if request.matches("isMaster"):
        request.replies({'ok': 1, 'maxWireVersion': 5})

    elif request.matches("ping"):
        request.replies({'ok': 1})

    elif request.matches(find="articles"):
        search = elasticsearch_dsl.Search(using=es, index="twitter_pipeline")
        search = search.source(include=['id', 'created_at', 'text', 'user.name', 'entities'])

        if 'filter' in request and len(request['filter']) > 0:
            search = search.query('bool', filter=[mongodb_query_to_es(request['filter'])])
        if 'sort' in request:
            search = search.sort(*mongodb_sort_to_es(request['sort']))
        if 'limit' in request:
            search = search[:request['limit']]
            results = search.execute()
        else:
            results = search.scan()

        articles = []
        for hit in results:
            articles.append({'id':   int_to_bson_oid(hit['id']),
                             'time': string_to_datetime(hit['created_at'])})

            try:
                entities = article_cache[hit['id']]
            except KeyError:
                entities = process_article(hit)
            article_cache[hit['id']] = entities

        if 'limit' in request:
            assert len(articles) <= request['limit']

        request.replies({'cursor': {'id': 0, 'firstBatch': articles}})

    elif request.matches(find="entities"):
        if 'filter' not in request:
            raise NotImplementedError
        if len(request['filter']) != 1:
            raise NotImplementedError
        key, value = list(request['filter'].items())[0]
        if key != "article":
            raise NotImplementedError

        article_id = bson_oid_to_int(value)

        try:
            entities = article_cache[article_id]

        except KeyError:
            search = elasticsearch_dsl.Search(using=es, index="twitter_pipeline")
            search = search.source(include=['id', 'created_at', 'text', 'user.name', 'entities'])
            search = search.query('bool', filter=[elasticsearch_dsl.Q("terms", **{"id": [article_id]})])

            response = search.execute()
            if len(response) != 1:
                raise RuntimeError("expected 1 article, got %d" % len(response))

            assert response[0]['id'] == article_id
            entities = process_article(response[0])

        article_cache[article_id] = entities
        request.replies({'cursor': {'id': 0, 'firstBatch': entities}})

    else:
        raise NotImplementedError

def autoresponder(request):
    try:
        handle_request(request)
    except:
        print("Exception in handle_request:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        request.fail()

    # Never put into the request queue.
    return True

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.expanduser("~/.config/twitter_pipeline.conf"))

    try:
        username = config.get('elasticsearch', 'username')
    except (configparser.NoSectionError, configparser.NoOptionError):
        username = getpass.getuser()
        sys.stderr.write("Assuming elasticsearch username is %s\n" % username)

    try:
        password = config.get('elasticsearch', 'password')
    except (configparser.NoSectionError, configparser.NoOptionError):
        password = getpass.getpass()

    es = elasticsearch.Elasticsearch(['https://elastic-dbs.ifi.uni-heidelberg.de'],
                                     http_auth=(username, password), scheme='https', port=443)

    server = mockupdb.MockupDB(port=27022)
    responder = server.autoresponds(autoresponder)
    server.run()

    print("MongoDB server running at %s" % server.uri)

    while True:
        time.sleep(1)
