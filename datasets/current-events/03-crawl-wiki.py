#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import os

def crawl(website):
    print ("Downloading %s" % (website,))

    headers = {
        "Accept-Language":  "en-US,en;q=0.9",
        "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
        "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection":       "keep-alive",
    }

    req = urllib.request.Request(website, headers=headers)
    with urllib.request.urlopen(req, timeout=60*3) as fp:
        charset = fp.info().get_content_charset()
        if charset is None:
            charset = "utf8"
        html = fp.read()
        try:
            html = html.decode(charset, errors='ignore')
        except LookupError:
            html = html.decode("utf-8", errors='ignore')

    return html

if __name__ == '__main__':
    try:
        os.makedirs("wiki")
    except FileExistsError:
        pass

    with open("events.json", 'r') as fp:
        events = json.load(fp)

    urls = set()
    for event in events:
        for url in event['section_urls'].values():
            if url.startswith("/wiki/"):
                urls.add(url)

    for url in urls:
        assert url.startswith("/wiki/")

        filename = "wiki/%s.html" % (url[6:].replace("/", "_"),)
        if os.path.exists(filename):
            continue

        try:
            html = crawl("https://en.wikipedia.org%s" % (url,))
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            else:
                raise

        with open(filename, 'w') as fp:
            fp.write(html)
