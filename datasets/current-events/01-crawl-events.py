#!/usr/bin/python3
import urllib.request
import urllib.error
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
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    try:
        os.makedirs("events")
    except FileExistsError:
        pass

    for year in range(2016, 2020):
        for month in month_names:
            filename = "events/%s_%d.html" % (month, year)
            if os.path.exists(filename):
                continue

            try:
                html = crawl("https://en.wikipedia.org/wiki/Portal:Current_events/%s_%d" % (month, year))
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    continue
                else:
                    raise

            with open(filename, 'w') as fp:
                fp.write(html)
