#!/usr/bin/env python3
from bs4 import BeautifulSoup
import dateparser
import functools
import json
import re
import os

def is_parent(tag, parent):
    tag = tag.parent
    while tag != parent:
        assert tag is not None
        if tag.name == "li":
            return False
        tag = tag.parent
    return True

@functools.lru_cache(maxsize=8192)
def extract_daterange(url, ref_date=None):
    if not url.startswith("/wiki/"):
        return None

    filename = "wiki/%s.html" % (url[6:].replace("/", "_"),)
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as fp:
        html = fp.read()

    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {'class': "infobox"})
    if not table:
        return None

    for row in table.find_all("tr"):
        columns = row.find_all(["th", "td"])
        if len(columns) != 2:
            continue
        if columns[0].get_text().strip().lower() == "date":
            break
    else:
        return None

    element = columns[1]
    for child in reversed(element.find_all("sup")):
        child.extract()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    return dateparser.parse_date(element.get_text(), ref_date=ref_date)

def extract_element(element, category=None, ref_date=None):
    children = element.find_all("li")
    children = [child for child in children if is_parent(child, element)]

    # No nested enumeration found - this must be a News entry.
    if len(children) == 0:
        text = element.get_text().strip()
        urls = [(a.get_text().strip(), a['href']) for a in element.find_all("a")]
        return [{'text':        text,
                 'urls':        dict(urls),
                 'section':     [],
                 'category':    category,
                 'date':        ref_date,
                 'text_urls':   dict(urls),
                 'section_urls': dict(),
                 'section_dates': dict()}]

    # Handle all nested elements.
    content = []
    for child in children:
        content += extract_element(child, category=category, ref_date=ref_date)
    for child in reversed(children):
        child.extract()

    # Add inherited attributes from the parent element.
    text = element.get_text().strip()
    urls = [(a.get_text().strip(), a['href']) for a in element.find_all("a")]

    dates = {}
    for title, url in urls:
        date = extract_daterange(url, ref_date=ref_date)
        if date is None:
            continue
        dates[title] = date

    for obj in content:
        obj['urls'].update(urls)
        obj['section'].append(text)
        obj['section_urls'].update(urls)
        obj['section_dates'].update(dates)

    return content

def extract_content(html):
    soup = BeautifulSoup(html, "html.parser")

    content = []
    for table in soup.find_all(["table", "div"], {'class': 'vevent'}):

        # Extract date from the table header
        span = table.find("span", {'class': "summary"})
        assert span is not None
        m = re.match("^.*\\(([0-9]{4}-[0-9]{2}-[0-9]{2})\\).*$", span.get_text())
        assert m is not None
        ref_date = m.group(1)

        children = table.find_all(["dl", "div", "li"])
        children = [child for child in children if is_parent(child, table)]

        # Handle all listed categories and elements
        category = None
        for child in children:
            if child.name == "dl":
                category = child.get_text().strip()
                continue
            if child.name == "div":
                if child.get('role', None) == "heading":
                    category = child.get_text().strip()
                continue

            content += extract_element(child, category=category, ref_date=ref_date)

    return content

if __name__ == '__main__':
    content = []
    for filename in sorted(os.listdir("events")):
        if not filename.endswith(".html"):
            continue

        print ("Processing %s" % (filename,))

        filename = os.path.join("events", filename)
        with open(filename, 'r') as fp:
            html = fp.read()

        content += extract_content(html)

    content = sorted(content, key=lambda obj: obj['date'])
    with open("events.json", 'w', encoding='utf8') as fp:
        json.dump(content, fp, indent=2, ensure_ascii=False)
