#!/usr/bin/env python3

import unittest
import datetime
import sys

import news_pipeline as news

class TestEntityPreprocessing(unittest.TestCase):

    def setUp(self):
        pass

    # test_parse_date
    def test_parse_date_format_simple(self):
        # Input
        publishing_date = "2019-04-15"
        crawltime = ""
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = datetime.datetime(2019, 4, 15, 0, 0)
        self.assertEqual(result, expected)

    def test_parse_date_format_utc_offset(self):
        # Input
        publishing_date = "2019-04-15T17:15:38+02:00"
        crawltime = ""
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = datetime.datetime(2019, 4, 15, 17, 15, 38, tzinfo=datetime.timezone(datetime.timedelta(0, 7200)))
        self.assertEqual(result, expected)

    def test_parse_date_format_utc_no_offset(self):
        # Input
        publishing_date = "2019-05-23T11:13:00Z"
        crawltime = ""
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = datetime.datetime(2019, 5, 23, 11, 13, tzinfo=datetime.timezone.utc)
        self.assertEqual(result, expected)

    def test_parse_date_format_utc_millisec_offset(self):
        # Input
        publishing_date = "2019-05-22T03:33:14.929+02:00"
        crawltime = ""
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = datetime.datetime(2019, 5, 22, 3, 33, 14, 929000, tzinfo=datetime.timezone(datetime.timedelta(0, 7200)))
        self.assertEqual(result, expected)

    def test_parse_date_format_utc_millisec_no_offset(self):
        # Input
        publishing_date = "2019-05-22T03:33:14.929Z"
        crawltime = ""
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = datetime.datetime(2019, 5, 22, 3, 33, 14, 929000, tzinfo=datetime.timezone.utc)
        self.assertEqual(result, expected)

    def test_parse_date_format_invalid(self):
        # Input
        publishing_date = "2019-05-Invalid"
        crawltime = datetime.datetime(2019, 1, 1)
        # Result
        result = news.parse_date_format(publishing_date, crawltime)
        # Expected
        expected = crawltime
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
