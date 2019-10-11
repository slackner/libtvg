import calendar
import re

month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
               'august', 'september', 'october', 'november', 'december']

s_weekday   = "(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
s_prefix    = "(?:[0-5]?[0-9]:[0-5][0-9]|%s)" % (s_weekday,)
s_day       = "(0?[1-9]|[12][0-9]|3[01])"
s_month     = "(%s)" % ("|".join(month_names),)
s_year      = "([12][0-9]{3})"
s_present   = "(present|ongoing|now)"

re_date     = re.compile("^([0-9]{4})-[0-9]{2}-[0-9]{2}$")
re_comment  = re.compile("\\([^)]*\\)")
re_d_dmy    = re.compile("^%s\\s*-\\s*%s\\s+%s(?:\\s+%s)?$" % (s_day, s_day, s_month, s_year))
re_md_dy    = re.compile("^%s\\s+%s\\s*-\\s*%s(?:\\s+%s)?$" % (s_month, s_day, s_day, s_year))
re_dmy      = re.compile("^(?:%s\\s+)?%s\\s+%s(?:\\s+%s(?:\\s*;\\s+[0-9]+\\s+(days|months|years)\\s+ago)?)?$" % (s_prefix, s_day, s_month, s_year))
re_mdy      = re.compile("^(?:%s\\s+)?%s\\s+%s(?:\\s+%s(?:\\s*;\\s+[0-9]+\\s+(days|months|years)\\s+ago)?)?$" % (s_prefix, s_month, s_day, s_year))
re_my       = re.compile("^%s(?:\\s+%s)?$" % (s_month, s_year))
re_y        = re.compile("^%s$" % (s_year,))
re_p        = re.compile("^%s$" % (s_present,))

def parse_date_field(text):
    """Parse a single date field and return a tuple with (day, month, year)."""

    # 1 april 2015
    m = re_dmy.match(text)
    if m is not None:
        day   = int(m.group(1))
        month = month_names.index(m.group(2)) + 1
        year  = int(m.group(3)) if m.group(3) is not None else None
        return (year, month, day)

    # april 1 2015
    m = re_mdy.match(text)
    if m is not None:
        month = month_names.index(m.group(1)) + 1
        day   = int(m.group(2))
        year  = int(m.group(3)) if m.group(3) is not None else None
        return (year, month, day)

    # april 2015
    m = re_my.match(text)
    if m is not None:
        month = month_names.index(m.group(1)) + 1
        year  = int(m.group(2)) if m.group(2) is not None else None
        return (year, month, None)

    # 2015
    m = re_y.match(text)
    if m is not None:
        year = int(m.group(1))
        return (year, None, None)

    # present
    m = re_p.match(text)
    if m is not None:
        return (-1, -1, -1)

    raise ValueError

def parse_date_range(text, ref_year=None):
    """Parse a date range."""

    fields = text.split("-")
    if len(fields) != 2:
        raise ValueError

    start_year, start_month, start_day = parse_date_field(fields[0].strip())
    end_year, end_month, end_day = parse_date_field(fields[1].strip())

    if start_year is None:
        start_year = end_year
    if start_year is None:
        start_year = ref_year
        end_year   = ref_year
    if start_year is None:
        raise ValueError
    if start_year < 0:
        raise ValueError
    if start_month is None:
        start_month = 1
    if start_day is None:
        start_day = 1

    if end_year is None:
        raise ValueError
    if end_year < 0:
        return ["%04d-%02d-%02d" % (start_year, start_month, start_day), "NOW"]
    if end_month is None:
        end_month = 12
    if end_day is None:
        end_day = calendar.monthrange(end_year, end_month)[1]

    return ["%04d-%02d-%02d" % (start_year, start_month, start_day),
            "%04d-%02d-%02d" % (end_year, end_month, end_day)]

def parse_date_single(text, ref_year=None):
    """Parse a single date."""

    start_year, start_month, start_day = parse_date_field(text)
    end_year, end_month, end_day = start_year, start_month, start_day

    if start_year is None:
        start_year = ref_year
        end_year   = ref_year
    if start_year is None:
        raise ValueError
    if start_year < 0:
        raise ValueError
    if start_month is None:
        start_month = 1
        end_month = 12
    if start_day is None:
        start_day = 1
        end_day = calendar.monthrange(end_year, end_month)[1]

    return ["%04d-%02d-%02d" % (start_year, start_month, start_day),
            "%04d-%02d-%02d" % (end_year, end_month, end_day)]

def parse_date_special(text, ref_year=None):
    """Parse special date formats."""

    # 1 - 12 november 2016
    m = re_d_dmy.match(text)
    if m is not None:
        start_day   = int(m.group(1))
        end_day     = int(m.group(2))
        month       = month_names.index(m.group(3)) + 1
        if m.group(4) is not None:
            year    = int(m.group(4))
        elif ref_year is not None:
            year    = ref_year
        else:
            raise ValueError
        return ["%04d-%02d-%02d" % (year, month, start_day),
                "%04d-%02d-%02d" % (year, month, end_day)]

    # april 21 - 22 2016
    m = re_md_dy.match(text)
    if m is not None:
        month       = month_names.index(m.group(1)) + 1
        start_day   = int(m.group(2))
        end_day     = int(m.group(3))
        if m.group(4) is not None:
            year    = int(m.group(4))
        elif ref_year is not None:
            year    = ref_year
        else:
            raise ValueError
        return ["%04d-%02d-%02d" % (year, month, start_day),
                "%04d-%02d-%02d" % (year, month, end_day)]

    raise ValueError

def parse_date(text, ref_date=None):
    text = text.replace("\xa0", " ")
    text = re_comment.sub("", text).strip()
    text = text.lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace(",", " ")

    if text.startswith("total:"):
        text = text[6:]

    lines = text.split("\n")
    i = 1
    while i < len(lines) and "".join(lines[:i]).strip().endswith("-"):
        i += 1
    text = "".join(lines[:i]).strip()

    ref_year = None
    if ref_date is not None:
        m = re_date.match(ref_date)
        if m is not None:
            ref_year = int(m.group(1))

    try:
        return parse_date_range(text, ref_year=ref_year)
    except ValueError:
        pass

    try:
        return parse_date_single(text, ref_year=ref_year)
    except ValueError:
        pass

    try:
        return parse_date_special(text, ref_year=ref_year)
    except ValueError:
        pass

    print ("Unhandled text %s" % (repr(text),))
    return None

if __name__ == '__main__':
    import unittest

    class TestDateParser(unittest.TestCase):
        def test_comment(self):
            result = parse_date("1 april 2015 (really!) - 12 november 2016 (exactly)")
            self.assertEqual(result, ['2015-04-01', '2016-11-12'])

        def test_multiline(self):
            result = parse_date("1 april 2015\nother stuff")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("1 april 2015 -\nother stuff")
            self.assertEqual(result, None)

            result = parse_date("1 april 2015 -\n\n\n12 november 2016")
            self.assertEqual(result, ['2015-04-01', '2016-11-12'])

        def test_range(self):
            result = parse_date("1 april 2015 - 12 november 2016")
            self.assertEqual(result, ['2015-04-01', '2016-11-12'])

            result = parse_date("april 1 2015 - november 12 2016")
            self.assertEqual(result, ['2015-04-01', '2016-11-12'])

            result = parse_date("april 1, 2015 - november 12, 2016")
            self.assertEqual(result, ['2015-04-01', '2016-11-12'])

            result = parse_date("1 april - 12 november 2016")
            self.assertEqual(result, ['2016-04-01', '2016-11-12'])

            result = parse_date("april 1 - november 12 2016")
            self.assertEqual(result, ['2016-04-01', '2016-11-12'])

            result = parse_date("april 1 - november 12, 2016")
            self.assertEqual(result, ['2016-04-01', '2016-11-12'])

            result = parse_date("1 april 2015 - present")
            self.assertEqual(result, ['2015-04-01', 'NOW'])

            result = parse_date("april 1 2015 - present")
            self.assertEqual(result, ['2015-04-01', 'NOW'])

            result = parse_date("april 1, 2015 - present")
            self.assertEqual(result, ['2015-04-01', 'NOW'])

            result = parse_date("30 december 1998 - present")
            self.assertEqual(result, ['1998-12-30', 'NOW'])

            result = parse_date("april 2015 - november 2016")
            self.assertEqual(result, ['2015-04-01', '2016-11-30'])

            result = parse_date("april - november 2016")
            self.assertEqual(result, ['2016-04-01', '2016-11-30'])

            result = parse_date("2002 - 2019")
            self.assertEqual(result, ['2002-01-01', '2019-12-31'])

            result = parse_date("2002 - present")
            self.assertEqual(result, ['2002-01-01', 'NOW'])

            result = parse_date("january 2018 - ongoing")
            self.assertEqual(result, ['2018-01-01', 'NOW'])

            result = parse_date("5 august 2016 - october 2016")
            self.assertEqual(result, ['2016-08-05', '2016-10-31'])

            result = parse_date("5 august - october 2016")
            self.assertEqual(result, ['2016-08-05', '2016-10-31'])

            result = parse_date("january, 2018 - february, 2018")
            self.assertEqual(result, ['2018-01-01', '2018-02-28'])

            result = parse_date("april 1 - april 5")
            self.assertEqual(result, None)

            result = parse_date("april 1 - april 5", ref_date="2017-01-01")
            self.assertEqual(result, ['2017-04-01', '2017-04-05'])

            result = parse_date("april - may")
            self.assertEqual(result, None)

            result = parse_date("april - may", ref_date="2017-01-01")
            self.assertEqual(result, ['2017-04-01', '2017-05-31'])

        def test_single(self):
            result = parse_date("1 april 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("april 1 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("april 1, 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("monday 1 april 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("monday, 1 april 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("monday april 1 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("monday, april 1 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("1:00 1 april 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("12:00, 1 april 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("1:00 april 1 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("12:00, april 1 2015")
            self.assertEqual(result, ['2015-04-01', '2015-04-01'])

            result = parse_date("26 april 1986; 33 years ago")
            self.assertEqual(result, ['1986-04-26', '1986-04-26'])

            result = parse_date("april 1")
            self.assertEqual(result, None)

            result = parse_date("april 1", ref_date="2017-01-01")
            self.assertEqual(result, ['2017-04-01', '2017-04-01'])

            result = parse_date("april")
            self.assertEqual(result, None)

            result = parse_date("april", ref_date="2017-01-01")
            self.assertEqual(result, ['2017-04-01', '2017-04-30'])

        def test_special(self):
            result = parse_date("1 - 12 november 2016")
            self.assertEqual(result, ['2016-11-01', '2016-11-12'])

            result = parse_date("april 21-22 2016")
            self.assertEqual(result, ['2016-04-21', '2016-04-22'])

            result = parse_date("april 21-22, 2016")
            self.assertEqual(result, ['2016-04-21', '2016-04-22'])

            result = parse_date("april 1-5")
            self.assertEqual(result, None)

            result = parse_date("april 1-5", ref_date="2017-01-01")
            self.assertEqual(result, ['2017-04-01', '2017-04-05'])


    unittest.main()
