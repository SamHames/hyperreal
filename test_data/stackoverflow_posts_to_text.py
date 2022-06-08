import csv
from io import StringIO
from html.parser import HTMLParser
import sys
import xml.etree.ElementTree as ET


class MLStripper(HTMLParser):
    """Via https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python"""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


if __name__ == "__main__":
    print(sys.argv)
    xml_file = sys.argv[1]
    output_file = sys.argv[2]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(output_file, "w", encoding="UTF-8") as outfile:
        csvout = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        for row in root:
            body = row.attrib["Body"]
            body_text = strip_tags(body)
            csvout.writerow([body_text])
