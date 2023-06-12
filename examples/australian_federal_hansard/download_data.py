"""
# Australian Federal Hansard data preparation script

This script downloads available machine readable data for Australian Federal
Hansard. Note that the data downloaded here is licensed CC-BY-NC-ND.

Specifically this script runs through the following steps:

1. Downloads recent Hansard data, downloading only new files - if transcripts of
new sitting days are available they can be retrieved by rerunning this script.

2. Downloads (if needed) a precompiled dataset for Hansard from 1901-2005. This
only needs to be done once.


# Setup and running this script

You can install the requirements for this script from the commandline:

```
pip install requests

```

You can run this script from it's current directory like so - this will create
the needed folders if they don't already exist and download any missing data.

```
python download_data.py

```

"""

from html.parser import HTMLParser
import os
import time
import webbrowser
import zipfile

import requests


# Part 1: Hansard data 2005-present.


class ExtractXMLLinks(HTMLParser):
    """
    A simple approach from the Python standard library to extract all known proceedings from the source site.

    """

    def __init__(self):
        super().__init__()
        self.xml_links = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value.endswith(".xml"):
                    self.xml_links.add(value)


# Retrieve the index files for the senate and house of representatives, and store them in the zipfile.
hansard_data_url = "http://data.openaustralia.org.au/scrapedxml/"
houses = ["senate_debates/", "representatives_debates/"]

known_proceedings = set()

for house in houses:
    house_index_page = requests.get(hansard_data_url + house)
    parser = ExtractXMLLinks()
    parser.feed(house_index_page.text)
    proceedings = {house + xml_link for xml_link in parser.xml_links}
    known_proceedings |= proceedings

try:
    os.mkdir("data")
except:
    pass

with zipfile.ZipFile(
    os.path.join("data", "hansard_2005-present.zip"),
    "a",
    compression=zipfile.ZIP_DEFLATED,
) as hansard_zip:
    already_downloaded = set(hansard_zip.namelist())
    to_download = known_proceedings - already_downloaded

    if len(to_download):
        print(f"Downloading {len(to_download)} proceedings from 2005-present")
    else:
        print("Already downloaded all proceedings from 2005-present")

    for proceeding in sorted(to_download):
        print(f"Downloading {proceeding}")
        try:
            proceeding_data = requests.get(hansard_data_url + proceeding)
            proceeding_data.raise_for_status()
            hansard_zip.writestr(proceeding, proceeding_data.text)

        except requests.HTTPError:
            print(f"Failed to download {proceeding}, continuing on")
        time.sleep(2)


# Part 2: Historical Hansard data via the Glam Workbench precompiled dataset.
temp_data_file = os.path.join("data", "hansard_1901-1980_1998-2005.zip.temp")
data_file = os.path.join("data", "hansard_1901-1980_1998-2005.zip")

if not os.path.exists(data_file):
    print("Downloading Hansard 1901-1980/1998-2005 dataset.")
    data_url = "https://github.com/wragge/hansard-xml/archive/master.zip"

    response = requests.get(data_url, stream=True)
    with open(temp_data_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=10 * 2**20):
            f.write(chunk)

    os.rename(temp_data_file, data_file)

else:
    print("Already downloaded historical Hansard dataset.")
