# Hyperreal

Hyperreal is a Python tool for interactive qualitative analysis of large
collections of documents.


## Requirements

Hyperreal requires the installation of [the Python programming language](https://www.python.org/downloads/).


## Installation

Hyperreal can be installed using Pip from the command line (
[Windows](https://learn.openwaterfoundation.org/owf-learn-windows-shell/introduction/introduction/#windows-command-shell),
[Mac](https://support.apple.com/en-au/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac))
by running the following commands:

```
python -m pip install hyperreal
```

## Usage

Hyperreal can be used in three different ways to flexibly support different
use cases:

- as a command line application
- as a Python library
- via the built in web application

All of hyperreal's functionality is available from the Python library, but you
will need to write Python code to use it directly. The command line interface
allows for quick and repeatable experimentation and automation for standard
data types - for example if you often work with Twitter data the command line
will allow you to rapidly work with many different Twitter data collections.
The web application is currently focused solely on creating and interactive
editing of models.


### Command Line

The following script gives a basic example of using the command line interface
for hyperreal. This will work for cases where you have a plain text file
(here called `corpus.txt`), with each `document` in the collection on its own
line.

If you haven't worked with the command line before, you might find the
following resources useful:

- [Software Carpentry resources for Mac](https://swcarpentry.github.io/shell-novice/)
- [Open Water Foundation resources for Windows](https://learn.openwaterfoundation.org/owf-learn-windows-shell/)

```
# Create a corpus database from the plaintext file
hyperreal plaintext-corpus create corpus.txt corpus.db

# Create an index from the corpus
hyperreal plaintext-corpus index corpus.db corpus_index.db

# Create a model from that index, in this case with 128 clusters and
# only include features present in 10 or more documents.
hyperreal model corpus_index.db --min-docs 10 --clusters 128

# Use the web interface to serve the results of that modelling
# After running this command point your web browser to http://localhost:8080
hyperreal plaintext-corpus serve corpus.db corpus_index.db

```

### Library

This example script performs the same steps as the command line example.


``` python

from hyperreal import corpus, index

# create and populate the corpus with some documents
c = corpus.PlainTextSqliteCorpus('corpus.db')

with open('corpus.txt', 'r') as f:
  # This will drop any line that has no text (such as a paragraph break)
  docs = (line for line in f if line.strip())
  c.replace_docs(docs)


# Index that corpus - note that we need to pass the corpus object for
# initialisation.
i = index.Index('corpus_index.db', corpus=c)
# This only needs to be done once, unless the corpus changes.
i.index()

# Create a model on this index, with 128 clusters and only including features
# that match at least 10 documents.
i.initialise_clusters(n_clusters=128, min_docs=10)
# Refine the model for 10 iterations. Note that you can continue to refine
# the model without initialising the clusters.
i.refine_clusters(iterations=10)

# Inspect the output of the model using the index instance (currently quite
# limited). This will print the top 10 most frequent features in each
# cluster.
for cluster_id in i.cluster_ids:
    cluster_features = i.cluster_features(cluster_id)
    for feature in cluster_features[:10]:
        print(feature)

# Perform a boolean query on the corpus, looking for documents that contain
# both apples AND oranges in the text field.
q = i[('text', 'oranges')] & i[('text', 'oranges')]
# Lookup all of the documents in the corpus that match this query.
docs = i.get_docs(q)

# 'Pivot' the features in the index with respect to all cluster in the model.
#  This will show the top 10 features in each cluster that are similar to the
#  query.
i.pivot_clusters_by_query(query, top_k=10)

# This will show the top 10 features for a selected set of cluster_ids.
i.pivot_clusters_by_query(query, cluster_ids=[3,5,7], top_k=10)

```


## Development

### Installation

To install the development version:

1. Clone the repository using git.
2. From the cloned repository, use pip for an editable install:

    `pip install -e .`

### Running Tests

The full test suite and other checks are orchestrated via tox:

```
python -m pip install -e .[test]

# To run just the testsuite
pytest

# To run everything, including code formatting via black and check coverage
tox

```
