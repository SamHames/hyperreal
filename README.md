# Hyperreal

Hyperreal is a Python tool for qualitative analysis of large collections of
documents.

# Development

## Installation

To install the development version:

1. Clone the repository using git.
2. From the cloned repository, use pip for an editable install:
    
    `pip install -e .`

## Running Tests

We use pytest to run tests and tox to coordinate everything. Install the necessary packages and run the tests like so:

```
pip install -e .[test]
# To run just the testsuite
pytest
# To run everything, including code formatting via black and generate coverage
tox

```

# Usage

## Command Line

The following script gives a basic example of using the command line interface
for hyperreal. This will work for cases where you have a plain text file
(here called `corpus.txt`), with each `document` in the collection on its own
line.

```
# Create a corpus database from the plaintext file
hyperreal plaintext-corpus create corpus.txt corpus.db

# Create an index from the corpus
hyperreal plaintext-corpus index corpus.db corpus_index.db

# Create a model from that index, in this case with 128 clusters and 
# only include features present in 10 or more documents.
hyperreal model corpus_index.db --min-docs 10 --clusters 128

# Use the web interface to serve the results of that modelling (currently this
# is very limited)
hyperreal serve corpus_index.db

```

## Library

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

# Create a model on this index, with 256 clusters and only including features
# that match at least 100 documents.
i.initialise_clusters(n_clusters=256, min_docs=100)
# Refine the model for 10 iterations. Note that you can continue to refine
# the model without initialising the clusters.
i.refine_clusters(iterations=10)

# Inspect the output of the model using the index instance (currently quite
# limited). This will print the top 10 most frequent features in each cluster.
for cluster_id in i.cluster_ids:
    cluster_features = i.cluster_features(cluster_id)
    for feature in cluster_features[:10]:
        print(feature)

```

# Extending Hyperreal

Using Hyperreal as a library can be as simple as using an existing corpus to
represent your data, or in more complex cases by writing your own Corpus.
Going even further, it is possible to extend the CLI and web interfaces for
hyperreal as well by defining modules



This package provides a plugin mechanism for customising behaviour
(primarily interfacing with your own documents via an implementation of a
corpus.)

- any top level python module called hyperreal_<> will be treated as a source
  of extensions for hyperreal
- there needs to be a top level attribute of that lists classes that implement
  the different interfaces
