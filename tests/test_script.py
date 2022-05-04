from hyperreal import corpus, index

if __name__ == "__main__":
    c = corpus.CirrusSearchWikiCorpus("test_data/wiki_test.db")
    print("ingesting")
    # c.ingest("test_data/simplewiki-20200323-cirrussearch-content.json.gz")

    i = index.Index("test_data/wiki_index.db", c)

    print("indexing")
    # i.index()

    print("initialising")
    i.initialise_clusters(144, min_docs=10)

    print("refining")
    i.refine_clusters(iterations=5)
