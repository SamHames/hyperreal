# Design of a Hyperreal

Hyperreal is a software tool for computational interpretive analysis of collections of documents.

## Background


## Other Software

https://github.com/alicia-ziying-yang/conTEXT-explorer

Python Libraries:

- gensim

Taguette

Commercial Qualitative Analysis Tools

## Use Cases

## Design Constraints

- This is a research tool first
- Run locally first
- Support batch/bulk workflows for indexing and document management
- needs to be supported/maintained in a very small amount of developer time

Primary Deploy target: a typical analysts laptop

- corpus is an adapter: leave documents where they are, but show describe how hyperreal can structure and access them.
	+ contrast to other systems/ don't form an explicit data of your
	+ documents are Python objects
	+ provide functionality in your corpus describe how to retrieve specific documents, how to extract features from those documents, and also how to render them to different formats for different use cases. If you're using hyperreal as a library you only need to do the first two, the second enable you to do a little bit of work then work with your data via the web/command line interface

Interface layer:

- Progressive enhancement of Server Side Rendered HTML

### Out of Current Scope

To keep the scope of the projec contained, the following items are out of scope:

- Supporting more documents than can fit in a roaring bitmap (4,294,967,296 documents is likely to be enough for most research projects).
- Supporting anything other than Boolean querying.
- User management and authentication.
- A centrally hosted service.
- Incremental or live updates of documents in an index.

## Design Notes

1. Full text search applications

blockdiag {
	orientation = portrait;
	user -> browser -> app -> interactive -> bg_processor -> bg_pool;
	group {
      user; 
      browser;
    }
	"Web Server \n (Starlette)" -> Queue -> "Background Processor";
}


## As a Python Library

## As a CLI

## As a Web Application

## Extending Hyperreal
