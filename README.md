# Hyperreal

Hyperreal is a Python tool for qualitative analysis of large collections of
documents.



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
