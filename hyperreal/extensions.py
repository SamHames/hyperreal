"""
This module constructs a registry of compatible extensions for this library
for use by the command line and web interfaces.

Extension modules for hyperreal are specified using the naming scheme
hyperreal_extensionname. Top level modules matching this naming convention
will automatically be inspected for compatible implementations.

All internal implementations of extensible components will also be patched in
here.

"""

import importlib
import pkgutil

from hyperreal.corpus import PlainTextSqliteCorpus


def load_registry():

    registry = {PlainTextSqliteCorpus.CORPUS_TYPE: PlainTextSqliteCorpus}

    extension_candidates = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith("hyperreal_")
    }

    for name, module in extension_candidates.items():
        try:
            corpus_classes = module.HYPERREAL_CORPUS
            for corpus_class in corpus_classes:
                if isinstance(corpus_class, BaseCorpus):
                    registry[corpus_class.CORPUS_TYPE] = corpus_class
        except AttributeError:
            # TODO: log this
            pass

    return registry


registry = load_registry()
