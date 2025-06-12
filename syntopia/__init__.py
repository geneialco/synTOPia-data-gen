"""synTOPia - A toolkit for generating synthetic datasets based on TOPMed data dictionaries."""

__version__ = "0.1.0"

from .parsing import parsers, schema

__all__ = [
    'parsers',
    'schema',
] 