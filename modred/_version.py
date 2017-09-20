"""
Module where the version is written.

It is executed in setup.py and imported in modred/__init__.py.

See:

http://en.wikipedia.org/wiki/Software_versioning
http://legacy.python.org/dev/peps/pep-0386/

'a' or 'alpha' means alpha version (internal testing),
'b' or 'beta' means beta version (external testing).

Use -revN (-rev1, -rev2, etc.) to denote post-release revisions having to do
with packaging alone.  This may be necessary as PyPI no longer allows file
uploads that share the same name.  As such, any changes to a package requires a
change in the version number.
"""
__version__ = '2.0.4-rev1'
