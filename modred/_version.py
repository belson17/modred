"""
Module where the version is written.

It is executed in setup.py and imported in modred/__init__.py.

See:

http://en.wikipedia.org/wiki/Software_versioning
http://legacy.python.org/dev/peps/pep-0386/

'a' or 'alpha' means alpha version (internal testing),
'b' or 'beta' means beta version (external testing).

Append with .postN for post-release updates.  This may be necessary for changes
in packaging or documentation, as PyPI does not allow uploads of files with the
same filename (which corresponds to the version number).
"""
__version__ = '2.0.4.post5'
