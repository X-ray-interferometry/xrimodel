try:  # written by setuptools_scm at build time
    from ._version import version
except ImportError:  # pragma: no cover - running from a plain source tree
    version = "0.1.0.dev0"
