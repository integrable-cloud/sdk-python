"""The package version, in one place.

`__init__` re-exports it and `_client` builds the User-Agent from it, so the
two cannot disagree. pyproject.toml must match it too; a test checks that,
since the release workflow publishes whatever pyproject.toml says.
"""

__version__ = "0.2.0"
