"""Keep local test runs isolated from globally installed pytest plugins."""

from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
