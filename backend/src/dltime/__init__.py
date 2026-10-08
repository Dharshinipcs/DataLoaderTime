"""dltime - Data Loader execution-time prediction (pre-execution ML pipeline)."""

__version__ = "0.1.0"

from dltime.config import DATA_DIR, PROJECT_ROOT
from dltime.sources import SourceSchemaError, load_all_sources

__all__ = ["DATA_DIR", "PROJECT_ROOT", "SourceSchemaError", "load_all_sources", "__version__"]