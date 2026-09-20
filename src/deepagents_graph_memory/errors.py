# - Gives graph problems clear names so the rest of the package can handle them.
# - Tests: test_paths.py and test_validation.py check errors for bad paths and unsafe data;
#        test_optional_vgs_dependency.py checks the missing-LadybugDB message.

"""Error types for graph memory."""


class GraphMemoryError(Exception):
    """Base exception for graph memory errors."""


class GraphMemoryConfigurationError(GraphMemoryError):
    """Raised when graph memory is not configured correctly."""


class GraphMemoryPathError(GraphMemoryError):
    """Raised when a virtual graph path is invalid."""


class GraphMemoryValidationError(GraphMemoryError):
    """Raised when a graph write payload fails validation."""
