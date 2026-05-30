"""Error types for graph memory."""


class GraphMemoryError(Exception):
    """Base exception for graph memory errors."""


class GraphMemoryConfigurationError(GraphMemoryError):
    """Raised when graph memory is not configured correctly."""


class GraphMemoryPathError(GraphMemoryError):
    """Raised when a virtual graph path is invalid."""


class GraphMemoryValidationError(GraphMemoryError):
    """Raised when a graph write payload fails validation."""
