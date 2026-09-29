"""
FedDyna custom exception classes

Provide a hierarchy of exception types for error handling and diagnostics.
"""


class FedDynaError(Exception):
    """Base exception for the FedDyna framework"""
    pass


class ConfigurationError(FedDynaError):
    """Configuration error (argument validation, configuration parsing, etc.)"""
    pass


class ValidationError(ConfigurationError, ValueError):
    """Argument-validation error (compatible with existing ValueError handlers)"""
    pass


class AlgorithmError(FedDynaError):
    """Algorithm execution error"""
    pass


class ModelError(FedDynaError):
    """Model error (loading, saving, architecture, etc.)"""
    pass


class DataError(FedDynaError):
    """Data error (loading, preprocessing, partitioning, etc.)"""
    pass


class TrainingError(FedDynaError):
    """Training error"""
    pass


class AggregationError(FedDynaError):
    """Federated aggregation error"""
    pass


class PersonalizationError(FedDynaError):
    """Personalization error"""
    pass


class InvalidParameterError(ValidationError):
    """Invalid-argument error"""
    pass


class UnsupportedOperationError(FedDynaError):
    """Unsupported-operation error"""
    pass