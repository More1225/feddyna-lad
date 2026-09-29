# coding: utf-8
"""
Federated-learning method registry

Provide a unified mechanism for registering and retrieving methods.
"""

from typing import Callable, Dict, Optional, Type


class MethodRegistry:
    """Federated-learning method registry"""

    _methods: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a method"""
        def decorator(method_class: Type) -> Type:
            cls._methods[name.lower()] = method_class
            return method_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a method class"""
        return cls._methods.get(name.lower())

    @classmethod
    def list_methods(cls) -> list:
        """List all registered methods"""
        return list(cls._methods.keys())


def register_method(name: str) -> Callable:
    """Decorator to register a federated-learning method"""
    return MethodRegistry.register(name)


def get_method(name: str, args, model, device):
    """
    Get a federated-learning method instance

    Args:
        name: Method name (fedavg, fedprox, ditto, fedpac, etc.)
        args: Argument configuration
        model: Model instance
        device: Compute device

    Returns:
        Method instance, or None if the method is unavailable
    """
    method_class = MethodRegistry.get(name)
    if method_class is None:
        raise ValueError(f"Unknown FL method: {name}. Available: {MethodRegistry.list_methods()}")

    return method_class(args, model, device)
