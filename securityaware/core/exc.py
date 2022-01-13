
class SecurityAwareError(Exception):
    """Generic errors."""
    pass


class SecurityAwareWarning(Exception):
    """Generic warning."""
    pass


class CommandError(Exception):
    """Command error"""
    pass


class Skip(Exception):
    """Skip node execution exception"""
    pass
