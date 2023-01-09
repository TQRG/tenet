
class TenetError(Exception):
    """Generic errors."""
    pass


class TenetWarning(Exception):
    """Generic warning."""
    pass


class CommandError(Exception):
    """Command error"""
    pass


class Skip(Exception):
    """Skip node execution exception"""
    pass


class Found(Exception):
    """Found exception to break nested loops"""
    pass
