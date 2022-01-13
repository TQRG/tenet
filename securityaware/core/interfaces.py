"""
    Interface for handlers
"""

from cement import Interface


class HandlersInterface(Interface):
    """
        Handlers' Interface
    """
    class Meta:
        """
            Meta class
        """
        interface = 'handlers'
