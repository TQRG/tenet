from cement import Controller, ex

from securityaware.handlers.cwe_list import CWEListHandler


class CWE(Controller):
    """
        Plugin Controller to handle plugin operations
    """

    class Meta:
        label = 'cwe'
        stacked_on = 'base'
        stacked_type = 'nested'

    def __init__(self, **kw):
        super().__init__(**kw)
        self._cwe_list_handler: CWEListHandler = None

    @property
    def cwe_list_handler(self):
        if not self._cwe_list_handler:
            self._cwe_list_handler = self.app.handler.get('handlers', 'cwe_list', setup=True)
        return self._cwe_list_handler

    @cwe_list_handler.deleter
    def cwe_list_handler(self):
        self._cwe_list_handler = None

    @ex(
        help="Prints available abstractions"
    )
    def abstractions(self):
        for p in self.cwe_list_handler.mappings.keys():
            self.app.log.info(p)

    @ex(
        help='Finds SFP cluster for a given CWE-ID',
        arguments=[
            (['--id'], {'help': 'CWE-ID', 'type': int, 'required': True})
        ]
    )
    def sfp_cluster(self):
        self.app.log.info(self.cwe_list_handler.find_sfp_cluster(self.app.pargs.id))
