import pandas as pd
from cement import Handler

from tenet.core.exc import TenetError
from tenet.core.interfaces import HandlersInterface
from tenet.core.sampling.scenario import Fix, Controlled, Random, Scenario


class SamplingHandler(HandlersInterface, Handler):
    """
        Plugin handler abstraction
    """
    class Meta:
        label = 'sampling'

    def __init__(self, **kw):
        super().__init__(**kw)
        self._sec_keywords = []

    @property
    def sec_keywords(self):
        if not self._sec_keywords:
            self._sec_keywords = self.app.config.get_section_dict('keywords')['security']

        return self._sec_keywords

    def get_scenario(self, df: pd.DataFrame, scenario: str, negative: pd.DataFrame = None,
                     extension: list = None) -> Scenario:
        scenario = scenario.lower()

        if negative is not None:
            self.app.log.info(f"count of negative samples {len(negative)}")

        if scenario == 'fix':
            return Fix(df, extensions=extension)
        elif scenario == 'random':
            return Random(df, negative=negative)
        elif scenario == 'controlled':
            return Controlled(df, negative=negative, keywords=self.sec_keywords)
        else:
            raise TenetError(f"Scenario '{scenario}' not found. "
                                     f"Should be one of the following: [fix, random, controlled]")
