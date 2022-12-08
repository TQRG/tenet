import re
from typing import Union

from securityaware.core.interfaces import HandlersInterface
from cement import Handler


class CWEListHandler(HandlersInterface, Handler):
    """
        CWE handler for abstraction
    """
    class Meta:
        label = 'cwe_list'

    def __init__(self, **kw):
        super().__init__(**kw)
        self._sfp_primary_entries = None
        self._sfp_primary_clusters = None
        self._sfp_secondary_entries = None
        self._sfp_none_entries = None
        self._spk_category_abstractions = None
        self._sw_category_abstractions = None
        self._sfp_category_abstractions = None
        self._listing = None
        self._bugs_framework = None

    def has_sfp(self, cwe_id: int) -> bool:
        return cwe_id in self.sfp_primary_entries or cwe_id in self.sfp_secondary_entries or \
               cwe_id in self.sfp_none_entries or cwe_id in self.sfp_primary_clusters

    def get_sfp_title(self, cwe_id: int) -> str:
        if not cwe_id:
            return "Unknown"

        elif cwe_id in self.sfp_primary_entries:
            return self.sfp_primary_entries[cwe_id]['title']
        elif cwe_id in self.sfp_primary_clusters:
            return self.sfp_primary_clusters[cwe_id]['title']
        elif cwe_id in self.sfp_secondary_entries:
            return self.sfp_secondary_entries[cwe_id]['title']
        elif cwe_id in self.sfp_none_entries:
            return self.sfp_none_entries[cwe_id]['title']
        else:
            return "Unk"

    @property
    def mappings(self) -> dict:
        return self.app.config.get_section_dict('mappings')

    @property
    def abstractions(self) -> dict:
        return self.app.config.get_section_dict('abstractions')

    @property
    def categories(self) -> dict:
        return self.abstractions['categories']

    @property
    def deprecated(self) -> dict:
        return self.abstractions['deprecated']

    @property
    def listing(self) -> list:
        if self._listing is None:
            self._listing = self.abstractions['list']
        return self._listing

    @property
    def spk_category_abstractions(self):
        if not self._spk_category_abstractions:
            self._spk_category_abstractions = self.categories['seven_kingdoms']
        return self._spk_category_abstractions

    @property
    def sw_category_abstractions(self):
        if not self._sw_category_abstractions:
            self._sw_category_abstractions = self.categories['software_development']
        return self._sw_category_abstractions

    @property
    def sfp_category_abstractions(self):
        if not self._sfp_category_abstractions:
            self._sfp_category_abstractions = self.categories['software_fault_pattern']
        return self._sfp_category_abstractions

    @property
    def software_fault_pattern(self):
        return self.mappings['software_fault_pattern']

    @property
    def software_development(self):
        return self.mappings['software_development']

    @property
    def sfp_primary_entries(self):
        if self._sfp_primary_entries is None:
            primary_entries = self.software_fault_pattern['primary']
            self._sfp_primary_entries = {_id: el for _id, el in primary_entries.items() if 'entries' in el}
        return self._sfp_primary_entries

    @property
    def sfp_primary_clusters(self):
        if self._sfp_primary_clusters is None:
            primary_clusters = self.software_fault_pattern['primary']
            self._sfp_primary_clusters = {_id: el for _id, el in primary_clusters.items() if 'clusters' in el}
        return self._sfp_primary_clusters

    @property
    def sfp_primary_ids(self):
        return list(self.sfp_primary_entries.keys()) + list(self.sfp_primary_clusters.keys())

    @property
    def sfp_secondary_entries(self):
        if self._sfp_secondary_entries is None:
            secondary_entries = self.software_fault_pattern['secondary']
            self._sfp_secondary_entries = {_id: el for _id, el in secondary_entries.items() if 'entries' in el}
        return self._sfp_secondary_entries

    @property
    def sfp_none_entries(self):
        if self._sfp_none_entries is None:
            none_entries = self.software_fault_pattern['none']
            self._sfp_none_entries = {_id: el for _id, el in none_entries.items() if 'entries' in el}
        return self._sfp_none_entries

    @property
    def bugs_framework(self):
        if self._bugs_framework is None:
            self._bugs_framework = self.mappings['bugs_framework']
        return self._bugs_framework

    @property
    def data_type_bugs(self):
        return self.bugs_framework['data_type']

    def parse_cwe_id(self, cwe_id: Union[int, str]):
        if isinstance(cwe_id, str):
            match = re.search('\d+', cwe_id)

            if not match:
                self.app.log.warning(f"Could not find integers in {cwe_id}")
                return None

            cwe_id = int(match.group())

        return cwe_id

    def find_bf_class(self, cwe_id: Union[int, str], category: str = None) -> Union[str, None]:
        cwe_id = self.parse_cwe_id(cwe_id)

        if cwe_id is None:
            return None

        if category and category in self.bugs_framework:
            categories = [self.bugs_framework[category]]
        else:
            categories = self.bugs_framework.values()

        for target in categories:
            for bf_class, operations in target.items():
                for operation, entries in operations.items():
                    if cwe_id in entries:
                        return f"{bf_class} {operation}"

        return None

    def find_primary_sfp_cluster(self, cwe_id: Union[int, str], only_id: bool = False):
        cwe_id = self.parse_cwe_id(cwe_id)

        if cwe_id is None:
            return None

        for _c, cluster in self.sfp_primary_entries.items():
            if cwe_id in cluster['entries']:
                return str(_c) if only_id else f"CWE-{_c}: {cluster['title']}"

        for _c, cluster in self.sfp_secondary_entries.items():
            if cwe_id in cluster['entries']:
                # return _c + ': ' + cluster['title']
                for _c1, cluster1 in self.sfp_primary_clusters.items():
                    if _c in cluster1['clusters']:
                        return str(_c1) if only_id else f"CWE-{_c1}: {cluster1['title']}"

        #for _c, cluster in self.sfp_none_entries.items():
        #    if cwe_id in cluster['entries']:
        #        return f"CWE-{_c}: {cluster['title']}"

    def find_sw_category(self, cwe_id: int):
        for _c, cluster in self.software_development.items():
            if cwe_id in cluster['entries']:
                return f"CWE-{_c}: {cluster['title']}"
        # return 'CWE without SW Cluster'

    def find_abstractions(self, cwes_counts: dict, filter_abst: list, print_most_common: bool = False):
        mapping = {}

        for c in cwes_counts.keys():
            if c in self.listing:
                mapping[c] = self.listing[c]
            elif c in self.deprecated:
                mapping[c] = f"Deprecated {self.deprecated[c]}"
            elif c in self.sw_category_abstractions:
                mapping[c] = "Software Development View Category"
            elif c in self.sfp_category_abstractions:
                mapping[c] = "Software Fault Pattern View Category"
            elif c in self.spk_category_abstractions:
                mapping[c] = "7 Pernicious Kingdoms View Category"
            else:
                self.app.log.warning(f"{c} not in abstractions")

        if print_most_common:
            most_common_abst = {}

            for cwe, abst in mapping.items():
                if abst not in most_common_abst:
                    most_common_abst[abst] = cwes_counts[cwe]
                else:
                    most_common_abst[abst] += cwes_counts[cwe]

            self.app.log.info(most_common_abst)

        if filter_abst:
            return {_id: _type for _id, _type in mapping.items() if _type in filter_abst}

        return mapping
