from pathlib import Path
from typing import Union

import pandas as pd
import jq
import json

from securityaware.handlers.plugin import PluginHandler


class CodeQLExtractLabelsHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "codeql_extract_labels"

    def run(self, node: dict, cell: dict, dataset: pd.DataFrame, files_path: Path,
            report: str = "") -> Union[pd.DataFrame, None]:
        """
            extracts the labels from the CodeQL's reports
        """

        report_file = cell['path'] / report

        if not report_file.exists():
            return None

        results_path = cell['path'] / 'results'

        if not results_path.exists():
            results_path.mkdir()

        with report_file.open(mode='r') as rf:
            json_report = json.loads(rf.read())
            result = jq.all(r'.runs[0].results', json_report)
            dataset = []

            for i, data_point in enumerate(jq.iter(r'.[]', result[0])):
                self.app.log.info(f"Processing data point {i}")
                vuln_entry = jq.all(r'.', data_point)[0]
                fpath = jq.first(r'.locations[0].physicalLocation.artifactLocation.uri', vuln_entry)

                if fpath.endswith('.js'):
                    label = jq.first(r'.ruleId', vuln_entry)
                    sline = jq.first(r'.locations[0].physicalLocation.region.startLine', vuln_entry)
                    scol = jq.first(r'.locations[0].physicalLocation.region.startColumn', vuln_entry)
                    eline = jq.first(r'.locations[0].physicalLocation.region.endLine', vuln_entry)
                    ecol = jq.first(r'.locations[0].physicalLocation.region.endColumn', vuln_entry)

                    if eline is None or eline == "null":
                        eline = sline

                    project = fpath.split('/')[0]
                    fpath = fpath.replace(f"{project}/", '')
                    dataset.append([project, fpath, sline, scol, eline, ecol, label])

            if dataset:
                return pd.DataFrame(dataset, columns=['project', 'fpath', 'sline', 'scol', 'eline', 'ecol', 'label'])

        return None

