from pathlib import Path

import pandas as pd
from typing import Union

from tenet.handlers.plugin import PluginHandler


class MatchPathContextsHandler(PluginHandler):
    """
        Separate plugin
    """

    class Meta:
        label = "match_path_contexts"

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if not self.get('path_contexts_file'):
            self.app.log.warning(f"Path contexts file not instantiated.")
            return None

        if not Path(self.get('path_contexts_file')).exists():
            self.app.log.warning(f"Path contexts file not found.")
            return None

        if not self.get('raw_files_path'):
            self.app.log.warning(f"Raw files path not instantiated.")
            return None

        if not Path(self.get('raw_files_path')).exists():
            self.app.log.warning(f"Raw files path file not found.")
            return None

        if not self.get('raw_fn_bounds_file'):
            self.app.log.warning(f"Raw function bounds file not instantiated.")
            return None

        if not Path(self.get('raw_fn_bounds_file')).exists():
            self.app.log.warning(f"Raw function bounds file not found.")
            return None

        path_ctxs_out = Path(self.path, Path(self.get('path_contexts_file').name))
        self.set('path_contexts_out', path_ctxs_out)

        fn_bounds = self.parse_fn_bounds_dataset()
        path_ctxs = self.parse_path_contexts()

        self.app.log.info(f"#Functions: ", len(fn_bounds))
        self.app.log.info(f"#Path contexts: ", len(path_ctxs))
        merged = path_ctxs.merge(fn_bounds, how='inner', on=['path', 'sline', 'scol', 'eline', 'ecol'])

        with path_ctxs_out.open(mode='w') as f:
            f.write('\n'.join(merged.raw.tolist()))

        merged.drop(columns=['raw'], inplace=True)

        return merged

    def parse_fn_bounds_dataset(self) -> pd.DataFrame:
        data_fn = pd.read_csv(self.get('raw_fn_bounds_file'), index_col=False)
        path = []
        fsize = []
        oline = []
        labels = []

        for i, r in data_fn.iterrows():
            path.append(f"{r.project}/{r.fpath}")

            if r.label != 'safe':
                labels.append('unsafe')
            else:
                labels.append('safe')

            if (r.eline - r.sline) == 0:
                oline.append(True)
                # fsize.append(r.ecol - r.scol)
                self.app.log.info('Reading', path[i])
                self.app.log.info(f"{r.sline}, {r.scol}, {r.eline}, {r.ecol}")

                with Path(self.get('raw_files_path'), path[i]).open(mode='r') as f:
                    lines = f.readlines()
                    multi_line_func = lines[r.sline - 1][r.scol:r.ecol].split(';')
                    self.app.log.info(len(multi_line_func))
                    fsize.append(len(multi_line_func))
            else:
                oline.append(False)
                fsize.append(r.eline - r.sline)

        data_fn['oline'] = oline
        data_fn['path'] = path
        data_fn['fsize'] = fsize
        data_fn['label'] = labels
        data_fn.drop(columns=['project', 'fpath', 'func_id'], inplace=True)

        return data_fn.drop_duplicates()

    def parse_path_contexts(self):
        data = []
        path_ctxs = open(self.get('path_contexts_file')).read().splitlines()
        raw_files_path_container = Path(str(self.get('raw_files_path')).replace(str(self.app.workdir), str(self.app.bind)))

        for line in path_ctxs:
            parts = line.rstrip('\n').split(' ')
            raw_path = Path(parts[0].replace(str(raw_files_path_container) + '/', ''))
            path = f"{raw_path.parent}/{raw_path.stem}.js"
            sline, scol, eline, ecol = raw_path.suffix.split('_')[1:]
            dict_row = {'path': path, 'hash': parts[2], 'label': parts[1], 'cs': len(parts[2:]),
                        'sline': int(sline), 'scol': int(scol), 'eline': int(eline), 'ecol': int(ecol),
                        'raw': line}

            data.append(dict_row)

        return pd.DataFrame(data)


def load(app):
    app.handler.register(MatchPathContextsHandler)
