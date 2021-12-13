from pathlib import Path
from typing import AnyStr, List, Dict, Union

import pandas as pd
from cement import Handler

from securityaware.core.interfaces import HandlersInterface
from securityaware.data.schema import Cell, Layer, Plugin, Container


class PrepareHandler(HandlersInterface, Handler):
    class Meta:
        label = 'prepare'

    def lookup_dataset(self, path: Path, keyword: str = ""):
        """
            Loads the dataset

            :return: pandas dataframe
        """
        for f in path.iterdir():
            if f.is_file and keyword in f.name and f.suffix == '.csv':
                return f, pd.read_csv(str(f))

        self.app.log.error("raw dataset not found")
        exit(1)

    def load_context(self, path: Path) -> dict:
        """
            Loads all the related paths to the node

            :param path: root path to the cell
            :return: dictionary with paths for each cell
        """
        node = {}

        if not path.exists():
            path.mkdir()
            return node

        for d in path.iterdir():
            if not d.is_dir():
                continue

            self.app.log.info(f"Loading cell {d.name}")
            node[d.name] = {'path': d}
            # TODO: include correct datasets, and add the layer as well

        return node

    def __call__(self, layers: Dict[str, Layer], root: Path):
        node = self.load_context(root.parent)

        for name, layer in layers.items():
            if name not in node:
                node[name] = {}
            self.app.log.info(f"Running layer {name}")

            dataset_path, dataframe = self.lookup_dataset(node[layer.input]['path'], layer.input)

            for cell in layer.cells:

                if cell.name not in node[name]:
                    node[name][cell.name] = {}

                if 'path' not in node[name][cell.name]:
                    node[name][cell.name]['path'] = self.app.workdir / name

                    if not node[name][cell.name]['path'].exists():
                        node[name][cell.name]['path'].mkdir()

                if isinstance(cell, Plugin):
                    # TODO: handle None return
                    dataframe = self.run_plugin(node=node, layer=name, cell=cell, data_path=dataset_path, data=dataframe)
                elif isinstance(cell, Container):
                    self.run_container(node=node, layer=name, cell=cell)

    def run_container(self, node: dict, layer: str, cell: Container):
        """
            Runs container

            :param node: node with the context information
            :param layer: name of the layer
            :param cell: cell plugin to run
        """
        container_handler = self.app.handler.get('handlers', 'container', setup=True)
        container = container_handler.get(cell.name)

        if not container:
            self.app.log.warning(f"Container {cell.name} not found.")

            # TODO: Fix this, folder set to the layer name (astminer container inside layer gets codeql fodler)
            container_handler.create(cell.image, cell.name, bind=str(node[layer][cell.name]['path']))
            container = container_handler.get(cell.name)

            if not container:
                self.app.log.warning(f"Container {cell.name} created but not found.")
                return None

        container_handler.start(cell.name)
        container_handler.run_cmds(container.id, cell.cmds)
        self.app.log.info("Done")

    def run_plugin(self, node: dict, layer: str, cell: Plugin, data_path: Path,
                   data: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
            Runs plugin

            :param node: node with the context information
            :param layer: name of the layer
            :param cell: cell plugin to run
            :param data_path: path to the dataset
            :param data: dataframe
        """
        output_path = node[layer][cell.name]['path'] / cell.label

        if not output_path.exists():
            output_path.mkdir()

        cell_output = output_path / f"{data_path.stem}.{cell.name}.csv"

        if cell_output.exists() and not cell.force:
            self.app.log.info(f"{cell.name}: dataset {cell_output} exists.")
            return pd.read_csv(cell_output)

        plugin_handler = self.app.get_plugin_handler(cell.label)
        dataset = plugin_handler.run(node[layer], node[layer][cell.name], data,
                                     data_path.parent / 'files', **cell.args)

        if dataset is not None:
            node[layer][cell.name]['data'] = cell_output
            dataset.to_csv(str(cell_output), index=False)

        return dataset
