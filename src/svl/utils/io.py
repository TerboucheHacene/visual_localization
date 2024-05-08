import json
from abc import ABC, abstractmethod
from typing import Dict

import yaml


class Parser(ABC):
    """Abstract class for parsers."""

    @staticmethod
    @abstractmethod
    def load(file: str) -> Dict:
        """Load data from a file.

        Parameters
        ----------
        file : str
            file path

        Returns
        -------
        Dict
            data
        """
        pass

    @staticmethod
    @abstractmethod
    def dump(data: Dict, file: str) -> None:
        """Dump data to a file.

        Parameters
        ----------
        data : Dict
            data to dump
        file : str
            file path
        """
        pass


class YamlParser(Parser):
    """YAML parser."""

    @staticmethod
    def load_yaml(yaml_file: str) -> Dict:
        with open(yaml_file, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    @staticmethod
    def dump_yaml(data: Dict, yaml_file: str) -> None:
        with open(yaml_file, "w") as file:
            yaml.dump(data, file)


class JsonParser(Parser):
    """JSON parser."""

    @staticmethod
    def load_json(json_file: str) -> Dict:
        with open(json_file, "r") as file:
            return json.load(file)

    @staticmethod
    def dump_json(data: Dict, json_file: str) -> None:
        with open(json_file, "w") as file:
            json.dump(data, file)
