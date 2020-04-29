from pathlib import Path
from types import ModuleType
from typing import Dict

import thriftpy2

_INTERFACES_PATH = Path("interfaces")
_INTERFACES: Dict[str, ModuleType] = {}


def __getattr__(name):
    try:
        return _INTERFACES[name]
    except KeyError:
        interface = thriftpy2.load(str(_INTERFACES_PATH.joinpath(f"{name}.thrift")))
        _INTERFACES[name] = interface
        return interface
