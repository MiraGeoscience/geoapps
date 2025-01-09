# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from notebook.notebookapp import main as notebook_main


def main():
    """Start the notebook server, opening geoapps/index.ipynb."""

    geoapps_root = Path(importlib.util.find_spec("geoapps").origin).parent
    index_notebook = geoapps_root / "index.ipynb"
    if not index_notebook.is_file():
        logging.getLogger(__package__).error(
            "Could not find index.ipynb (looking in %s)", geoapps_root.resolve()
        )

    notebook_main([str(index_notebook.resolve())])


if __name__ == "__main__":
    main()
