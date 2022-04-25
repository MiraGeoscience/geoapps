#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import warnings


def soft_import(package, objects=None, interrupt=False):

    packagename = package.split(".")[0]
    packagename = "gdal" if packagename == "osgeo" else packagename
    err = (
        f"Module '{packagename}' is missing from the environment. "
        f"Consider installing with: 'conda install -c conda-forge {packagename}'"
    )

    try:
        imports = __import__(package, fromlist=objects)
        if objects is not None:
            imports = [getattr(imports, o) for o in objects]
            return imports[0] if len(imports) == 1 else imports
        else:
            return imports

    except ModuleNotFoundError:
        if interrupt:
            raise ModuleNotFoundError(err)
        else:
            warnings.warn(err)
            if objects is None:
                return None
            else:
                n_obj = len(objects)
                return [None] * n_obj if n_obj > 1 else None
