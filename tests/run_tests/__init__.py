#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    processes=False,
    # n_workers=int(multiprocessing.cpu_count() / 2) - 1,
    # threads_per_worker=1,
)
Client(cluster)
