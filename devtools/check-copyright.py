#!/usr/bin/env python3

#  Copyright (c) 2022 Mira Geoscience Ltd.
#
# This file is part of geoapps.
#
# geoapps is distributed under the terms and conditions of the MIT License
# (see LICENSE file at the root of this source code package).

import re
import sys
from datetime import date

if __name__ == "__main__":
    current_year = date.today().year
    copyright_re = re.compile(rf"\bcopyright \(c\) \b{current_year}\b", re.IGNORECASE)
    files = sys.argv[1:]
    max_lines = 10
    report_files = []
    for f in files:
        with open(f, encoding="utf-8") as file:
            count = 0
            has_dated_copyright = False
            for line in file:
                count += 1
                if count >= max_lines:
                    break
                if re.search(copyright_re, line):
                    has_dated_copyright = True
                    break

            if not has_dated_copyright:
                report_files.append(f)

    if len(report_files) > 0:
        for f in report_files:
            sys.stderr.write(f"{f}: No copyright or invalid year\n")
        exit(1)

# readonly CURRENT_YEAR=$(date +"%Y")

# if ! grep -e "Copyright (c) .*$CURRENT_YEAR" $(head -10 $f) 2>&1 1>/dev/null; then
#    echo "File '$f' has no copyright or an invalid year"
#    exit 1
# fi
