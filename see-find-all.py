import os
from pathlib import Path
import joblib
import json
import numpy as np
import sys
from pprint import pprint

DIR = sys.argv[1]
keyword = sys.argv[2] if len(sys.argv) > 2 else None

for folder in sorted(Path(os.path.abspath(DIR)).iterdir(), key=os.path.getmtime):
    try:
        full_path = os.path.join(DIR, folder, "findings.obj")
        try:
            findings = joblib.load(full_path)
        except KeyError:
            findings = json.load(open(full_path))
        print(folder)
        for i, _findings in enumerate(findings):
            if keyword is None:
                print("Stage {}".format(i))
                pprint(_findings, indent=1, sort_dicts=False)
            else:
                print("Stage {}: {}".format(i, [_finding["statistics"][keyword] for _finding in _findings]))
        # if len(findings) == 2:
        #     for i, _findings in enumerate(findings[0]):
        #         print("stage {}:".format(i), _findings)
        #     print(findings[1])
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        pass
# json.dump(ref_config, open("ref_config.json", "w"), indent=4)

