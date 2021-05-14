import os
from pathlib import Path
import joblib
import json
import numpy as np
import sys

DIR = "./sync-results/stag-hunt-gw/find-all/"

for folder in sorted(Path(os.path.abspath(DIR)).iterdir(), key=os.path.getmtime):
    try:
        full_path = os.path.join(DIR, folder, "findings.obj")
        try:
            findings = joblib.load(full_path)
        except KeyError:
            findings = json.load(open(full_path))
        print(folder)
        for i, _findings in enumerate(findings):
            for _, rewards, gather_count in _findings:
                print("stage {}:".format(i), rewards)
                print(gather_count.tolist())
        # print(findings[1])
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        pass
# json.dump(ref_config, open("ref_config.json", "w"), indent=4)

