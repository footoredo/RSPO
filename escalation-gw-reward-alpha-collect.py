import os
from pathlib import Path
import joblib
import json
import numpy as np
import sys

DIR = f"./sync-results/escalation-gw/"

data = dict(i_alpha=[], reward=[])

for i in range(1, 10):
    dir = os.path.join(DIR, "1-prediction-alpha-0.{}".format(i))
    for folder in sorted(Path(os.path.abspath(dir)).iterdir(), key=os.path.getmtime):
        try:
            # full_path = os.path.join(DIR, folder, "gather_count.obj")
            # gather_count = joblib.load(full_path)
            full_path = os.path.join(dir, folder, "play_statistics.obj")
            statistics = joblib.load(full_path)
            data['i_alpha'].append(i)
            data['reward'].append(sum(statistics["rewards"]) / 2)
        except FileNotFoundError:
            pass

json.dump(data, open("data.json", "w"))
