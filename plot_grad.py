import numpy as np
import joblib

from a2c_ppo_acktr.multi_agent.utils import tsne


def main():
    fgs_5, adv_5 = joblib.load("grad-5.obj")
    fgs_10, adv_10 = joblib.load("grad-10.obj")
    fgs_15, adv_15 = joblib.load("grad-15.obj")

    tsne(fgs_5 + fgs_10 + fgs_15, ["g-5"] * len(fgs_5) + ["g-10"] * len(fgs_10) + ["g-15"] * len(fgs_15))


if __name__ == "__main__":
    main()
