import json

escalation_config = {
  "share_reward": False,
  "shape_reward": False,
  "shape_beta": 0.8,
  "defect_coef": -0.9,
  "symmetry_plan": None
}

for i in range(10):
    escalation_config["defect_coef"] = - i / 10
    json.dump(escalation_config, open(f"./env-configs/escalation-gw-rr/-0.{i}.json", "w"))