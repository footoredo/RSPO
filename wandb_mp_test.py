import wandb
import multiprocessing
import time
import numpy as np


class Process(multiprocessing.Process):
    def __init__(self, id, q):
        super(Process, self).__init__()
        self.id = id
        self.q = q

    def run(self):
        for i in range(10):
            self.q.put({f"i": i,
                        f"process-{self.id}/y": np.power(i, self.id + 2),
                        f"process-{self.id}/fff": [i + 1, i + 2, i + 3]})
            # print(i)
            time.sleep(0.1)
        self.q.put(None)


run = wandb.init(project="diversity-escalation-gw", entity="footoredo", group="mp-test")

q = multiprocessing.Queue()

processes = []
for z in range(2):
    p = Process(z, q)
    p.start()
    processes.append(p)

finish_cnt = 0
while finish_cnt < 2:
    data = q.get()
    # print(data)
    if data is None:
        finish_cnt += 1
    else:
        # pass
        wandb.log(data)

for p in processes:
    p.join()
