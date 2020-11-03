from ma_main import main
from a2c_ppo_acktr.arguments import get_args
import multiprocessing as mp
import logging


def test_main():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.WARNING)
    args = get_args()

    datas = []
    for i in range(args.num_env_steps // (args.num_steps * args.num_processes) + 1):
        args.reseed_step = i * args.num_steps * args.num_processes
        for z in range(5):
            args.reseed_z = z + 1

            close_to = main(args, logger)
            print(i, z, close_to)
            datas.append((i, z, close_to))

    import joblib
    joblib.dump(datas, "test_branching.data")


if __name__ == "__main__":
    test_main()