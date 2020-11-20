from ma_main import main
from a2c_ppo_acktr.arguments import get_args
import multiprocessing as mp
import logging


def test_main():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.WARNING)
    args = get_args()

    datas = []
    # for i in range(-1, args.num_env_steps // (args.num_steps * args.num_processes) + 1):
    for i in range(-1, 0):
    # for i in [-1]:
    #     print("i:", i)
        args.reseed_step = i * args.num_steps * args.num_processes
        for z in range(20):
            args.reseed_z = z + 1

            close_to = main(args, logger)
            print(i, z, close_to)
            datas.append((i, z, close_to))

    import joblib
    joblib.dump(datas, "{}-{}.data".format(args.test_branching_name, args.seed))


if __name__ == "__main__":
    test_main()
