import argparse

from config import SimConfig
from experiments import run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafeScale-MATD3 experiment suite"
    )
    parser.add_argument(
        "--nn",
        action="store_true",
        default=False,
        help="Use PyTorch+CUDA neural-network TD3 policies (requires torch)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Quick smoke-test: fewer seeds/episodes (n_episodes=40, n_seeds=2, eval_episodes=10)",
    )
    args = parser.parse_args()

    if args.fast:
        cfg = SimConfig(n_episodes=100, n_seeds=2, eval_episodes=20, episode_slots=60)
    else:
        cfg = SimConfig()

    run_all(cfg, use_nn=args.nn)


if __name__ == "__main__":
    main()
