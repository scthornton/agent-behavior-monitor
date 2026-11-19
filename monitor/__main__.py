import argparse
from pathlib import Path

from .airs import MODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=["generate", "train", "evaluate", "robustness", "demo", "serve"],
    )
    ap.add_argument("--out", type=Path, default=Path("run"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seeds", type=int, default=12, help="training seeds to compare")
    ap.add_argument(
        "--airs",
        choices=MODES,
        default="offline",
    )
    args = ap.parse_args()
    if args.command == "generate":
        from .fixtures import generate

        generate(args.out)
    elif args.command in ("train", "evaluate"):
        from .experiment import evaluate, train

        (train if args.command == "train" else evaluate)(args.out)
    elif args.command == "robustness":
        from .experiment import robustness

        robustness(args.out, args.seeds)
    elif args.command == "serve":
        import uvicorn

        from .service import create_app

        uvicorn.run(create_app(args.out), host="127.0.0.1", port=args.port)
    else:
        from .demo import demo

        demo(args.out, args.airs)


if __name__ == "__main__":
    main()
