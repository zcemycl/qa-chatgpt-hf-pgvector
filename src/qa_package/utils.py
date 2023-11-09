import argparse


def parse_args(args: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--article-csv",
        type=str,
        default="/Users/spare/Documents/data/articles.csv",
    )
    p.add_argument("--initialise-embeddings", action="store_true")
    args = p.parse_args(args)
    return args
