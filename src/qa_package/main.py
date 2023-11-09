import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--article_csv",
        type=str,
        default="/Users/spare/Documents/data/articles.csv",
    )
    args = p.parse_args()

    print(args)
