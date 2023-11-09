import argparse

pre_encoding_format = """\
product name: {prod_name},\
product type name: {product_type_name},\
product group name: {product_group_name},\
graphical appearance name: {graphical_appearance_name},\
color: {colour_group_name},\
garment group name: {garment_group_name},\
details: {detail_desc}\
"""

post_reply = """\
{answer} \n
Here is the product id: {product_id}\
"""


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
