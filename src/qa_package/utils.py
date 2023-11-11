import argparse

# For openai embedding text
pre_encoding_format = """\
product name: {prod_name},\
product type name: {product_type_name},\
product group name: {product_group_name},\
graphical appearance name: {graphical_appearance_name},\
color: {colour_group_name},\
garment group name: {garment_group_name},\
details: {detail_desc}\
"""

# Standard reply for product advice
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


def conversation_loop_info():
    print("[INFO] Chatbot starts...")
    print("[INFO] Type 'mode 1' to start Product Advice Mode.")
    print(
        """[INFO] Type 'mode 2' to start Customer Conversation Mode \
for fashion guidance."""
    )
    print(
        """[INFO] Type 'mode 3' to find similar garments \
based on text and image"""
    )
    print(
        """[INFO] Type 'mode 4' to suggest complementarity garments \
based on text and image"""
    )
    print("[INFO] Type 'exit' to terminate the chatbot.")
    print("[INFO] Type 'restart' to clear chat history.")
    print("[INFO] Current Mode: 'mode 2'.")
