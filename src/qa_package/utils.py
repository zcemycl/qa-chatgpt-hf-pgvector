import argparse
import http
from urllib.request import urlopen

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    p.add_argument("--batch-size", type=int, default=16)  # max: 16
    p.add_argument(
        "--root-image-dir",
        type=str,
        default="/Users/spare/Documents/data/images/",
    )
    p.add_argument("--visualise", action="store_true")
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


def replace_to_fit_ltree(string: str) -> str:
    return (
        string.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_or_")
        .replace("&", "_and_")
    )


def read_plot_images(
    paths: list[str],
    supertitle: str = "",
    url: str = None,
    localpath: str = None,
):
    L = len(paths)
    if L == 1:
        img = mpimg.imread(paths[0])
        plt.imshow(img)
        plt.axis("off")
    else:
        if url is not None or localpath is not None:
            L += 1
        gs = gridspec.GridSpec(L // 3 + int(L % 3 > 0), 3)
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.1, hspace=0)
        norm = plt.Normalize(0, 1)
        if url is not None:
            f = urlopen(url)
            paths += [f]
        elif localpath is not None:
            paths += [localpath]
        for i in range(len(paths)):
            if isinstance(paths[i], str):
                img = mpimg.imread(paths[i])
            elif isinstance(paths[i], http.client.HTTPResponse):
                img = np.array(Image.open(paths[i]))
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.imshow(img, norm=norm)
        ax.set_title("Original", loc="right", y=-0.01)
        for ax in fig.axes:
            ax.axis("off")
        fig.suptitle(supertitle)
    plt.show()
