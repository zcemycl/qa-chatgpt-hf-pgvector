import argparse
import os
import sys
import time
from typing import Callable, Tuple

import pandas as pd
import PIL
from dotenv import load_dotenv
from sqlalchemy import func
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from sqlalchemy_utils import Ltree
from tqdm import tqdm

from .dataclasses.orm import color, garment, metadata, pattern, record
from .services.guardrails import guard_image_search
from .services.huggingface import HuggingFace
from .services.openai import OpenAI
from .utils import (
    conversation_loop_info,
    parse_args,
    post_reply,
    pre_encoding_format,
    read_plot_images,
    replace_to_fit_ltree,
)

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
db_url = "postgresql://postgres:postgres@localhost/postgres"
max_suggestions = 5

set_groups = {
    "garment upper body": "set1",
    "garment lower body": "set1",
    "garment full body": "set2",
    "swimwear": "set3",
    "nightwear": "set4",
}


class Chatbot:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        df = pd.read_csv(config.article_csv)
        self.df = df.apply(lambda x: x.astype(str).str.lower())
        print(config)

    def initialise_embeddings(
        self,
        sess: Session,
    ):
        # restart table
        metadata.drop_all(bind=self.engine)
        metadata.create_all(bind=self.engine)
        sess.query(record).delete()
        sess.query(color).delete()
        sess.query(garment).delete()
        sess.commit()

        # embeddings in batches
        print("[INFO] Documents encoding...")
        BATCH = self.df.shape[0] // self.config.batch_size + int(
            self.df.shape[0] % self.config.batch_size > 0
        )
        for i in tqdm(range(BATCH)):
            ids = self.df["article_id"][
                i * self.config.batch_size : (i + 1) * self.config.batch_size
            ].tolist()
            docs = [
                pre_encoding_format.format(**tmp)
                for tmp in self.df.iloc[
                    i
                    * self.config.batch_size : (i + 1)
                    * self.config.batch_size
                ].to_dict(orient="records")
            ]
            vecs = self.client.create_embeddings(
                docs, EMBEDDING_DEPLOYMENT_NAME
            )

            for tmpid, tmpvec in zip(ids, vecs):
                tmprow = record(id=tmpid, factors=tmpvec)
                sess.add(tmprow)
            sess.commit()
            # avoid hitting max limit of openai api calls
            time.sleep(5)

        # color embeddings
        print("[INFO] Colors encoding...")
        unique_colors = self.df.colour_group_name.unique().tolist()
        BATCH = len(unique_colors) // self.config.batch_size + int(
            len(unique_colors) % self.config.batch_size > 0
        )
        for i in tqdm(range(BATCH)):
            docs = unique_colors[
                i * self.config.batch_size : (i + 1) * self.config.batch_size
            ]
            vecs = self.client.create_embeddings(
                docs, EMBEDDING_DEPLOYMENT_NAME
            )
            for c, v in zip(docs, vecs):
                sess.add(color(name=c.lower(), factors=v))
            sess.commit()
            if i % 30 == 0:
                time.sleep(10)

        # pattern embeddings
        print("[INFO] Patterns encoding...")
        unique_patterns = self.df.graphical_appearance_name.unique().tolist()
        BATCH = len(unique_patterns) // self.config.batch_size + int(
            len(unique_patterns) % self.config.batch_size > 0
        )
        for i in tqdm(range(BATCH)):
            docs = unique_patterns[
                i * self.config.batch_size : (i + 1) * self.config.batch_size
            ]
            vecs = self.client.create_embeddings(
                docs, EMBEDDING_DEPLOYMENT_NAME
            )
            for c, v in zip(docs, vecs):
                sess.add(pattern(name=c.lower(), factors=v))
            sess.commit()
            if i % 30 == 0:
                time.sleep(10)

        # garment embeddings + ltree
        print("[INFO] Garments encoding...")
        unique_groups = self.df.product_group_name.unique()
        for i in range(5):
            sess.add(
                garment(
                    name=f"set{i+1}",
                    factors=[0] * 1536,
                    path=Ltree(f"set{i+1}"),
                )
            )
        sess.commit()
        for gp in unique_groups:
            set_name = set_groups[gp] if gp in set_groups else "set5"
            vecgp = self.client.create_embeddings(
                [gp], EMBEDDING_DEPLOYMENT_NAME
            )
            rootname = replace_to_fit_ltree(gp)
            sess.add(
                garment(
                    name=gp,
                    factors=vecgp[0],
                    path=Ltree(f"{set_name}.{rootname}"),
                )
            )
            sess.commit()
            unique_garments = list(
                self.df[
                    self.df.product_group_name == gp
                ].product_type_name.unique()
            )
            vecs = []
            BATCH = len(unique_garments) // self.config.batch_size + int(
                len(unique_garments) % self.config.batch_size > 0
            )
            for i in tqdm(range(BATCH)):
                docs = unique_garments[
                    i
                    * self.config.batch_size : (i + 1)
                    * self.config.batch_size
                ]
                vecs += self.client.create_embeddings(
                    docs, EMBEDDING_DEPLOYMENT_NAME
                )
            for ga, v in zip(unique_garments, vecs):
                childname = replace_to_fit_ltree(ga)
                sess.add(
                    garment(
                        name=ga,
                        factors=list(v),
                        path=Ltree(f"{set_name}.{rootname}.{childname}"),
                    )
                )
            sess.commit()
            time.sleep(5)

    def product_advice(
        self, user_input: str, session: Session
    ) -> Tuple[str, dict[str, str]]:
        # embed user input
        vec = self.client.create_embeddings(
            [user_input], EMBEDDING_DEPLOYMENT_NAME
        )[0]
        # find best answer (product) from vector database
        res = (
            session.execute(
                select(record.id)
                .order_by(record.factors.cosine_distance(vec))
                .limit(1)
            )
            .scalars()
            .all()[0]
        )
        tmp_row = self.df[self.df.article_id == str(res)].to_dict(
            orient="records"
        )[0]

        # ask chatgpt to rewrite product description
        dict_resp = self.client.advice_product(
            {"question": user_input, **tmp_row}, CHAT_DEPLOYMENT_NAME
        )
        reply = post_reply.format(**dict_resp)
        return reply, tmp_row

    def image_search(
        self, func: Callable, user_input: str, session: Session
    ) -> int:
        try:
            _, validated_output = guard_image_search(
                self.client.openai.ChatCompletion.create,
                prompt_params={"question": user_input},
                deployment_id=CHAT_DEPLOYMENT_NAME,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                max_tokens=1024,
                temperature=0.3,
            )
            url_or_path = None
            if validated_output["url"] != "None":
                url_or_path = validated_output["url"]
            if validated_output["path"] != "None":
                url_or_path = validated_output["path"]
            caption = self.hf.captioner(url_or_path)[0]["generated_text"]
            print("Assistant: ", caption)
            vec = self.client.create_embeddings(
                [caption], EMBEDDING_DEPLOYMENT_NAME
            )[0]
            ids = func(vec, session)
        except PIL.UnidentifiedImageError:
            print(
                "Assistant: I am sorry, ",
                "I cannot see any image from link or path.",
            )
            return 1
        except ValueError:
            print(
                "Assistant: I am sorry, ",
                "I cannot see any image from link or path.",
            )
            return 1
        tmp_rows = self.df[
            self.df.article_id.isin([str(i) for i in ids])
        ].to_dict(orient="records")

        imgpaths = []
        for row in tmp_rows:
            print(
                "Assistant: product id -- ",
                row["article_id"],
            )
            imgpaths.append(
                self.config.root_image_dir + f"0{row['article_id']}.jpg"
            )
        if self.config.visualise:
            if validated_output["url"] == "None":
                validated_output["url"] = None
            if validated_output["path"] == "None":
                validated_output["path"] = None
            read_plot_images(
                imgpaths,
                url=validated_output["url"],
                localpath=validated_output["path"],
                supertitle=func.__name__.replace("_", " "),
            )
        return 0

    def find_similar_garments_with_image(
        self, vec: list[float], session: Session
    ) -> list[str]:
        res = (
            session.execute(
                select(record.id)
                .order_by(record.factors.cosine_distance(vec))
                .limit(5)
            )
            .scalars()
            .all()
        )
        return res

    def suggest_complementarity(
        self, vec: list[float], session: Session
    ) -> list[str]:
        cres = (
            session.execute(
                select(color.name)
                .order_by(color.factors.cosine_distance(vec))
                .limit(5)
            )
            .scalars()
            .all()
        )
        print("Colors: ", cres)
        approx_records = (
            session.execute(
                select(record.id)
                .order_by(record.factors.cosine_distance(vec))
                .limit(5)
            )
            .scalars()
            .all()
        )
        df_records = self.df[
            self.df.article_id.isin([str(i) for i in approx_records])
        ]
        target_group = df_records.product_group_name.to_list()[0]
        print(df_records)
        l2group = session.query(garment).filter_by(name=target_group).scalar()
        print(l2group.name)
        l2siblings = (
            session.query(garment.name)
            .filter(
                garment.path.descendant_of(l2group.path[:-1]),
                func.nlevel(garment.path) == 2,
                garment.id != l2group.id,
            )
            .all()
        )
        l2siblings = [i[0] for i in l2siblings]
        print(l2siblings)
        if len(l2siblings) == 0:
            # search in set 5
            l3_s5_types = (
                session.query(garment)
                .filter(
                    garment.path.descendant_of(Ltree("set5")),
                    func.nlevel(garment.path) == 3,
                )
                .all()
            )
            print([i.name for i in l3_s5_types])
            dftmp = self.df[
                (self.df.colour_group_name.isin(cres))
                & self.df.product_type_name.isin([i.name for i in l3_s5_types])
            ]
        else:
            l2_s5_types = (
                session.query(garment)
                .filter(
                    garment.path.descendant_of(Ltree("set5")),
                    func.nlevel(garment.path) == 2,
                )
                .all()
            )
            l2_all = [i.name for i in l2_s5_types] + l2siblings
            dftmp = self.df[
                (self.df.colour_group_name.isin(cres))
                & (self.df.product_group_name.isin(l2_all))
            ]
        print(dftmp)
        stmt = (
            select(record.id)
            .where(record.id.in_(dftmp.article_id.to_list()))
            .order_by(record.factors.cosine_distance(vec))
            .limit(max_suggestions)
        )
        res_in_order = session.execute(stmt).scalars().all()
        return res_in_order

    def conversation_loop(self, session: Session):
        init_messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        mode = "mode 2"
        messages = init_messages.copy()

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            elif (
                user_input.lower() == "mode 1"
                or user_input.lower() == "mode 2"
                or user_input.lower() == "mode 3"
                or user_input.lower() == "mode 4"
            ):
                # mode switching, skip adding to messages
                mode = user_input.lower()
                print(f"[INFO] Current Mode: '{mode}'.")
                continue
            elif user_input.lower() == "restart":
                # restart dialogues
                mode = "mode 2"
                messages = init_messages.copy()
                print("[INFO] Restarting...")
                print("[INFO] Current Mode: 'mode 2'.")
                continue
            elif user_input.lower() == "info":
                # print instructions for client
                conversation_loop_info()
                continue

            messages.append({"role": "user", "content": user_input})

            if mode == "mode 1":
                reply, tmp_row = self.product_advice(user_input, session)
                print(f"Assistant: {reply}")
                messages.append({"role": "assistant", "content": reply})

                if self.config.visualise:
                    read_plot_images(
                        [
                            self.config.root_image_dir
                            + f"0{tmp_row['article_id']}.jpg"
                        ],
                        supertitle="product advice",
                    )

            elif mode == "mode 2":
                # chatgpt conversation with client
                response = self.client.chat(
                    messages=messages,
                    chat_deployment_name=CHAT_DEPLOYMENT_NAME,
                )
                print(f"Assistant: {response}")
                messages.append({"role": "assistant", "content": response})

            elif mode == "mode 3":
                if self.image_search(
                    self.find_similar_garments_with_image, user_input, session
                ):
                    continue

            elif mode == "mode 4":
                if self.image_search(
                    self.suggest_complementarity, user_input, session
                ):
                    continue

    def __call__(self):
        # initialisation
        # - openai client wrapper
        # - postgres connection
        # - article dataframe
        self.client = OpenAI(
            api_key=API_KEY, api_base=API_BASE, api_version=API_VERSION
        )
        self.engine = create_engine(db_url)
        self.hf = HuggingFace()

        with Session(self.engine) as sess:
            # embeddings of all articles if not exist
            if self.config.initialise_embeddings:
                self.initialise_embeddings(sess)

            # chatbot loop
            conversation_loop_info()
            self.conversation_loop(sess)

        self.engine.dispose()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # print(args)
    # main(args)
    main = Chatbot(args)
    main()
