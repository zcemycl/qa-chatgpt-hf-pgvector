import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from tqdm import tqdm

from .dataclasses.orm import metadata, record
from .services.openai import OpenAI
from .utils import (
    conversation_loop_info,
    parse_args,
    post_reply,
    pre_encoding_format,
)

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
BATCH_SIZE = 5
db_url = "postgresql://postgres:postgres@localhost/postgres"


def initialise_embeddings(
    df: pd.DataFrame, client: OpenAI, engine: Engine, sess: Session
):
    print("[INFO] Documents encoding...")
    df = df.apply(lambda x: x.astype(str).str.lower())
    BATCH = df.shape[0] // BATCH_SIZE + int(df.shape[0] % BATCH_SIZE > 0)

    # restart table
    metadata.drop_all(bind=engine)
    metadata.create_all(bind=engine)
    sess.query(record).delete()
    sess.commit()

    # embeddings in batches
    for i in tqdm(range(BATCH)):
        ids = df["article_id"][i * BATCH_SIZE : (i + 1) * BATCH_SIZE].tolist()
        docs = [
            pre_encoding_format.format(**tmp)
            for tmp in df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE].to_dict(
                orient="records"
            )
        ]
        vecs = client.create_embeddings(docs, EMBEDDING_DEPLOYMENT_NAME)

        for tmpid, tmpvec in zip(ids, vecs):
            tmprow = record(id=tmpid, factors=tmpvec)
            sess.add(tmprow)
        sess.commit()
        # avoid hitting max limit of openai api calls
        time.sleep(5)


def conversation_loop(df: pd.DataFrame, client: OpenAI, session: Session):
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
            # embed user input
            vec = client.create_embeddings(
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
            tmp_row = df[df.article_id == res].to_dict(orient="records")[0]

            # ask chatgpt to rewrite product description
            dict_resp = client.advice_product(
                {"question": user_input, **tmp_row}, CHAT_DEPLOYMENT_NAME
            )
            reply = post_reply.format(**dict_resp)
            print(f"Assistant: {reply}")
            messages.append({"role": "assistant", "content": reply})

        elif mode == "mode 2":
            # chatgpt conversation with client
            response = client.chat(
                messages=messages,
                chat_deployment_name=CHAT_DEPLOYMENT_NAME,
            )
            print(f"Assistant: {response}")
            messages.append({"role": "assistant", "content": response})

        elif mode == "mode 3":
            pass

        elif mode == "mode 4":
            pass


def main(config: argparse.Namespace):
    # initialisation
    # - openai client wrapper
    # - postgres connection
    # - article dataframe
    client = OpenAI(
        api_key=API_KEY, api_base=API_BASE, api_version=API_VERSION
    )
    engine = create_engine(db_url)
    df = pd.read_csv(config.article_csv)

    with Session(engine) as sess:
        # embeddings of all articles if not exist
        if config.initialise_embeddings:
            initialise_embeddings(df, client, engine, sess)

        # chatbot loop
        conversation_loop_info()
        conversation_loop(df, client, sess)

    engine.dispose()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)
