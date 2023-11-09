import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from tqdm import tqdm

from .dataclasses.orm import metadata, record
from .services.openai import OpenAI
from .utils import parse_args, post_reply, pre_encoding_format

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
BATCH_SIZE = 5
db_url = "postgresql://postgres:postgres@localhost/postgres"


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
        elif user_input.lower() == "mode 1" or user_input.lower() == "mode 2":
            mode = user_input.lower()
            print(f"[INFO] Current Mode: '{mode}'.")
            continue
        elif user_input.lower() == "restart":
            mode = "mode 2"
            messages = init_messages.copy()
            print("[INFO] Restarting...")
            print("[INFO] Current Mode: 'mode 2'.")
            continue

        messages.append({"role": "user", "content": user_input})

        if mode == "mode 1":
            vec = client.create_embeddings(
                [user_input], EMBEDDING_DEPLOYMENT_NAME
            )[0]
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

            dict_resp = client.advice_product(
                {"question": user_input, **tmp_row}, CHAT_DEPLOYMENT_NAME
            )
            reply = post_reply.format(**dict_resp)
            print(f"Assistant: {reply}")
            messages.append({"role": "assistant", "content": reply})

        elif mode == "mode 2":
            response = client.chat(
                messages=messages,
                chat_deployment_name=CHAT_DEPLOYMENT_NAME,
            )
            print(f"Assistant: {response}")
            messages.append({"role": "assistant", "content": response})


def main(config: argparse.Namespace):
    client = OpenAI(
        api_key=API_KEY, api_base=API_BASE, api_version=API_VERSION
    )
    engine = create_engine(db_url)
    df = pd.read_csv(config.article_csv)
    with Session(engine) as sess:
        if config.initialise_embeddings:
            print("[INFO] Documents encoding...")
            df = df.apply(lambda x: x.astype(str).str.lower())
            BATCH = df.shape[0] // BATCH_SIZE + int(
                df.shape[0] % BATCH_SIZE > 0
            )

            metadata.drop_all(bind=engine)
            metadata.create_all(bind=engine)
            sess.query(record).delete()
            sess.commit()

            for i in tqdm(range(BATCH)):
                ids = df["article_id"][
                    i * BATCH_SIZE : (i + 1) * BATCH_SIZE
                ].tolist()
                docs = [
                    pre_encoding_format.format(**tmp)
                    for tmp in df.iloc[
                        i * BATCH_SIZE : (i + 1) * BATCH_SIZE
                    ].to_dict(orient="records")
                ]
                vecs = client.create_embeddings(
                    docs, EMBEDDING_DEPLOYMENT_NAME
                )

                for tmpid, tmpvec in zip(ids, vecs):
                    tmprow = record(id=tmpid, factors=tmpvec)
                    sess.add(tmprow)
                sess.commit()
                time.sleep(5)

        print("[INFO] Chatbot starts...")
        print("[INFO] Type 'mode 1' to start Product Advice Mode.")
        print(
            """[INFO] Type 'mode 2' to start Customer Conversation Mode \
for fashion guidance."""
        )
        print("[INFO] Type 'exit' to terminate the chatbot.")
        print("[INFO] Type 'restart' to clear chat history.")
        print("[INFO] Current Mode: 'mode 2'.")
        conversation_loop(df, client, sess)

    engine.dispose()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)
