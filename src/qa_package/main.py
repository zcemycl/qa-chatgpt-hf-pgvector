import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from .dataclasses.orm import metadata, record
from .services.openai import OpenAI
from .utils import parse_args

# from sqlalchemy.sql import select


load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
BATCH_SIZE = 5
db_url = "postgresql://postgres:postgres@localhost/postgres"


def conversation_loop(client, session: Session):
    init_messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    mode = "mode 2"
    messages = init_messages.copy()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "mode 1":
            mode = user_input.lower()
        elif user_input.lower() == "mode 2":
            mode = user_input.lower()
            messages = init_messages.copy()

        if mode == "mode 1":
            user_input = input("You: ")
        elif mode == "mode 2":
            if len(messages) == 1:
                user_input = input("You: ")
            messages.append({"role": "user", "content": user_input})
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
    with Session(engine) as sess:
        if config.initialise_embeddings:
            df = pd.read_csv(config.article_csv)
            BATCH = df.shape[0] // BATCH_SIZE + int(
                df.shape[0] % BATCH_SIZE > 0
            )

            metadata.drop_all(bind=engine)
            metadata.create_all(bind=engine)
            sess.query(record).delete()
            sess.commit()

            for i in tqdm(range(BATCH)):
                pass

        conversation_loop(client, sess)

    engine.dispose()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)
