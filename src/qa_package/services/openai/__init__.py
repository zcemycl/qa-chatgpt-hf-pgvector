import openai

from ..guardrails import guard_product_advice


class OpenAI:
    def __init__(self, api_key: str, api_base: str, api_version: str):
        openai.api_key = api_key
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_type = "azure"
        self.openai = openai

    def create_embeddings(
        self, docs: list[str], embedding_deployment_name: str
    ) -> list[list[float]]:
        vecs = self.openai.Embedding.create(
            input=docs, engine=embedding_deployment_name
        )
        return [tmp["embedding"] for tmp in vecs["data"]]

    def chat(
        self, messages: list[dict[str, str]], chat_deployment_name: str
    ) -> str:
        response = self.openai.ChatCompletion.create(
            deployment_id=chat_deployment_name, messages=messages
        )
        return response.to_dict()["choices"][0].message["content"]

    def advice_product(
        self, prompt_params: dict[str, str], chat_deployment_name: str
    ) -> dict[str, str]:
        _, validated_output = guard_product_advice(
            self.openai.ChatCompletion.create,
            prompt_params=prompt_params,
            deployment_id=chat_deployment_name,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=1024,
            temperature=0.3,
        )
        return validated_output
