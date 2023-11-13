from transformers import pipeline


class HuggingFace:
    def __init__(self):
        self.captioner = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base"
        )

    def caption_url_image(self, url: str) -> list[dict[str, str]]:
        return self.captioner(url)

    def caption_local_image(self, path: str) -> list[dict[str, str]]:
        return self.captioner(path)
