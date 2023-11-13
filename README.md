# qa-chatgpt-pgvector

![sample](resources/sample.png)

## Schedule (8-13 Nov)
1. docker-compose.yml + Dockerfile.pgvector -> 8 Nov
    - initialise a database with vector support.
1. test-openai.ipynb
    - test openai==1.1.1 api, including embeddings and chat completions.
2. create-embeddings.ipynb
    - create embeddings for all articles.
    - discover 3 nan details in record.
    - store embeddings in postgres using pgvector.
3. doc-search.ipynb
    - prove that pgvector can help cosine similiarity calculation.
    - only embedding details from dataframe are not enough as answers can be inaccurate.
    - try question-answering with given information.
4. chatbot.ipynb
    - implement chat completion loop.
5. guardrail_openai-0-28-1.ipynb
    - try guardrails-ai.
    - switch openai version to 0.28.x as 1.1.x is not compatible with guardrails-ai.
    - try all embeddings and chat completions in 0.28.x openai api.
6. qa_package.main (Mode 1 and 2) -> 9 Nov
    - create chatbot.
7. hf-captions -> 12 Nov
    - test gpt4 vision, it requires payment to use. (rejected)
    - test hugging face Salesforce/blip-image-captioning-base model for captioning.
    - analyse if clusterings can help suggestion in same or different categories.
8. qa_package.main (Mode 3) -> 13 Nov
    - find garments based on text + image.

## How to run?
1. Edit environment variables.
    ```
    cp .env.example .env
    cp .env jpnotes/
    # Then fill in .env variables
    ```
1. Install required packages. `pip install -e .`
2. Initialise postgres in Docker. `docker compose up --build`
3. Run chatbot. (Only run with `--initialise-embeddings` for the first time)
    ```python
    python -m qa_package.main \
    --batch-size int \
    --root-image-dir str \
    --article-csv str \
    --initialise-embeddings \
    --visualise
    ```
    for example,
    ```
    python -m qa_package.main \
    --batch-size 16 \
    --initialise-embeddings \
    --article-csv /Users/spare/Documents/data/articles.csv \
    --root-image-dir /Users/spare/Documents/data/images/ \
    --visualise
    ```

## References
1. https://pypi.org/project/openai/0.28.1/
    - Old Documentation of openai 0.28.1
2. https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints
    - openai 0.28.1 vs openai 1.1.1
3. https://www.kommunicate.io/blog/create-a-customer-service-chatbot-using-chatgpt/
    - Conversation loop for chatbot.
4. https://www.mlq.ai/fine-tuning-gpt-3-question-answer-bot/
    - Question Answering with given information.
5. https://docs.guardrailsai.com/defining_guards/pydantic/
    - Define Guardrails with Pydantic.
6. https://docs.guardrailsai.com/guardrails_ai/getting_started/#creating-a-rail-spec
    - Guardrails example.
7. https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
    - openai 0.28.1 can support gpt4 vision preview.
8. https://huggingface.co/tasks/image-to-text
    - hugging face image caption model
