# qa-chatgpt-pgvector

## How to run?
1. Install required packages. `pip install -e .`
2. Initialise postgres in Docker. `docker compose up --build`
3. Run chatbot. (Only run with `--initialise-embeddings` for the first time)
    ```python
    python -m qa_package.main \
        --article-csv str \
        --initialise-embeddings
    ```

## Backlogs
#### Experiments
1. test-openai.ipynb
2. create-embeddings.ipynb
3. doc-search.ipynb
4. chatbot.ipynb
5. guardrail_openai-0-28-1.ipynb
