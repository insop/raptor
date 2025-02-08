import logging
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class AzureOpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model=None):
        self.api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        self.model = model if model else self.deployment_name
        self.client = AzureOpenAI(
            api_key=self.api_key, 
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            azure_deployment=self.deployment_name
        )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
