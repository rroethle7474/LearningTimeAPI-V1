from fastapi import Depends
from db.vector_store import VectorStore
from embeddings.generator import EmbeddingGenerator
from llm.factory import LLMFactory
from generators.tutorial import TutorialGenerator
from search.semantic_search import SemanticSearch

def get_embedding_generator():
    """Dependency to get embedding generator instance"""
    from main import embedding_generator
    return embedding_generator

def get_vector_store(
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency to get vector store instance"""
    return VectorStore(
        embedding_generator=embedding_generator,
        persist_directory="./chromadb"
    )

def get_llm_client():
    """Dependency to get LLM client instance"""
    from main import llm_client
    return llm_client

def get_tutorial_generator(
    llm_client = Depends(get_llm_client),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency injection for TutorialGenerator"""
    return TutorialGenerator(llm_client, vector_store, embedding_generator)

def get_semantic_search(
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Dependency to get semantic search instance"""
    return SemanticSearch(vector_store, embedding_generator) 