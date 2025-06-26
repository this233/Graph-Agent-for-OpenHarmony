# export OPENAI_API_KEY=sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb


import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

os.environ['OPENAI_API_KEY'] = 'sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb'
def main():

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/openai_test'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    llm_model_name = 'qwen-plus-latest'
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = 'text-embedding-3-large'

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.vveai.com/v1',
                        embedding_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name,
                        embedding_base_url='https://api.vveai.com/v1'
                        )

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name,
                        embedding_base_url='https://api.vveai.com/v1'
                        )

    new_docs = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."]

    # Run indexing
    hipporag.index(docs=new_docs)

    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."
    ]

    hipporag.delete(docs_to_delete)

    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

if __name__ == "__main__":
    main()
