import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def load_triples_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

os.environ['OPENAI_API_KEY'] = 'siq3nBr8C75Pv89E0CQaKq4c3KTCpOREj8Umj8OMCM5ByKkBrHxm-IOPiLuFlEOjnU3HFE5Hv-sfLzShM8CCoA'

def main():

    # Prepare datasets and evaluation
    # docs = [
    #     "Oliver Badman is a politician.",
    #     "George Rankin is a politician.",
    #     "Thomas Marwick is a politician.",
    #     "Cinderella attended the royal ball.",
    #     "The prince used the lost glass slipper to search the kingdom.",
    #     "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    #     "Erik Hort's birthplace is Montebello.",
    #     "Marina is bom in Minsk.",
    #     "Montebello is a part of Rockland County."
    # ]
    my_json_path = "outputs/Harmony_docs_zh_cn/markdown_parse/triples.json"
    my_json_data = load_triples_json(my_json_path)

    # save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    save_dir = 'outputs/Harmony_docs_zh_cn'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    llm_model_name = 'DeepSeekV3'
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = 'Qwen3-Embedding-4B'

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.vveai.com/v1',
                        # embedding_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name
                        # embedding_base_url='127.0.0.1:11434',
                        # embedding_model_name='dengcao/Qwen3-Embedding-8B:Q5_K_M'
                        )

    # Run indexing
    # hipporag.index_from_json(my_json_data)

    # return
    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    # answers = [
    #     ["Politician"],
    #     ["By going to the ball."],
    #     ["Rockland County"]
    # ]

    # gold_docs = [
    #     ["George Rankin is a politician."],
    #     ["Cinderella attended the royal ball.",
    #      "The prince used the lost glass slipper to search the kingdom.",
    #      "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    #     ["Erik Hort's birthplace is Montebello.",
    #      "Montebello is a part of Rockland County."]
    # ]

    print(hipporag.rag_qa_hierarchical(queries=queries))
    # ,
    #                               gold_docs=gold_docs,
    #                               gold_answers=answers))

if __name__ == "__main__":
    main()
