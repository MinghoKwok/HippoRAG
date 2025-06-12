import os
from typing import List
import json

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse
import requests

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


def route_query_with_llm(query: str, model: str, api_key: str, base_url: str) -> str:
    prompt = f"""
You are an intelligent router. Given a user question, classify whether it should be answered using:

- local: personal/internal/private knowledge
    * Like information about a person(not famout)
    * Information inside a company that is not published
- global: general/public/world knowledge
    * General knowledge
    * Like information about a famous person, a historical event, a country, a city, a language, a food, a movie, a book, etc.
    * Like "Which language does the president of the United States speak?"
Only output one word: local or global.

Question: {query}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"].strip().lower()


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset_local', type=str, default='local', help='Dataset name')
    parser.add_argument('--dataset_global', type=str, default='global', help='Dataset name')
    parser.add_argument('--dataset_local_global', type=str, default='local_global', help='QA Set name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs_router', help='Save directory')
    parser.add_argument("--skip_graph", action="store_true", help="Skip OpenIE graph extraction.")
    args = parser.parse_args()

    dataset_local = args.dataset_local
    dataset_global = args.dataset_global
    dataset_local_global = args.dataset_local_global
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    if save_dir == 'outputs_router':
        save_dir_local = save_dir + '/' + dataset_local
        save_dir_global = save_dir + '/' + dataset_global
    else:
        save_dir_local = save_dir + '_' + dataset_local
        save_dir_global = save_dir + '_' + dataset_global

    corpus_path_local = f"reproduce/dataset/{dataset_local}_corpus.json"
    corpus_path_global = f"reproduce/dataset/{dataset_global}_corpus.json"
    with open(corpus_path_local, "r") as f:
        corpus_local = json.load(f)
    with open(corpus_path_global, "r") as f:
        corpus_global = json.load(f)

    docs_local = [f"{doc['title']}\n{doc['text']}" for doc in corpus_local]
    docs_global = [f"{doc['title']}\n{doc['text']}" for doc in corpus_global]
    print("local len:", len(docs_local))
    print("global len:", len(docs_global))

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_local_global}.json", "r"))
    all_queries = [s['question'] for s in samples]

    local_config = BaseConfig(
        save_dir=save_dir_local,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_local,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus_local),
        openie_mode=args.openie_mode
    )
    global_config = BaseConfig(
        save_dir=save_dir_global,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_global,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus_global),
        openie_mode=args.openie_mode
    )

    local_rag = HippoRAG(global_config=local_config)
    global_rag = HippoRAG(global_config=global_config)

    print(f"[Index] skip_graph = {args.skip_graph}")
    local_rag.index(docs_local, skip_graph=args.skip_graph)
    for i, doc in enumerate(docs_local):
        print(f"Doc {i}: {repr(doc)}")
    global_rag.index(docs_global, skip_graph=args.skip_graph)
    for i, doc in enumerate(docs_global):
        print(f"Doc {i}: {repr(doc)}")

    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_local_global) 

    results = []
    for i, query in enumerate(all_queries):
        route = route_query_with_llm(query, args.llm_name, os.getenv("OPENAI_API_KEY"), args.llm_base_url)
        rag = local_rag if route == "local" else global_rag

        result_obj = rag.rag_qa([query], gold_docs=[gold_docs[i]], gold_answers=[gold_answers[i]])[0][0]
        answer_text = getattr(result_obj, "answer", result_obj)

        results.append({
            "query": query,
            "route": route,
            "answer": answer_text,
            "gold_answer": list(gold_answers[i])
        })

    # 保存为 JSONL
    router_output_dir = os.path.join(args.save_dir, "router")
    os.makedirs(router_output_dir, exist_ok=True)
    output_path = os.path.join(router_output_dir, "router_results.jsonl")      
    with open(output_path, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")

    print(f"{i+1}/{len(all_queries)}: {query} → {route}")
    print(f"✅ Results saved to {output_path}")



if __name__ == "__main__":
    main()
