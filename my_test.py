from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import openai
import os
import json
from time import sleep

# Load BGE reranker
def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

# Reranking function
def rerank_results(model, tokenizer, query, hits, batch_size=8):
    model.eval()
    documents = [hit.raw for hit in hits]
    scores = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        queries = [query] * len(batch_docs)
        
        inputs = tokenizer(
            queries,
            batch_docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1)
            scores.extend(batch_scores.cpu().numpy())
    
    return list(zip(hits, scores))

import openai
import ast
from openai import AsyncOpenAI, OpenAI, APIConnectionError, RateLimitError


LLM_BASE_URL = "http://localhost:8000/v1/"
LLM_API_KEY = "sk-22"
MODEL = "DSF-CUG-LLM"

openai.api_key = " " ## Insert OpenAI's API key
openai_async_client = OpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )

def call_codex_read_api(prompt: str, n =1):
    def parse_api_result(result):
        to_return = []
        for idx, g in enumerate(result['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        res = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return res
    result = []
    for i in range(n):
        generated_output = openai_async_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        result.append(generated_output)
    
    return parse_api_result(result)

# HYDE generation function (similar to the reference code)
def generate_hyde_contexts(query):
    prompt = f"""Please write a passage to answer the question
Question: {query}
Passage:"""
    get_result = False
    while not get_result:
        try:
            contexts = [c.strip() for c in call_codex_read_api(prompt, n=8)] + [query]
            get_result = True
        except:
            sleep(1)
    return contexts

# Insturction for rerank generation function (similar to the reference code)
def generate_query_instruct(query)->str:
    prompt = f"""Please write an instruction to specifies the nature of the task. It should be a short sentence.
task: {query}
Instruction:"""
    get_result = False
    while not get_result:
        try:
            contexts = [c.strip() for c in call_codex_read_api(prompt, n=1)]
            get_result = True
        except:
            sleep(1)
    return contexts[0] + query

def get_averaged_hyde_embedding(query_encoder, contexts):
    """
    Encode all contexts and average their embeddings
    """
    all_emb_c = []
    for c in contexts:
        c_emb = query_encoder.encode(c)
        all_emb_c.append(np.array(c_emb))
    all_emb_c = np.array(all_emb_c)
    avg_emb_c = np.mean(all_emb_c, axis=0)
    return avg_emb_c.reshape((1, len(avg_emb_c)))

topics = get_topics('dl19-passage')
qrels = get_qrels('dl19-passage')

bge_searcher = FaissSearcher.from_prebuilt_index(
    'msmarco-v1-passage.bge-base-en-v1.5',
    'BAAI/bge-base-en-v1.5'
)
reranker_model, reranker_tokenizer = load_reranker()
query_encoder = AutoQueryEncoder('BAAI/bge-base-en-v1.5', device='cuda' if torch.cuda.is_available() else 'cpu')

# Result A: BGE base retrieval
with open('dl19-bge-base-top1000-trec', 'w') as f_A, open('dl19-bge-base-reranked-trec', 'w') as f_B:
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            hits = bge_searcher.search(query, k=1000)
            # Result A: BGE base retrieval
            for rank, hit in enumerate(hits, 1):
                f_A.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
            
            # Result B: BGE base + reranker
            rerank_scores = rerank_results(reranker_model, reranker_tokenizer, query, hits)
            
            # Sort by reranker scores
            ranked_results = sorted(rerank_scores, key=lambda x: x[1], reverse=True)
            for rank, (hit, score) in enumerate(ranked_results, 1):
                f_B.write(f'{qid} Q0 {hit.docid} {rank} {score} rank\n')

# Result X: BGE base + query with instruct retrieval
q_i_generations = {}
with open('dl19-bge-instrct-top1000-trec', 'w') as f_X, open('dl19-bge-instrct-reranked-base-trec', 'w') as f_Y, open('dl19-bge-instrct-reranked-instruct-trec', 'w') as f_Z:
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            print(query)
            query_instruct = generate_query_instruct(query)
            print("query_instruct:\n", query_instruct)

            hits = bge_searcher.search(query_instruct, k=1000)
            # Result X: BGE base + query with instruct retrieval
            for rank, hit in enumerate(hits, 1):
                f_X.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
            
            # Result Y: BGE base + query with instruct retrieval + reranker base
            rerank_scores = rerank_results(reranker_model, reranker_tokenizer, query, hits)
            
            # Sort by reranker scores
            ranked_results = sorted(rerank_scores, key=lambda x: x[1], reverse=True)
            for rank, (hit, score) in enumerate(ranked_results, 1):
                f_Y.write(f'{qid} Q0 {hit.docid} {rank} {score} rank\n')
            

            # Result Z: BGE base + query with instruct retrieval + reranker instruct
            rerank_scores = rerank_results(reranker_model, reranker_tokenizer, query_instruct, hits)
            
            # Sort by reranker scores
            ranked_results = sorted(rerank_scores, key=lambda x: x[1], reverse=True)
            for rank, (hit, score) in enumerate(ranked_results, 1):
                f_Z.write(f'{qid} Q0 {hit.docid} {rank} {score} rank\n')

# Save HYDE generations for analysis
with open('q_i_generations.json', 'w') as f:
    json.dump(q_i_generations, f, indent=2)

# Store HYDE generations for analysis
hyde_generations = {}

# Result D: HYDE + BGE base
with open('dl19-hyde-bge-base-trec', 'w') as f_D, open('dl19-hyde-bge-base-reranked-trec', 'w') as f_E:
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            print(query)
            contexts = generate_hyde_contexts(query)
            
            # Average embeddings of contexts
            contexts = generate_hyde_contexts(query)
            contexts.append(query)  # Add original query as an additional context
            hyde_generations[qid] = contexts  # Save generations for analysis

            # Get averaged embedding for HYDE
            avg_emb = get_averaged_hyde_embedding(query_encoder, contexts)
            
            # Search with averaged embedding
            hyde_hits = bge_searcher.search(avg_emb, k=1000)
            
            for rank, hit in enumerate(hyde_hits, 1):
                f_D.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')

            # Result E: HYDE + BGE base + reranker
            reranked_hyde = rerank_results(reranker_model, reranker_tokenizer, query, hyde_hits)
            
            # Sort by reranker scores
            ranked_hyde_results = sorted(reranked_hyde, key=lambda x: x[1], reverse=True)
            for rank, (hit, score) in enumerate(ranked_hyde_results, 1):
                f_E.write(f'{qid} Q0 {hit.docid} {rank} {score} rank\n')
                
# Save HYDE generations for analysis
with open('hyde_generations.json', 'w') as f:
    json.dump(hyde_generations, f, indent=2)

# Evaluate all results
for result_file in ['dl19-bge-base-top1000-trec', 
                    'dl19-bge-base-reranked-trec',
                    'dl19-hyde-bge-base-trec',
                    'dl19-hyde-bge-base-reranked-trec',
                    'dl19-bge-instrct-top1000-trec',
                    'dl19-bge-instrct-reranked-base-trec',
                    'dl19-bge-instrct-reranked-instruct-trec']:
    print(f"\nEvaluating {result_file}")
    os.system(f"python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage {result_file}")
    os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage {result_file}")
    os.system(f"python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage {result_file}")