import os
import json
import requests
from src.hipporag import HippoRAG


def call_vllm(prompt: str, base_url: str, model: str) -> str:
    """调用本地 vLLM 接口"""
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        print("🔴 LLM response:", response.status_code, response.text)
        return ""


def plan_subqueries_with_llm(query: str, base_url: str, model: str) -> list[str]:
    """调用 LLM 自动规划子问题，返回子问题列表"""
    prompt = f"""You are a reasoning planner. Your task is to decompose a multi-hop question into a list of simpler sub-questions.

Question: {query}

Please output the list in JSON format as follows:
[
  "First sub-question?",
  "Second sub-question?",
  ...
]
Only output valid JSON. Do not add any explanation."""
    response = call_vllm(prompt, base_url, model)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON from LLM response:")
        print(response)
        return []


def route_query(query: str) -> str:
    """简单规则路由"""
    if "Cinderella" in query or "prince" in query:
        return "local"
    else:
        return "global"


def main():
    save_dir = "outputs/routing_multihop_demo"
    llm_model_id = "/common/users/mg1998/models/Meta-Llama-3-8B-Instruct"
    embedding_model_name = "nvidia/NV-Embed-v2"
    base_url = "http://localhost:8001/v1"

    # 数据库内容
    local_docs = [
        "George Rankin is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom."
    ]
    global_docs = [
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.",
        "Rockland County is located in New York.",
        "New York is a state in the United States."
    ]

    # 初始化数据库
    local_rag = HippoRAG(
        save_dir=os.path.join(save_dir, "local"),
        llm_model_name=llm_model_id,
        embedding_model_name=embedding_model_name,
        llm_base_url=base_url,
    )
    global_rag = HippoRAG(
        save_dir=os.path.join(save_dir, "global"),
        llm_model_name=llm_model_id,
        embedding_model_name=embedding_model_name,
        llm_base_url=base_url,
    )

    local_rag.index(local_docs)
    global_rag.index(global_docs)

    # 多跳问题
    multi_hop_query = "What country is the birthplace of Erik Hort a part of?"

    subqueries = plan_subqueries_with_llm(multi_hop_query, base_url, llm_model_id)
    if not subqueries:
        print("❌ 子问题规划失败，退出。")
        return

    results = []
    fused_answer_texts = []

    for i, subquery in enumerate(subqueries):
        route = route_query(subquery)
        rag = local_rag if route == "local" else global_rag
        print(f"\n🔍 Routing subquery '{subquery}' to {route.upper()} DB")

        try:
            answer_obj = rag.rag_qa([subquery])[0][0]
            answer_text = getattr(answer_obj, "answer", answer_obj)
            docs = getattr(answer_obj, "docs", [])
            doc_scores = getattr(answer_obj, "doc_scores", [])
        except Exception as e:
            answer_text = f"Error: {str(e)}"
            docs, doc_scores = [], []

        results.append({
            "subquery": subquery,
            "routing": route,
            "answer": answer_text,
            "docs": docs,
            "doc_scores": doc_scores,
        })
        fused_answer_texts.append(f"{subquery} → {answer_text}")

    # 保存结果 txt（可读格式）
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, "routing_multihop_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Multi-hop query: {multi_hop_query}\n\n")
        for r in results:
            f.write(f"Subquery: {r['subquery']}\n")
            f.write(f"Routing: {r['routing']}\n")
            f.write(f"Answer: {r['answer']}\n")
            for i, doc in enumerate(r["docs"]):
                f.write(f"Doc[{i}]: {doc} (score: {r['doc_scores'][i]})\n")
            f.write("\n")
        f.write("Fused answer chain:\n")
        for step in fused_answer_texts:
            f.write(step + "\n")

    # 保存结构化 JSONL
    jsonl_path = os.path.join(save_dir, "routing_multihop_results.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            r["doc_scores"] = r["doc_scores"].tolist() if hasattr(r["doc_scores"], "tolist") else r["doc_scores"]
            json.dump(r, f)
            f.write("\n")

    print("\n✅ Results saved to:")
    print(f"   - {txt_path}")
    print(f"   - {jsonl_path}")


if __name__ == "__main__":
    main()
