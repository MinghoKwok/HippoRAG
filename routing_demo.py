import os
import json
from src.hipporag import HippoRAG


def route_query(query: str) -> str:
    """简单规则路由"""
    if "Cinderella" in query or "prince" in query:
        return "local"
    else:
        return "global"


def main():
    save_dir = "outputs/routing_multihop_demo"
    llm_model_name = "/common/users/mg1998/models/Meta-Llama-3-8B-Instruct"
    embedding_model_name = "nvidia/NV-Embed-v2"
    base_url = "http://localhost:8001/v1"

    # 构造两个数据库的内容
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

    # 初始化两个数据库
    local_rag = HippoRAG(
        save_dir=os.path.join(save_dir, "local"),
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=base_url
    )

    global_rag = HippoRAG(
        save_dir=os.path.join(save_dir, "global"),
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=base_url
    )

    # 构建索引
    local_rag.index(docs=local_docs)
    global_rag.index(docs=global_docs)

    # 多跳问题定义
    multi_hop_query = "What country is the birthplace of Erik Hort a part of?"

    subqueries = [
        "Where was Erik Hort born?",
        "What county is Montebello a part of?",
        "What state is Rockland County located in?",
        "What country is New York a part of?"
    ]

    # 模拟 gold 信息（仅用于回调计算 recall/F1）
    gold_docs = [
        ["Erik Hort's birthplace is Montebello."],
        ["Montebello is a part of Rockland County."],
        ["Rockland County is located in New York."],
        ["New York is a state in the United States."]
    ]

    gold_answers = [
        ["Montebello"],
        ["Rockland County"],
        ["New York"],
        ["United States"]
    ]

    results = []

    fused_answer_texts = []
    for i, subquery in enumerate(subqueries):
        route = route_query(subquery)
        rag = local_rag if route == "local" else global_rag
        print(f"\nRouting subquery '{subquery}' to {route.upper()} DB")

        qa_result = rag.rag_qa(
            queries=[subquery],
            gold_docs=[gold_docs[i]],
            gold_answers=[gold_answers[i]]
        )
        fused_answer_texts.append(f"{subquery} → {gold_answers[i][0]}")

        results.append({
            "subquery": subquery,
            "routing": route,
            "recall": qa_result[-2],
            "em_f1": qa_result[-1]
        })

    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "routing_multihop_results.txt"), "w") as f:
        f.write(f"Multi-hop query: {multi_hop_query}\n\n")
        for r in results:
            f.write(f"Subquery: {r['subquery']}\n")
            f.write(f"Routing: {r['routing']}\n")
            f.write(json.dumps(r["recall"], indent=2) + "\n")
            f.write(json.dumps(r["em_f1"], indent=2) + "\n\n")

        f.write("Fused answer chain:\n")
        for step in fused_answer_texts:
            f.write(step + "\n")

    print("\n✅ Results saved to:", os.path.join(save_dir, "routing_multihop_results.txt"))


if __name__ == "__main__":
    main()
