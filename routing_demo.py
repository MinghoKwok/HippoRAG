import os
import json
from src.hipporag import HippoRAG

def route_query(query: str) -> str:
    # 简单规则路由：含 Cinderella 或 prince 的走 local，其他走 global
    if "Cinderella" in query or "prince" in query:
        return "local"
    else:
        return "global"

def main():
    save_dir = "outputs/routing_demo"
    llm_model_name = "/common/users/mg1998/models/Meta-Llama-3-8B-Instruct"
    embedding_model_name = "nvidia/NV-Embed-v2"
    base_url = "http://localhost:8001/v1"

    # 数据划分
    local_docs = [
        "George Rankin is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom."
    ]

    global_docs = [
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County."
    ]

    # 初始化两个数据库实例
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

    # 分别构建索引
    local_rag.index(docs=local_docs)
    global_rag.index(docs=global_docs)

    # 查询与答案
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    results = []
    for i, query in enumerate(queries):
        route = route_query(query)
        print(f"Routing '{query}' to {route.upper()} DB")

        rag = local_rag if route == "local" else global_rag
        qa_result = rag.rag_qa(
            queries=[query],
            gold_docs=[gold_docs[i]],
            gold_answers=[answers[i]]
        )
        results.append((query, qa_result))

    # 写入文件
    print(results)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "routing_rag_results.txt"), "w") as f:
        for query, res in results:
            f.write(f"Query: {query}\n")
            f.write(json.dumps(res[-2], indent=2) + "\n")  # recall@k
            f.write(json.dumps(res[-1], indent=2) + "\n\n")  # EM/F1

    print(f"Results saved to {os.path.join(save_dir, 'routing_rag_results.txt')}")

if __name__ == "__main__":
    main()
