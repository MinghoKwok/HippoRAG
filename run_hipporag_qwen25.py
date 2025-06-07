from hipporag import HippoRAG

def main():
    # 初始文档
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County."
    ]

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
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince."
        ],
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County."
        ]
    ]

    # 初始化 HippoRAG 实例（使用本地 Qwen2.5）
    hipporag = HippoRAG(
        save_dir='outputs/qwen25_local',
        llm_model_name='Qwen2.5-14B-Instruct',
        llm_base_url='http://localhost:8001/v1',
        embedding_model_name='nvidia/NV-Embed-v2'
    )

    print("\n📥 Indexing...")
    hipporag.index(docs=docs, skip_graph=True)

    print("\n🤖 First QA results:")
    results = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    for r in results:
        print(f"\nQ: {r['question']}\nA: {r['generation']}")

    print("\n➕ Adding new documents...")
    new_docs = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."
    ]
    hipporag.index(docs=new_docs)

    print("\n🤖 QA after new docs added:")
    results = hipporag.rag_qa(queries=queries)
    for r in results:
        print(f"\nQ: {r['question']}\nA: {r['generation']}")

    print("\n❌ Deleting new documents...")
    hipporag.delete(docs=new_docs)

    print("\n🤖 QA after deletion:")
    results = hipporag.rag_qa(queries=queries)
    for r in results:
        print(f"\nQ: {r['question']}\nA: {r['generation']}")

if __name__ == "__main__":
    main()
