import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# モデルのロード
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 例として簡単なコードベース
code_snippets = [
    "def quicksort(arr): ...",
    "def mergesort(arr): ...",
    "def bubblesort(arr): ..."
]
file_paths = ["file1.py", "file2.py", "file3.py"]

# コードの埋め込みを取得
embeddings = embedding_model.encode(code_snippets)

# FAISSインデックスの作成
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))

# 検索関数
def search_similar_code(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), top_k)
    results = [(file_paths[i], code_snippets[i]) for i in indices[0]]
    return results

# クエリを実行
query = "def quicksort(arr):"
similar_code = search_similar_code(query)

# 検索結果を表示
for path, code in similar_code:
    print(f"Similar Code in {path}:\n{code}\n")
