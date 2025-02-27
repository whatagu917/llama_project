import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# RAG のセットアップ
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# サンプルのコードデータ（実際にはプロジェクトのコードを読み込む）
code_snippets = [
    "def quicksort(arr): return sorted(arr)",
    "def mergesort(arr): return merge(arr)",
    "def bubblesort(arr): return bubble(arr)"
]
file_paths = ["file1.py", "file2.py", "file3.py"]

# コードを埋め込み（ベクトル化）して FAISS に登録
embeddings = embedding_model.encode(code_snippets)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))

# ✅ `BitsAndBytesConfig` を使用して 8bit 量子化を設定
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8bit 量子化
    llm_int8_enable_fp32_cpu_offload=True  # CPU にオフロードしてメモリ節約
)

# CodeLlama-13B を 8bit 量子化でロード
model_name = "codellama/CodeLlama-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # CPU & GPU を自動振り分け
    quantization_config=bnb_config,  # ✅ 8bit 量子化を適用
    torch_dtype=torch.float16
)

# 類似コードを検索する関数
def search_similar_code(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), top_k)
    results = [(file_paths[i], code_snippets[i]) for i in indices[0]]
    return results

# Llama でコード補完を生成
def generate_code_with_context(query):
    similar_code = search_similar_code(query, top_k=2)
    context = "\n".join([f"# 参考コード:\n{code}" for _, code in similar_code])
    prompt = f"{context}\n# 補完したいコード:\n{query}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs, 
        max_length=200,
        pad_token_id=tokenizer.eos_token_id  
    )
    
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# **対話式の実装**
print("\n🔹🔹🔹 対話型コード補完システム 🔹🔹🔹")
print("💡 コードを入力すると、補完結果を表示します。（終了するには `exit` と入力）\n")

while True:
    query = input("📝 コードを入力: ")
    if query.lower() == "exit":
        print("🔚 終了します。")
        break
    
    print("\n🔍 類似コードを検索中...")
    generated_code = generate_code_with_context(query)
    
    print("\n🔹 🔹 🔹 AI が補完したコード 🔹 🔹 🔹\n")
    print(generated_code)
    print("\n──────────────────────────────────\n")
