import os
from embeddings.generate_embeddings import generate_word2vec_embeddings
from encoding.bert_encoder import encode_sentence  # โมดูลสำหรับแปลงประโยคเป็นเวกเตอร์ด้วย BERT
from retrieval.faiss_index import create_index, search_index  # โมดูลสำหรับสร้างและค้นหาใน FAISS Index
from generation.gpt_response import generate_response  # โมดูลสำหรับสร้างคำตอบด้วย GPT
import json
from concurrent.futures import ThreadPoolExecutor
import torch  # สำหรับจัดการกับ GPU
import time

# ปิดคำเตือนเกี่ยวกับ Symlink (สำหรับ Windows) และ OpenMP (การประมวลผลแบบ multi-threading)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ตรวจสอบว่า GPU พร้อมใช้งาน
print("Checking for GPU availability...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# ฟังก์ชันสำหรับประมวลผลคำถามบน GPU
def process_prompt(prompt):
    """
    รับคำถามและส่งคำถามไปยัง GPT เพื่อสร้างคำตอบ โดยประมวลผลบน GPU
    """
    response = generate_response(prompt)
    return {"prompt": prompt, "response": response}


if __name__ == "__main__":
    # 1. สร้าง Word2Vec Embedding
    sentences = [["this", "is", "a", "test"], ["how", "to", "create", "embeddings"]]
    embeddings = generate_word2vec_embeddings(sentences)
    vectors = embeddings.vectors
    print("Embeddings generated.")

    # 2. ใช้ BERT แปลงประโยคให้เป็นเวกเตอร์
    encoded_sentence = encode_sentence("This is a test sentence.")
    encoded_sentence = encoded_sentence.to(device)  # ย้ายเวกเตอร์ไปยัง GPU
    print(f"Encoded sentence tensor is on device: {encoded_sentence.device}")
    print("Encoded Sentence Shape:", encoded_sentence.shape)

    # 3. สร้าง FAISS Index สำหรับค้นหา
    index = create_index(vectors)
    query_vector = embeddings["test"].reshape(1, -1).astype("float32")
    results = search_index(index, query_vector, k=3)
    print("Search Results:", results)

    # 4. สร้างคำถามจำนวนมากเพื่อทดสอบประสิทธิภาพ
    prompts = [
    """Q: What is artificial intelligence (AI)?
A: Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It includes techniques such as machine learning, natural language processing, and computer vision.
---
Q: What are the main types of AI?
A: The main types of AI are:
   1. Narrow AI
   2. General AI
   3. Super AI
---
Q: Please provide a comprehensive explanation of artificial intelligence (AI). Include the following points: ...
"""
    ]  # สร้างคำถาม 1 ข้อ

    # 5. ประมวลผลคำถามแบบขนาน
    responses = []
    max_threads = 16  # จำนวน threads สำหรับการประมวลผลแบบขนาน
    print(f"Processing {len(prompts)} prompts in parallel with {max_threads} threads...")

    start_time = time.time()  # จับเวลาเริ่มต้น
    with ThreadPoolExecutor(max_threads) as executor:
        results = list(executor.map(process_prompt, prompts))
        responses.extend(results)
    end_time = time.time()  # จับเวลาสิ้นสุด

    # 6. บันทึกผลลัพธ์ลงไฟล์ JSON
    with open("responses.json", "w") as f:
        json.dump(responses, f, indent=4)

    print("Responses saved to responses.json")
    print(f"Time taken: {end_time - start_time:.2f} seconds")