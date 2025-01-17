import os
from embeddings.generate_embeddings import generate_word2vec_embeddings  # โมดูลสำหรับสร้าง Word2Vec Embeddings
from encoding.bert_encoder import encode_sentence  # โมดูลสำหรับแปลงประโยคเป็นเวกเตอร์ด้วย BERT
from retrieval.faiss_index import create_index, search_index  # โมดูลสำหรับสร้างและค้นหาใน FAISS Index
from generation.gpt_response import generate_response  # โมดูลสำหรับสร้างคำตอบด้วย GPT
import json  # ใช้สำหรับจัดการข้อมูลในรูปแบบ JSON

# ปิดคำเตือนเกี่ยวกับ Symlink (สำหรับ Windows) และ OpenMP (การประมวลผลแบบ multi-threading)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # 1. สร้าง Word2Vec Embedding จากชุดคำที่กำหนด
    sentences = [["this", "is", "a", "test"], ["how", "to", "create", "embeddings"]]
    embeddings = generate_word2vec_embeddings(sentences)  # สร้าง Embedding
    vectors = embeddings.vectors  # ดึงค่าเวกเตอร์จาก Embedding ที่สร้าง
    print("Embeddings generated.")  # แสดงข้อความยืนยันการสร้าง Embedding

    # 2. ใช้ BERT แปลงประโยคให้เป็นเวกเตอร์
    encoded_sentence = encode_sentence("This is a test sentence.")  # แปลงประโยคตัวอย่าง
    print("Encoded Sentence Shape:", encoded_sentence.shape)  # แสดงขนาดของเวกเตอร์ที่ได้จากการแปลง

    # 3. สร้าง FAISS Index สำหรับค้นหา และทดลองค้นหาเวกเตอร์
    index = create_index(vectors)  # สร้าง FAISS Index จากเวกเตอร์
    query_vector = embeddings["test"].reshape(1, -1).astype("float32")  # สร้างเวกเตอร์สำหรับค้นหา
    results = search_index(index, query_vector, k=3)  # ค้นหาเวกเตอร์ใกล้เคียงใน Index
    print("Search Results:", results)  # แสดงผลลัพธ์การค้นหา

    # 4. ใช้ GPT สร้างคำตอบจากคำถาม
    prompt = "What is artificial intelligence?"  # คำถามตัวอย่าง
    response = generate_response(prompt)  # สร้างคำตอบด้วย GPT
    print("Generated Response:", response)  # แสดงคำตอบที่ได้จาก GPT

    # 5. ทดสอบคำถามหลายรายการและบันทึกผลลัพธ์ลงในไฟล์ JSON
    responses = []  # สร้างรายการสำหรับเก็บคำถามและคำตอบ
    prompts = [
        "What is artificial intelligence?",
        "How does AI impact society?",
        "What are the challenges of artificial intelligence?",
    ]  # คำถามตัวอย่าง

    # วนลูปเพื่อสร้างคำตอบสำหรับแต่ละคำถาม
    for question in prompts:
        answer = generate_response(question)  # สร้างคำตอบ
        responses.append({"prompt": question, "response": answer})  # เพิ่มคำถามและคำตอบลงในรายการ

    # บันทึกคำถามและคำตอบทั้งหมดลงในไฟล์ JSON
    with open("responses.json", "w") as f:
        json.dump(responses, f, indent=4)  # บันทึกข้อมูลในรูปแบบที่อ่านง่าย (indent=4)

    print("Responses saved to responses.json")  # แสดงข้อความยืนยันการบันทึกไฟล์
