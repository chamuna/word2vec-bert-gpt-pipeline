from gensim.models import Word2Vec  # ไลบรารีสำหรับสร้าง Word2Vec embeddings

def generate_word2vec_embeddings(sentences):
    """
    สร้าง Word2Vec embeddings สำหรับคำในประโยค
    :param sentences: รายการของรายการคำ (list of tokenized sentences)
                      ตัวอย่าง: [["hello", "world"], ["word2vec", "is", "useful"]]
    :return: Word2Vec model's KeyedVectors (ชุดเวกเตอร์ของคำที่สร้างขึ้น)
    """
    # สร้างโมเดล Word2Vec โดยใช้พารามิเตอร์เริ่มต้น
    model = Word2Vec(
        sentences,           # ประโยคที่ผ่านการ tokenized แล้ว (list of lists)
        vector_size=100,     # ขนาดของเวกเตอร์สำหรับแต่ละคำ (100 มิติ)
        window=5,            # ขอบเขตของคำที่มองรอบข้าง (5 คำด้านซ้ายและขวา)
        min_count=1,         # จำนวนครั้งขั้นต่ำที่คำต้องปรากฏเพื่อสร้างเวกเตอร์
        workers=4            # จำนวน thread ที่ใช้ในการประมวลผล (4 threads)
    )
    
    # คืนค่าชุดเวกเตอร์ของคำที่อยู่ในโมเดล (KeyedVectors)
    return model.wv
