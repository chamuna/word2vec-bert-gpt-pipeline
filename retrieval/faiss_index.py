import faiss  # ไลบรารีสำหรับการจัดการและค้นหาเวกเตอร์อย่างมีประสิทธิภาพ
import numpy as np  # ไลบรารีสำหรับการจัดการข้อมูลในรูปแบบอาร์เรย์

def create_index(vectors):
    """
    สร้าง FAISS Index สำหรับการจัดเก็บและค้นหาเวกเตอร์
    :param vectors: numpy array ของเวกเตอร์ที่ต้องการจัดเก็บใน Index
                    ขนาด (จำนวนเวกเตอร์, มิติของเวกเตอร์)
    :return: FAISS Index ที่สร้างขึ้น (สามารถใช้ค้นหาเวกเตอร์ที่คล้ายคลึงกันได้)
    """
    # สร้าง FAISS Index โดยใช้ระยะทางแบบ L2 (Euclidean Distance)
    index = faiss.IndexFlatL2(vectors.shape[1])  # กำหนดมิติของเวกเตอร์จาก vectors.shape[1]
    
    # เพิ่มเวกเตอร์ที่ต้องการจัดเก็บลงใน Index
    index.add(vectors)
    
    # คืนค่า FAISS Index ที่สร้างเสร็จแล้ว
    return index

def search_index(index, query_vector, k=5):
    """
    ค้นหาเวกเตอร์ใน FAISS Index
    :param index: FAISS Index ที่สร้างขึ้น
    :param query_vector: numpy array เวกเตอร์ที่ต้องการค้นหา (ขนาด: (1, มิติของเวกเตอร์))
    :param k: จำนวนผลลัพธ์ที่ต้องการ (ค้นหา Top-k เวกเตอร์ที่ใกล้เคียงที่สุด)
    :return: 
        - distances: ระยะทางระหว่าง query_vector กับเวกเตอร์ใน Index (ค่า L2 Distance)
        - indices: ลำดับของเวกเตอร์ใน Index ที่ใกล้เคียงที่สุด
    """
    # ใช้ Index ค้นหาเวกเตอร์ที่ใกล้เคียงที่สุด k ตัว
    distances, indices = index.search(query_vector, k)
    
    # คืนค่า ระยะทาง (distances) และลำดับเวกเตอร์ (indices) ที่พบ
    return distances, indices
