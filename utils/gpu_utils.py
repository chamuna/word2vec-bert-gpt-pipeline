import torch  # ไลบรารีสำหรับการประมวลผลแบบ tensor และ deep learning

def get_device():
    """
    ตรวจสอบว่า GPU ใช้งานได้หรือไม่ และคืนค่าชื่อของอุปกรณ์ที่ใช้งาน (GPU หรือ CPU)
    :return: ชื่อของอุปกรณ์ที่ใช้งาน ('cuda' หาก GPU ใช้งานได้ หรือ 'cpu' หากไม่มี GPU)
    """
    # ตรวจสอบว่า GPU (ผ่าน CUDA) ใช้งานได้หรือไม่
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # แสดงข้อความอุปกรณ์ที่กำลังใช้งาน
    print(f"Using device: {device}")
    
    # คืนค่าชื่อของอุปกรณ์ที่ใช้งาน
    return device
