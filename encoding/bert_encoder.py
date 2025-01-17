from transformers import BertModel, BertTokenizer  # ไลบรารีสำหรับโหลดโมเดลและ tokenizer ของ BERT
import torch  # ไลบรารีสำหรับการประมวลผล tensor และ deep learning

def encode_sentence(sentence, model_name="bert-base-uncased"):
    """
    ใช้ BERT เพื่อแปลง (encode) ประโยคเป็นเวกเตอร์เชิงความหมาย
    :param sentence: ข้อความหรือประโยคที่ต้องการแปลงเป็นเวกเตอร์
    :param model_name: ชื่อของโมเดล BERT ที่ต้องการใช้งาน (ค่าเริ่มต้น: "bert-base-uncased")
    :return: เวกเตอร์ที่ได้จากการ encode (tensor: last_hidden_state)
    """
    # โหลด Tokenizer สำหรับโมเดลที่กำหนด
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # โหลดโมเดล BERT ที่กำหนด
    model = BertModel.from_pretrained(model_name)

    # แปลงข้อความเป็น token พร้อมกับสร้าง input ที่รองรับ PyTorch
    # return_tensors="pt" กำหนดให้คืนค่าข้อมูลในรูปแบบ tensor ของ PyTorch
    inputs = tokenizer(sentence, return_tensors="pt")

    # ส่งข้อมูลที่แปลงเป็น token เข้าโมเดล BERT
    # outputs ประกอบด้วย:
    # - last_hidden_state: เวกเตอร์เชิงบริบทของทุก token
    # - pooler_output: เวกเตอร์ที่ได้จาก CLS token (สำหรับการสรุปประโยค)
    outputs = model(**inputs)

    # คืนค่า last_hidden_state ซึ่งเป็นเวกเตอร์ที่ encode ได้จากประโยค
    return outputs.last_hidden_state
