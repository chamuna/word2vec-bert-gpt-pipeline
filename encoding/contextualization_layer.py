from transformers import BertModel, BertTokenizer  # ไลบรารีสำหรับโหลดโมเดลและ tokenizer ของ BERT

def contextualize_sentence(sentence, model_name="bert-base-uncased"):
    """
    สร้าง contextualized embedding จากประโยค (เวกเตอร์ที่มีความหมายตามบริบทของประโยค)
    :param sentence: ข้อความหรือประโยคที่ต้องการแปลงเป็น embedding
    :param model_name: ชื่อของโมเดล BERT ที่จะใช้ (ค่าเริ่มต้น: "bert-base-uncased")
    :return: เวกเตอร์ Contextualized ในรูปแบบ tensor (last_hidden_state)
    """
    # โหลด Tokenizer ของโมเดล BERT ที่ระบุ
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # โหลดโมเดล BERT ที่ระบุ
    model = BertModel.from_pretrained(model_name)

    # แปลงข้อความเป็น token และแปลงเป็น input สำหรับโมเดล
    # return_tensors="pt" ระบุให้คืนค่าข้อมูลในรูปแบบ PyTorch tensor
    inputs = tokenizer(sentence, return_tensors="pt")

    # ส่งข้อมูลที่แปลงแล้วเข้าโมเดลเพื่อประมวลผล
    # outputs ประกอบด้วย:
    # - last_hidden_state: เวกเตอร์ที่ได้จากแต่ละ token (มีบริบท)
    # - pooler_output: เวกเตอร์สรุปสำหรับ CLS token
    outputs = model(**inputs)

    # คืนค่า last_hidden_state ซึ่งเป็น contextualized embedding
    return outputs.last_hidden_state
