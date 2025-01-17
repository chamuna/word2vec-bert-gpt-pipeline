from transformers import GPT2LMHeadModel, GPT2Tokenizer  # ไลบรารีสำหรับการใช้งาน GPT-2 ใน Hugging Face

def generate_response(prompt, model_name="gpt2", max_length=500, num_beams=5, temperature=1, top_k=50, top_p=0.95):
    """
    สร้างคำตอบจากข้อความที่กำหนด (Prompt) ด้วย GPT
    :param prompt: ข้อความนำหรือคำถามที่ใช้เพื่อสร้างคำตอบ
    :param model_name: ชื่อโมเดล GPT ที่ต้องการใช้งาน (ค่าเริ่มต้น: "gpt2")
    :param max_length: ความยาวสูงสุดของข้อความที่สร้างขึ้น (ค่าเริ่มต้น: 500 token)
    :param num_beams: จำนวน Beam สำหรับ Beam Search (เพิ่มความหลากหลายในคำตอบ)
    :param temperature: ควบคุมความหลากหลายของการสุ่มคำ (ค่าต่ำ = คำตอบคาดการณ์ได้สูง)
    :param top_k: จำกัดจำนวนคำที่สุ่มพิจารณาในแต่ละตำแหน่ง
    :param top_p: กำหนด cumulative probability (ใช้สำหรับ nucleus sampling)
    :return: คำตอบที่สร้างขึ้นในรูปแบบข้อความ (string)
    """
    # โหลด Tokenizer และ Model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # โหลด Tokenizer ที่เหมาะสมกับโมเดล
    model = GPT2LMHeadModel.from_pretrained(model_name)    # โหลดโมเดล GPT

    # กำหนด pad_token หากยังไม่มีค่า (เช่น ใน GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ใช้ eos_token แทน pad_token

    # แปลงข้อความ (Prompt) เป็น token พร้อม Padding และ Truncation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # ใช้โมเดลสร้างข้อความจาก Token ที่ป้อนเข้าไป
    outputs = model.generate(
        inputs.input_ids,               # Input IDs จากข้อความนำ
        attention_mask=inputs.attention_mask,  # ระบุส่วนที่ต้องประมวลผล
        max_length=max_length,          # กำหนดความยาวสูงสุดของข้อความที่สร้าง
        num_beams=num_beams,            # ใช้ Beam Search เพื่อเพิ่มคุณภาพคำตอบ
        temperature=temperature,        # ควบคุมความหลากหลายของการสุ่มคำ
        top_k=top_k,                    # จำกัดจำนวนคำที่สุ่มพิจารณา
        top_p=top_p,                    # กำหนด cumulative probability (nucleus sampling)
        do_sample=True,                 # เปิดใช้งานการสุ่ม Sampling
        eos_token_id=tokenizer.eos_token_id,  # ใช้ end-of-sequence token สำหรับหยุดข้อความ
        pad_token_id=tokenizer.pad_token_id,  # ใช้ pad token สำหรับข้อความที่สั้นกว่า max_length
        no_repeat_ngram_size=3,         # ป้องกันการซ้ำของ n-gram (3-gram)
    )

    # ถอดรหัสข้อความที่สร้างจาก token เป็น string พร้อมลบ token พิเศษ
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
