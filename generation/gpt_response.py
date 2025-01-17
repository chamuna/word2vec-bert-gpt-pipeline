from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(prompt, model_name="meta-llama/Llama-2-7b-hf", max_length=500):
    # โหลด Tokenizer และ Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="float16",  # ลดการใช้หน่วยความจำ
        low_cpu_mem_usage=True
    )

    # กำหนด pad_token เป็น eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # แปลงข้อความเป็น Tensor พร้อม Padding และ Truncation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")

    # สร้างข้อความตอบกลับ


    outputs = model.generate(
    inputs.input_ids,
    max_length=512,
    min_length=200,
    temperature=0.3,
    top_k=40,
    top_p=0.85,
    no_repeat_ngram_size=3,
    num_beams=4,
    early_stopping=True
)
    # ถอดรหัสข้อความที่ได้
    return tokenizer.decode(outputs[0], skip_special_tokens=True)