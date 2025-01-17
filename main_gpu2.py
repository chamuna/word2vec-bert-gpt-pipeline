import os
import json
import time
import logging
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from embeddings.generate_embeddings import generate_word2vec_embeddings
from encoding.bert_encoder import encode_sentence
from retrieval.faiss_index import create_index, search_index
from generation.gpt_response import generate_response

# Configuration
config = {
    "max_threads": 16,
    "batch_size": 4,
    "output_file": "responses.json",
    "log_file": "pipeline.log",
    "model_settings": {
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=config['log_file']
)

# Environment settings
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

def setup_device():
    """Set up and return the appropriate device (GPU/CPU)"""
    print("Checking for GPU availability...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device

def process_prompt(prompt):
    """Process a single prompt with error handling"""
    try:
        response = generate_response(prompt)
        return {
            "prompt": prompt,
            "response": response,
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logging.error(f"Error processing prompt: {str(e)}")
        return {
            "prompt": prompt,
            "response": None,
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def process_batch(prompts, batch_size=4):
    """Process prompts in batches"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        with ThreadPoolExecutor(config['max_threads']) as executor:
            futures = [executor.submit(process_prompt, prompt) for prompt in batch]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
    return results

def main():
    # Initialize device
    device = setup_device()
    print_gpu_memory()

    try:
        # 1. Generate Word2Vec Embeddings
        logging.info("Generating Word2Vec embeddings...")
        sentences = [["this", "is", "a", "test"], ["how", "to", "create", "embeddings"]]
        embeddings = generate_word2vec_embeddings(sentences)
        vectors = embeddings.vectors
        print("Embeddings generated.")
        print_gpu_memory()

        # 2. BERT Encoding
        logging.info("Encoding sentence with BERT...")
        encoded_sentence = encode_sentence("This is a test sentence.")
        encoded_sentence = encoded_sentence.to(device)
        print(f"Encoded sentence tensor is on device: {encoded_sentence.device}")
        print("Encoded Sentence Shape:", encoded_sentence.shape)
        print_gpu_memory()

        # 3. Create FAISS Index
        logging.info("Creating FAISS index...")
        index = create_index(vectors)
        query_vector = embeddings["test"].reshape(1, -1).astype("float32")
        results = search_index(index, query_vector, k=3)
        print("Search Results:", results)
        print_gpu_memory()

        # 4. Process prompts
        prompts = [
            """Q: What is artificial intelligence (AI)?
            A: Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
            It includes techniques such as machine learning, natural language processing, and computer vision.
            ---
            Q: What are the main types of AI?
            A: The main types of AI are:
               1. Narrow AI
               2. General AI
               3. Super AI
            ---
            Q: Please provide a comprehensive explanation of artificial intelligence (AI).
            Include the following points: ...
            """
        ]

        # 5. Process prompts in batches with progress bar
        logging.info(f"Processing {len(prompts)} prompts...")
        start_time = time.time()
        
        print(f"Processing {len(prompts)} prompts in batches...")
        responses = []
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            batch_results = process_batch(prompts, config['batch_size'])
            responses.extend(batch_results)
            pbar.update(len(batch_results))

        end_time = time.time()
        processing_time = end_time - start_time

        # 6. Save results
        results_data = {
            "metadata": {
                "total_time": processing_time,
                "total_prompts": len(prompts),
                "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config
            },
            "responses": responses
        }

        with open(config['output_file'], "w", encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        print(f"Responses saved to {config['output_file']}")
        print(f"Time taken: {processing_time:.2f} seconds")
        print_gpu_memory()

    except Exception as e:
        logging.error(f"Main process error: {str(e)}")
        raise

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")

if __name__ == "__main__":
    main()