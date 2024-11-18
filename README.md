# word2vec-bert-gpt-pipeline
A complete pipeline for processing text using Word2Vec, BERT, and GPT, enabling word embeddings, contextual encoding, vector similarity search, and AI-generated responses. This repository combines state-of-the-art AI models to perform NLP tasks efficiently and effectively.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [How to Use](#how-to-use)
5. [Examples](#examples)
6. [File Structure](#file-structure)
7. [License](#license)

---

## **Overview**
This repository is a practical implementation of an end-to-end Natural Language Processing (NLP) pipeline combining the strengths of Word2Vec, BERT, and GPT models. It supports vector-based semantic search, sentence encoding, and AI-driven text generation.

By leveraging these models, users can:
- Create meaningful embeddings for text.
- Perform contextual sentence analysis.
- Generate coherent AI-based responses.

---

## **Features**
- **Word Embeddings**:
  - Generate word-level embeddings using the Word2Vec model.
- **Sentence Encoding**:
  - Transform sentences into contextual embeddings using pre-trained BERT models.
- **Vector Search**:
  - Perform efficient similarity searches with FAISS.
- **Text Generation**:
  - Generate natural language responses using GPT models.

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/chamuna/word2vec-bert-gpt-pipeline.git
   cd word2vec-bert-gpt-pipeline

   Install the required dependencies:

   bash

   pip install -r requirements.txt
   Run the main script:

   bash

   python main.py
   
## **How to Use**
   Run the pipeline: Execute the script main.py to process text using the pipeline.

   Customize prompts: Edit the prompts in main.py to test specific questions or queries.

   Output location:

   Results are saved in a responses.json file for further review.
## **Examples**
   Input Prompts
   plaintext

   What is artificial intelligence?
   How does AI impact society?
   What are the challenges of artificial intelligence?
   Sample Output (responses.json)
   json

   [
       {
           "prompt": "What is artificial intelligence?",
           "response": "AI is a term used to describe artificial intelligence that is designed to perform complex tasks..."
       },
       {
           "prompt": "How does AI impact society?",
           "response": "AI has impacted society in automation, healthcare, and more..."
       }
   ]
## **File Structure**
   plaintext

   ├── embeddings/
   │   ├── generate_embeddings.py      # Generates Word2Vec embeddings
   ├── encoding/
   │   ├── bert_encoder.py             # Encodes sentences using BERT
   ├── retrieval/
   │   ├── faiss_index.py              # Facilitates similarity search with FAISS
   ├── generation/
   │   ├── gpt_response.py             # Generates responses using GPT
   ├── utils/
   │   ├── config.py                   # Configuration for model and environment
   │   ├── gpu_utils.py                # GPU-related utilities
   ├── main.py                         # Entry point for running the pipeline
   ├── responses.json                  # Stores AI-generated responses
   ├── requirements.txt                # Required Python dependencies

## **License**
   This project is licensed under the MIT License. You are free to use, modify, and distribute it as per the terms of the license.

   Contribution
   Contributions are welcome! If you'd like to improve or extend the pipeline, feel free to fork the repository and submit a pull request.

   Contact
   For questions or feedback, reach out:
   This file includes all requested sections and can be directly saved as **README.md** in your repository. Let me know if you’d like further customizations!






