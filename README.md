# Next Word Prediction using LSTM (PyTorch)

This project demonstrates a complete **end-to-end NLP pipeline** for building a **Next Word Prediction model** using PyTorch.  
Although the model did not produce highly accurate predictions (mainly due to the small dataset), the project successfully helped me understand **how text data flows through an actual deep learning pipeline** — from raw sentences to word embeddings to LSTM-based sequence prediction.

The goal of the project was **learning**, not achieving state-of-the-art accuracy — and it served that purpose perfectly.

---

## Project Overview

The idea is simple:

Given a sequence of words → predict the next word.

To build this, I implemented the entire workflow manually:
- preprocessing raw text,
- building a vocabulary,
- converting sentences into training sequences,
- padding,
- embeddings,
- creating an LSTM model,
- training,
- and finally generating predictions.

This helped me deeply understand the internal mechanics behind sequence models and next-word prediction tasks.

---

## 🔄 Workflow (End-to-End)

1. **Load & clean text**  
   Lowercasing, removing unwanted characters, splitting sentences.

2. **Tokenization & Vocabulary Building**  
   Created `word_to_index` and `index_to_word` mappings.  
   Added `<PAD>` and `<UNK>` tokens.

3. **Sequence Creation**  
   Generated sliding-window input sequences and their next-word targets.

4. **Encoding**  
   Converted words → integer token IDs.

5. **Padding**  
   Ensured all sequences have the same length using `pad_sequence`.

6. **Embeddings**  
   Used `nn.Embedding` to map token IDs → 100-dimensional dense vectors.

7. **LSTM Model**  
   A simple architecture:  
   `Embedding → LSTM → Last Hidden State → Linear → Softmax`.

8. **Training**  
   Batched the data using DataLoader and trained using CrossEntropyLoss + Adam.

9. **Prediction Function**  
   Provided top-k predicted next words for any given text prompt.

---

## Model Architecture

- **Embedding Layer** (100 dimensions)  
- **LSTM**  
  - 2 layers  
  - hidden size = 256  
- **Fully Connected Layer** → prediction scores for each word in the vocabulary  
- **Softmax** for probability distribution

---

## Results

Since the dataset/corpus was very small, the predictions were not highly accurate — but the main objective was achieved:

✔ Learned how to build a complete NLP pipeline  
✔ Understood how sequences, padding, and embeddings work  
✔ Implemented an LSTM model and trained it end-to-end  
✔ Created a functioning “next-word prediction” system

This project acts as a **foundation** for more advanced NLP tasks or transformer-based models.

---

## Tech Stack

- Python  
- PyTorch  
- Torch utilities  
- NumPy  
- Basic text processing (regex, tokenization)

---

## Future Improvements

- Train on a larger corpus  
- Use pretrained embeddings (GloVe, FastText)  
- Try GRU, BiLSTM, or Transformer  
- Experiment with different sequence lengths & hyperparameters  

---

Feel free to explore, improve, or extend this project!

