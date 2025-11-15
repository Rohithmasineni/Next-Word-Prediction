# Next-Word-Prediction

# Next Word Prediction using LSTM (PyTorch)

This project implements a simple end-to-end **Next Word Prediction Model** using PyTorch.  
It demonstrates the full NLP pipeline — from text preprocessing to sequence modeling using an LSTM.

## Workflow
1. Load and clean text  
2. Build vocabulary + special tokens  
3. Convert text into sequences  
4. Encode words into integer IDs  
5. Pad sequences to equal length  
6. Create dataset & dataloader  
7. Build LSTM model  
8. Train and evaluate  
9. Predict next word for a given prompt

## Model Architecture
- **Embedding Layer** (100-dim)  
- **LSTM** (2 layers, hidden size 256)  
- **Fully-Connected Layer**  
- **Softmax output** over vocabulary

## Features
- Handles unknown words using `<UNK>`  
- Uses `<PAD>` tokens for variable sequence lengths  
- Predicts top-k next-word probabilities  
- Clean, modular code for easy extension

## Tech Stack
- Python  
- PyTorch  
- TorchText / basic preprocessing  
- NumPy

## Result
A functional next-word prediction pipeline demonstrating how sequence modeling works, suitable for learning LSTM-based NLP workflows.

---
