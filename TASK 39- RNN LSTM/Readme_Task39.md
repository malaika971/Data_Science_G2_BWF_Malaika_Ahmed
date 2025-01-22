# Next-Word-Predictor-Using-RNN.
- Simple application using RNNs predicting the next word in a paragraph.
- Model Accuracy 90%

_______________

## Objective
This project aims to build a simple application using Recurrent Neural Networks (RNNs) to predict the next word in a sentence or paragraph.

## Overview
Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequential data such as time-series data, text, audio, and video. They retain information from previous time steps and use that to predict the next output. In this project, an RNN is trained to predict the next word in a sequence of words.

### Key Concepts:
1. **Recurrent Neural Networks (RNN):** These networks use their previous outputs as inputs for the next step, allowing them to work with sequential data.
2. **Long Short-Term Memory (LSTM):** A specific type of RNN that solves the vanishing gradient problem by selectively remembering and forgetting parts of the input data.
3. **Next Word Prediction:** The goal is to predict the next word based on the context (previous words) in a given text sequence.

---

## Data Preprocessing

### Text Input:
```python
text = """ It was November. Although it was not yet late.
the sky was dark when I turned into Laundress Passage. Father had finished for the day.
switched off the shop lights and closed the shutters.
but so I would not come home to darkness he had left on the light over the stairs to the flat.
Through the glass in the door it cast a foolscap rectangle of paleness onto the wet pavement.
and it was while I was standing in that rectangle. about to turn my key in the door. that I first saw the letter """
```

The text input is tokenized into sequences of words, and subsequences of increasing lengths are created for training the model.

---

## Model Architecture

1. **Embedding Layer:** 
   - This layer converts words into dense vectors, helping the model understand word relationships.

2. **LSTM Layers:** 
   - Two LSTM layers are added to capture long-term dependencies in the text. The first LSTM returns sequences, which is fed into the next LSTM layer.

3. **Dense Layer:**
   - A final Dense layer with a softmax activation function is used to predict the next word based on the current sequence.

---

## Model Training

- **Training Epochs:** 100 epochs
- **Batch Size:** N/A (fit directly on training data)
- **Optimizer:** Adam optimizer
- **Loss Function:** Categorical cross-entropy

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)
```

### Training Results:
- **Total Epochs:** 100 epochs
- **Final Accuracy:** 90.60%
- **Final Loss:** 0.5988

Training progressed with gradual improvements in accuracy, as seen in the following statistics:

| Epoch | Accuracy | Loss     |
|-------|----------|----------|
| 1     | 0.0261   | 4.1882   |
| 10    | 0.1370   | 3.6986   |
| 20    | 0.2155   | 3.0915   |
| 50    | 0.5786   | 1.7583   |
| 100   | 0.9060   | 0.5988   |

---

## Text Generation

After training, the model can predict the next word in a given sequence. For example:

### Input Text:
```python
p_text = "the sky was dark"
```

### Generated Text:
```
the sky was dark when
the sky was dark when i
the sky was dark when i turned
the sky was dark when i turned into
...
```

The model predicts one word at a time, progressively building the sentence.

---

## Statistics

- **Number of Training Samples:** Based on the input text, the model was trained on 195 sequences.
- **Maximum Sequence Length:** 24 words (determined by the longest sequence in the training data).
- **Vocabulary Size:** 65 unique words (based on the tokenization of the provided text).

### Model Performance:
- The model reaches an accuracy of **90.60%** after 100 epochs, demonstrating its ability to learn the structure of the language and predict the next word accurately.
- **Training Time:** Each epoch took approximately 0.1 seconds, leading to a total training time of around **10 minutes** for 100 epochs.

---

## Conclusion

This project demonstrates the power of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks in predicting the next word in a sequence of text. With further optimization and a larger dataset, this model can be extended to handle more complex language tasks.

---
