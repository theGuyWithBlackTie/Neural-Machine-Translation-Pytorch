Repository contains code for Neural Machine Translation. Dataset is from German language text to English language text.

The model is a seq2seq with self-attention mechanism in decoder. This attention mechanism helps in predicting next word based on encoder's output and previous predicted word.

Decoder also utilizes "teacher-forcing" method while helps decoder in predicting next words.