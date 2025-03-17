## INTRO

1. Explain Self-attention mechanism in Transformers. What problem does it solve compared to traditional sequence models like RNNs and LSTMs?
2. Why do Transformers use positional encoding? How is it implemented mathematically?
3. What is multi-head attention, and why is it important in Transformers? How is it different from single-head attention?
4. How does the Transformer model handle long-range dependencies better than RNNs?

## NORMALIZATION

4. Why do Transformers use Layer Normalization instead of Batch Normalization?
5. Why do Transformers use Residual Connections and Layer Normalization in each block?

## FFN
6. What is the role of the feed-forward network (FFN) in a Transformer block? Why do we need it after the self-attention layer?
7. What happens if we remove the feed-forward network (FFN) from a Transformer?

## ENCODER DECODER
7. What are the key differences between the Encoder and Decoder in the Transformer model?
8. Why does the Transformer model use dropout, and where is it applied?

## ATTENTION
8. What is the difference between self-attention and cross-attention in a Transformer model?
6. Why do Transformers scale the dot-product attention by \frac{1}{\sqrt{d_k}}?
7. What is the computational complexity of self-attention in Transformers, and why is it a limitation?
8. What is the role of the softmax function in self-attention? Why do we need it?
9. Why do Transformers use multiple layers of self-attention instead of just one?

## BEYOND TRANSFORMERS
20. Transformers use weight tying in language models. What does that mean, and why is it useful? 
21. What are the main differences between BERT and GPT?
