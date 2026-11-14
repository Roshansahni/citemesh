# Transformer Architecture for Language Modelling
=====================================================

## Summary
-----------

The transformer architecture, introduced in 2017 by Vaswani et al. (2017), revolutionized the field of natural language processing (NLP) by replacing recurrent neural networks (RNNs) for language modelling tasks. This report provides an overview of the transformer architecture and its adoption for language modelling.

## Key Components of the Transformer Architecture
----------------------------------------------

The transformer architecture consists of the following key components:

*   **Self-Attention Mechanism**: The transformer uses self-attention to weigh the importance of different input elements relative to each other. This allows the model to attend to all positions in the input sequence simultaneously and weigh their importance.
*   **Multi-Head Attention**: The transformer uses multi-head attention, which allows the model to jointly attend to information from different representation subspaces at different positions.
*   **Positional Encoding**: The transformer uses positional encoding to preserve the order of the input sequence.
*   **Feed Forward Network (FFN)**: The transformer uses an FFN to transform the output of the self-attention mechanism.

## Why the Transformer Replaced RNNs for Language Modelling
--------------------------------------------------------

The transformer architecture replaced RNNs for language modelling due to its ability to:

*   **Parallelize Computation**: The transformer architecture can be parallelized more easily than RNNs, making it more computationally efficient.
*   **Improve Performance**: The transformer architecture outperformed RNNs on a variety of language modelling tasks, including the popular GLUE benchmark.
*   **Reduce Training Time**: The transformer architecture requires less training time than RNNs, making it more suitable for large-scale language modelling tasks.

## Comparison with RNNs
----------------------

The transformer architecture has several advantages over RNNs for language modelling:

*   **Parallelization**: The transformer architecture can be parallelized more easily than RNNs, making it more computationally efficient.
*   **Scalability**: The transformer architecture can handle longer input sequences than RNNs, making it more suitable for tasks that require longer input sequences.
*   **Performance**: The transformer architecture outperforms RNNs on a variety of language modelling tasks.

## Limitations
-------------

While the transformer architecture has several advantages over RNNs for language modelling, it also has some limitations:

*   **Computational Cost**: The transformer architecture requires more computational resources than RNNs, making it less suitable for low-resource environments.
*   **Training Time**: The transformer architecture requires more training time than RNNs, making it less suitable for tasks that require rapid prototyping.

## References
-------------

*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).
*   Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Advances in Neural Information Processing Systems (pp. 4694-4704).