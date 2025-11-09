# Multi-GNN with Graph Transformer

This project was forked from the [Multi-GNN](https://github.com/IBM/Multi-GNN) project, which provides a framework for running multiple Graph Neural Network models for Anti-Money Laundering. This fork extends the original project by adding a Graph Transformer model.

## Graph Transformer Model

The Graph Transformer model is an adaptation of the Transformer architecture for graph-structured data. It uses self-attention mechanisms to weigh the importance of neighboring nodes and edges, allowing it to capture complex relationships in the data.

### Model Parameters

The `GraphTransformer` model has the following parameters, which can be tuned:

*   `--n_hidden`: The number of hidden units in the model.
*   `--n_gnn_layers`: The number of Graph Transformer layers.
*   `--n_heads`: The number of attention heads in the `TransformerConv` layer. Note that `n_hidden` must be divisible by `n_heads`.
*   `--dropout`: The dropout rate for the `TransformerConv` layer.
*   `--final_dropout`: The dropout rate for the final MLP layer.

## Usage

To use the Graph Transformer model, set the `--model` argument to `graph_transformer`. You can also specify the model parameters as arguments.

For example, to run the Graph Transformer model with 2 layers, 128 hidden units, and 4 attention heads, you can use the following command:

```bash
python main.py --data Small_HI --model graph_transformer --n_gnn_layers 2 --n_hidden 128 --n_heads 4
```

For other setup and usage instructions, please refer to the original project's `README.md`.