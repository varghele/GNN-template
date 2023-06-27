# GNN-template
Basically just a custom message passing GNN with MLPs as update functions as well as node AND edge aggregation on graph updates. Implemented with the MetaLayer class of Pytorch Geometric (Pytorch2 and PytorchGeometric 2.3).

I uploaded this because I was annoyed that I had to scour through old code every time I wanted to set up a message passing net.
With this baseline code you can implement a message-passing GNN with choosable MLP depth, activation and normalization functions, and number of message-passing steps.
