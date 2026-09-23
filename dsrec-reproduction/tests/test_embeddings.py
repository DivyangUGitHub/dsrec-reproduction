import torch

from src.models.embeddings import ItemEmbedding, TimeEmbedding


def test_item_embedding_shape():
    assert ItemEmbedding(10, 8)(torch.tensor([[1, 2]])).shape == (1, 2, 8)


def test_time_embedding_shape():
    assert TimeEmbedding(10, 8)(torch.tensor([[0, 1]])).shape == (1, 2, 8)
