import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNetHead(nn.Module):
    """Prototypical Network head for few-shot classification."""
    def __init__(self):
        super().__init__()

    def forward(self, support, support_labels, query):
        """
        support: (n_support, feat_dim)
        support_labels: (n_support,)
        query: (n_query, feat_dim)
        """
        # Compute prototypes for each class
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            # Get all support examples for this class
            class_support = support[support_labels == label]
            # Compute prototype as mean of support examples
            prototype = class_support.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, feat_dim)
        
        # Compute squared Euclidean distances between query examples and prototypes
        # Using the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        n_query = query.size(0)
        n_proto = prototypes.size(0)
        
        # Expand dimensions for broadcasting
        query_expanded = query.unsqueeze(1).expand(-1, n_proto, -1)  # (n_query, n_proto, feat_dim)
        proto_expanded = prototypes.unsqueeze(0).expand(n_query, -1, -1)  # (n_query, n_proto, feat_dim)
        
        # Compute distances
        distances = torch.pow(query_expanded - proto_expanded, 2).sum(dim=2)  # (n_query, n_proto)
        
        # Return negative distances as logits (higher is better)
        return -distances
