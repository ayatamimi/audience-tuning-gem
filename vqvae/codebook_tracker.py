import torch

class CodebookTracker:
    def __init__(self, levels: int):
        """
        Initializes a tracker for multiple hierarchical levels.

        Args:
            levels (int): Number of hierarchical levels in the model.
        """
        self.levels = levels
        self.rolling_10_batches = {level: [] for level in range(levels)}
        self.used_codebooks_over_10batch = {level: [] for level in range(levels)}

    def update_codebook_usage(self, indices: torch.Tensor, level: int):
        """
        Updates the codebook usage tracking at over batch and 10-batch for a given hierarchical level.

        Args:
            indices (torch.Tensor): Indices of used codebooks for the batch.
            level (int): The hierarchical level (e.g., 0 for bottom, 1 for top, etc.).

        Returns:
            Tuple[int, int]: (num_used_codebooks_over_batch, num_unique_indices_10batch)
        """
        if level not in self.rolling_10_batches:
            raise ValueError(f"Invalid level: {level}. Expected levels between 0 and {self.levels - 1}.")

        unique_indices_batch = torch.unique(torch.flatten(indices, start_dim=1), return_counts=False)
        num_used_codebooks_over_batch = unique_indices_batch.numel()
        self.rolling_10_batches[level].append(unique_indices_batch)
        if len(self.rolling_10_batches[level]) < 10:
            num_unique_indices_10batch = num_used_codebooks_over_batch
        else:
            unique_indices_10batch = torch.unique(torch.cat(self.rolling_10_batches[level], dim=0))
            num_unique_indices_10batch = unique_indices_10batch.numel()
            self.used_codebooks_over_10batch[level].append(num_unique_indices_10batch)
            self.rolling_10_batches[level].pop(0)  #remove oldest batch in the window

        return num_used_codebooks_over_batch, num_unique_indices_10batch
