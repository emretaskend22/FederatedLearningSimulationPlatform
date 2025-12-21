import numpy as np

class DataPartitioner:
    def __init__(self, dataset, num_clients, partition='iid', beta=0.5):
        """
        Args:
            dataset: PyTorch Dataset (or any object with .y labels as tensor/array)
            num_clients: Number of clients
            partition: 'iid' or 'non-iid'
            beta: Dirichlet concentration parameter for non-iid skew (lower = more skewed)
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition = partition
        self.beta = beta
        
        # Extract labels
        # Assuming dataset.y is [N, 1] tensor or [N] tensor
        if hasattr(dataset, 'y'):
            if hasattr(dataset.y, 'numpy'):
                 self.labels = dataset.y.numpy().flatten()
            else:
                 self.labels = np.array(dataset.y).flatten()
                 
        # If it's a TensorDataset, getting labels might be different depending on construction
        # But our AdultDataset provides .y
        self.client_dict = self._partition_data()

    def _partition_data(self):
        N = len(self.dataset)
        idxs = np.random.permutation(N)
        
        if self.partition == 'iid':
             # Uniform split
             batch_idxs = np.array_split(idxs, self.num_clients)
             return {i: batch_idxs[i] for i in range(self.num_clients)}
             
        elif self.partition == 'non-iid':
            # Dirichlet distribution
            min_size = 0
            # Label classes
            K = len(np.unique(self.labels))
            N = self.labels.shape[0]
            
            while min_size < 10: # Ensure each client gets at least 10 samples
                idx_batch = [[] for _ in range(self.num_clients)]
                for k in range(K):
                    idx_k = np.where(self.labels == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.beta, self.num_clients))
                    
                    # Balance proportions to ensure no empty
                    proportions = np.array([p * (len(idx_j) < N / self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            return {i: np.array(idx_batch[i]) for i in range(self.num_clients)}
        
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition}")

    def use(self, client_id):
        """Returns indices for client_id"""
        return self.client_dict[client_id]
