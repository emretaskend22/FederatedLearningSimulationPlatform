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
        
        # Extract labels - support multiple dataset types
        self.labels = self._extract_labels(dataset)
        self.client_dict = self._partition_data()
    
    def _extract_labels(self, dataset):
        """
        Extract labels from various dataset types.
        
        Supports:
        - Custom datasets with .y attribute (AdultDataset, PneumoniaMNISTDataset)
        - TensorDataset (second tensor is labels)
        - Datasets with .labels or .targets attributes
        """
        # Check for .y attribute (AdultDataset, PneumoniaMNISTDataset)
        if hasattr(dataset, 'y'):
            if hasattr(dataset.y, 'numpy'):
                return dataset.y.numpy().flatten()
            else:
                return np.array(dataset.y).flatten()
        
        # Check for .labels attribute (medmnist datasets)
        if hasattr(dataset, 'labels'):
            if hasattr(dataset.labels, 'numpy'):
                return dataset.labels.numpy().flatten()
            else:
                return np.array(dataset.labels).flatten()
        
        # Check for .targets attribute (torchvision datasets)
        if hasattr(dataset, 'targets'):
            if hasattr(dataset.targets, 'numpy'):
                return dataset.targets.numpy().flatten()
            else:
                return np.array(dataset.targets).flatten()
        
        # TensorDataset - second element is labels
        if hasattr(dataset, 'tensors') and len(dataset.tensors) >= 2:
            return dataset.tensors[1].numpy().flatten()
        
        # Fallback: iterate through dataset to extract labels
        # This is slower but works for any dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if hasattr(label, 'item'):
                labels.append(label.item())
            elif hasattr(label, 'numpy'):
                labels.append(label.numpy().flatten()[0])
            else:
                labels.append(float(label))
        return np.array(labels).flatten()

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
