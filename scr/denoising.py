import numpy as np

class Denoiser:
    """
    A class to perform denoising using a multi-scale diffusion process.
    
    Attributes:
    ----------
    affinity_matrix : np.ndarray
        Affinity matrix for the diffusion process.
    data : np.ndarray
        Input data to be denoised.
    denoised_data : np.ndarray
        Denoised data.

    Methods:
    -------
    multi_scale_diffusion(scales):
        Applies a multi-scale diffusion process to the input data.
    """
    
    def __init__(self, affinity_matrix, data):
        """
        Constructs all the necessary attributes for the Denoiser object.
        
        Parameters:
        ----------
        affinity_matrix : np.ndarray
            Affinity matrix for the diffusion process.
        data : np.ndarray
            Input data to be denoised.
        """
        self.affinity_matrix = affinity_matrix
        self.data = data
    
    def multi_scale_diffusion(self, scales=[1, 5, 10]):
        """
        Applies a multi-scale diffusion process to the input data.
        
        Parameters:
        ----------
        scales : list, optional
            List of scales to apply in the diffusion process (default is [1, 5, 10]).
        
        Returns:
        -------
        np.ndarray
            Denoised data after applying the multi-scale diffusion process.
        """
        diffused_data = np.zeros_like(self.data)
        for scale in scales:
            row_sums = self.affinity_matrix.sum(axis=1)
            D = np.diag(1.0 / np.sqrt(row_sums))
            M = np.dot(D, np.dot(self.affinity_matrix, D))
            diffused_data += np.dot(M, self.data)
        self.denoised_data = diffused_data / len(scales)
        return self.denoised_data
