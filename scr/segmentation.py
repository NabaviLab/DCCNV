import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import numpy as np

numpy2ri.activate()
DNAcopy = importr('DNAcopy')

robjects.r('''
    library(DNAcopy)

    perform_segmentation <- function(signal) {
        chrom <- rep("1", length(signal))
        position <- 1:length(signal)
        genomdat <- signal

        cna_obj <- CNA(cbind(genomdat), chrom, position, data.type="logratio")
        segmented <- segment(cna_obj, alpha=0.01, min.width=2, undo.splits="sdundo", undo.SD=2)

        return(segmented$output)
    }
''')

perform_segmentation = robjects.globalenv['perform_segmentation']

class Segmenter:
    """
    A class to perform segmentation using Circular Binary Segmentation (CBS).
    
    Methods:
    -------
    segment_signal_with_CBS(signal):
        Segments the input signal using CBS.
    segmented_to_array(segmented, signal_length):
        Converts the segmented result to an array format.
    min_max_normalize(array):
        Normalizes the input array using min-max normalization.
    plot_segmented_vs_ground_truth(segmented, ground_truth, cell_idx):
        Plots the segmented signal against the ground truth.
    """
    
    @staticmethod
    def segment_signal_with_CBS(signal):
        """
        Segments the input signal using Circular Binary Segmentation (CBS).
        
        Parameters:
        ----------
        signal : np.ndarray
            Input signal to be segmented.
        
        Returns:
        -------
        pandas.DataFrame
            Segmented result as a DataFrame.
        """
        segmented = perform_segmentation(robjects.FloatVector(signal))
        with localconverter(robjects.default_converter + pandas2ri.converter):
            df = robjects.conversion.rpy2py(segmented)
        return df
    
    @staticmethod
    def segmented_to_array(segmented, signal_length):
        """
        Converts the segmented result to an array format.
        
        Parameters:
        ----------
        segmented : pandas.DataFrame
            Segmented result as a DataFrame.
        signal_length : int
            Length of the input signal.
        
        Returns:
        -------
        np.ndarray
            Segmented result in array format.
        """
        result = np.zeros(signal_length)
        for record in segmented.itertuples(index=False):
            start, end, value = record[2]-1, record[3], record[5]
            result[start:end] = value
        return result
    
    @staticmethod
    def min_max_normalize(array):
        """
        Normalizes the input array using min-max normalization.
        
        Parameters:
        ----------
        array : np.ndarray
            Input array to be normalized.
        
        Returns:
        -------
        np.ndarray
            Normalized array.
        """
        if np.any(np.isnan(array)):
            array = np.nan_to_num(array)
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val - min_val == 0:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)
    
    @staticmethod
    def plot_segmented_vs_ground_truth(segmented, ground_truth, cell_idx):
        """
        Plots the segmented signal against the ground truth.
        
        Parameters:
        ----------
        segmented : np.ndarray
            Segmented signal.
        ground_truth : np.ndarray
            Ground truth signal.
        cell_idx : int
            Index of the cell being plotted.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(segmented, label='Segmented Signal', linewidth=1)
        plt.plot(ground_truth, label='Ground Truth', linewidth=1)
        plt.legend()
        plt.title(f"Comparison for Cell {cell_idx}")
        plt.xlabel('Genomic Position')
        plt.ylabel('Normalized Value')
        plt.show()
