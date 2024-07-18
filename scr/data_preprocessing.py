import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    A class to handle data preprocessing including loading and normalizing the data.
    
    Attributes:
    ----------
    data_path : str
        Path to the data file.
    readcounts : np.ndarray
        Array of read counts extracted from the data file.
    readcounts_normalized : np.ndarray
        Normalized read counts.

    Methods:
    -------
    load_data():
        Loads data from the specified path and returns read counts.
    normalize_data():
        Normalizes the read counts and returns the normalized data.
    """
    
    def __init__(self, data_path):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.
        
        Parameters:
        ----------
        data_path : str
            Path to the data file.
        """
        self.data_path = data_path
    
    def load_data(self):
        """
        Loads data from the specified path and returns read counts.

        Returns:
        -------
        np.ndarray
            Array of read counts extracted from the data file.
        """
        data = pd.read_csv(self.data_path, sep='\t')
        self.readcounts = data.drop(columns="cell").values
        return self.readcounts
    
    def normalize_data(self):
        """
        Normalizes the read counts using standard scaling and returns the normalized data.

        Returns:
        -------
        np.ndarray
            Normalized read counts.
        """
        self.scaler = StandardScaler()
        self.readcounts_normalized = self.scaler.fit_transform(self.readcounts)
        return self.readcounts_normalized
