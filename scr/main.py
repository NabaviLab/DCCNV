import numpy as np
from data_preprocessing import DataPreprocessor
from graph_analysis import GraphAnalyzer
from denoising import Denoiser
from contrastive_learning import ContrastiveLearningModel
from segmentation import Segmenter

def main(data_path):
    """
    Main function to run the entire DCCNV pipeline.
    
    Parameters:
    ----------
    data_path : str
        Path to the input data file.
    """
    # Data Preprocessing
    preprocessor = DataPreprocessor(data_path)
    readcounts = preprocessor.load_data()
    readcounts_normalized = preprocessor.normalize_data()
    
    # Graph Analysis
    analyzer = GraphAnalyzer(readcounts_normalized)
    knn_graph = analyzer.construct_knn_graph(max_k=10)
    affinity_matrix = analyzer.compute_affinity_matrix()
    
    # Denoising
    denoiser = Denoiser(affinity_matrix, readcounts_normalized)
    denoised_data = denoiser.multi_scale_diffusion(scales=[1, 5, 10])
    
    # Contrastive Learning
    input_shape = readcounts_normalized.shape[1]
    model = ContrastiveLearningModel(input_shape)
    contrastive_model = model.create_contrastive_model()
    
    # Generate pairs for training
    hard_negatives = get_hard_negatives(readcounts_normalized, denoised_data)
    positive_pairs = [(readcounts_normalized[i], denoised_data[i]) for i in range(len(readcounts_normalized))]
    negative_pairs = [(readcounts_normalized[i], denoised_data[j]) for i, j in enumerate(hard_negatives[:, 0])]
    
    paired_data = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    train_data, test_data, labels_train, labels_test = train_test_split(paired_data, labels, test_size=0.25)
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)
    
    # Compile and train the contrastive model
    model.compile_and_train(train_data, np.array(labels_train), test_data, np.array(labels_test))
    
    # Segmentation
    segmenter = Segmenter()
    segmented_signals = []
    for i in range(denoised_data.shape[0]):
        segmented = segmenter.segment_signal_with_CBS(denoised_data[i])
        segmented_array = segmenter.segmented_to_array(segmented, len(denoised_data[i]))
        normalized_segmented = segmenter.min_max_normalize(segmented_array)
        segmented_signals.append(normalized_segmented)
    
    # Plot comparison for the first cell
    ground_truth = preprocessor.load_ground_truth("path/to/ground_truth.tsv")
    segmenter.plot_segmented_vs_ground_truth(segmented_signals[0], ground_truth[0], 0)

if __name__ == "__main__":
    main("path/to/data.tsv")
