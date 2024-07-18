import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, callbacks

class ContrastiveLearningModel:
    """
    A class to define and train a contrastive learning model using a Siamese network architecture.
    
    Attributes:
    ----------
    input_shape : tuple
        Shape of the input data.

    Methods:
    -------
    create_base_network():
        Defines the base network architecture.
    create_contrastive_model():
        Creates the contrastive learning model using the Siamese network architecture.
    compile_and_train(train_data, labels_train, val_data, labels_val):
        Compiles and trains the contrastive learning model.
    """
    
    def __init__(self, input_shape):
        """
        Constructs all the necessary attributes for the ContrastiveLearningModel object.
        
        Parameters:
        ----------
        input_shape : tuple
            Shape of the input data.
        """
        self.input_shape = input_shape
    
    def create_base_network(self):
        """
        Defines the base network architecture for the Siamese network.
        
        Returns:
        -------
        tensorflow.keras.Model
            Base network model.
        """
        input = layers.Input(shape=self.input_shape)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        return Model(input, x)
    
    def create_contrastive_model(self):
        """
        Creates the contrastive learning model using the Siamese network architecture.
        
        Returns:
        -------
        tensorflow.keras.Model
            Contrastive learning model.
        """
        base_network = self.create_base_network()
        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
        outputs = layers.Dense(1, activation='sigmoid')(distance)
        self.model = Model(inputs=[input_a, input_b], outputs=outputs)
        return self.model
    
    def compile_and_train(self, train_data, labels_train, val_data, labels_val):
        """
        Compiles and trains the contrastive learning model.
        
        Parameters:
        ----------
        train_data : list of np.ndarray
            Training data consisting of pairs of input vectors.
        labels_train : np.ndarray
            Labels for the training data.
        val_data : list of np.ndarray
            Validation data consisting of pairs of input vectors.
        labels_val : np.ndarray
            Labels for the validation data.
        
        Returns:
        -------
        tensorflow.keras.callbacks.History
            History object containing details about the training process.
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        history = self.model.fit(train_data, labels_train, epochs=100, validation_data=(val_data, labels_val), batch_size=32, callbacks=[lr_scheduler, early_stopping])
        return history
