import tensorflow as tf
import numpy as np
import json

class TensorFlowImageClassifier:
    def __init__(self, model_path, label_path=None):
        
        """
        Initialize the TensorFlow image classifier
        
        Args:
            model_path: Path to the saved TensorFlow model
            labels: List of class labels
        """
        self.model = self._load_model(model_path)
        self.labels = self.load_labels(label_path)
    
    @staticmethod
    def _load_model(model_path):
        try:
            # Try loading as Keras model first
            return tf.keras.models.load_model(model_path)
        except:
            try:
                return tf.saved_model.load(model_path)
            except Exception as e:
                print(f"Model loading failed: {str(e)}")
                raise

    @staticmethod
    def load_labels(path):
        with open(path,'r') as f:
            labels = json.load(f)
        return labels

    def predict(self, img_array):
        
        """
        Make prediction on preprocessed image
        
        Args:
            img_array: Preprocessed image array with shape [1, height, width, channels]
            
        Returns:
            class_id: Predicted class ID
            probabilities: List of class probabilities
        """
        
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
            
        predictions = self.model.predict(img_array)
        
        
        if len(predictions.shape) == 2:  
            class_id = np.argmax(predictions[0])
            probabilities = predictions[0].tolist()
        else:  
            class_id = np.argmax(predictions)
            probabilities = predictions.flatten().tolist()
            
        return class_id, probabilities

