import torch

class BaseMetric():

    """
    Base class for defining various types of metrics.

    Attributes:
        vendor (str): The vendor of the model.
        model_id (str): The identifier for the model.
        name (str): The name of the metric.
    """

    def __init__(self, vendor, model_id) -> None:

        """
        Initializes the BaseMetric class with vendor and model ID.

        Args:
            vendor (str): The vendor of the model.
            model_id (str): The identifier for the model.
        """
        self.vendor = vendor
        self.model_id = model_id
        self.name = ''

    def compute(self):
        """
        Abstract method to compute the metric. 
        Must be implemented by subclasses.
        """
        pass


class TemporalMetrics(BaseMetric):

    def __init__(self) -> None:
        """
        Class for computing temporal metrics. Inherits from BaseMetric.
        """
        pass

    def compute(self,frame1, frame2):
        pass

class SpatialMetrics(BaseMetric):

    def __init__(self) -> None:
        """
        Initializes the SpatialMetrics class.
        """
        super().__init__()

    def compute(self, frame1):
        """
        Computes the spatial metric for a frame.

        Args:
            frame1 (torch.Tensor): The frame to compute the spatial metric on.
        """
        pass


class gtMetrics(BaseMetric):
    """
    Class for computing ground truth metrics. Inherits from BaseMetric.
    """

    def __init__(self) -> None:
        """
        Initializes the gtMetrics class.
        """
        pass

    def compute(self, y_true, y_pred):
        """
        Computes the metric between true and predicted values.

        Args:
            y_true (torch.Tensor): The ground truth values.
            y_pred (torch.Tensor): The predicted values.
        """
        pass

    def compute_batch(self, y_true_list, y_pred_list):
        """
        Computes the metric for a batch of true and predicted values.

        Args:
            y_true_list (list of torch.Tensor): A list of ground truth values.
            y_pred_list (list of torch.Tensor): A list of predicted values.
        """
        pass

class embMetrics(BaseMetric):

    def __init__(self, model_text, model_image) -> None:
        """
        Initializes the embMetrics class with text and image models.

        Args:
            model_text (torch.nn.Module): The model used for text encoding.
            model_image (torch.nn.Module): The model used for image encoding.
        """

        self.model_text = model_text
        self.model_image = model_image

    def compute_textEmb(self, x):
        """
        Computes the text embedding.

        Args:
            x (str): The text input.

        Returns:
            torch.Tensor: The encoded text embedding.
        """
        return self.model_text.encode(x)
    
    def compute_imageEmb(self, x):
        """
        Computes the image embedding.

        Args:
            x (torch.Tensor): The image input.

        Returns:
            torch.Tensor: The encoded image embedding.
        """
        return self.model_image.endcode(x)
    
    def compute(self, text, image):
        """
        Computes the metric between text and image embeddings.

        Args:
            text (str): The text input.
            image (torch.Tensor): The image input.
        """
        pass

    def compute(self, text_1, text_2):
        """
        Computes the metric between two text embeddings.

        Args:
            text_1 (str): The first text input.
            text_2 (str): The second text input.
        """
        pass
    
    def compute(self, image_1, image_2):
        """
        Computes the metric between two image embeddings.

        Args:
            image_1 (torch.Tensor): The first image input.
            image_2 (torch.Tensor): The second image input.
        """
        
        pass

    def compute_batch(self, text, image):
        """
        Computes the metric for a batch of text and image embeddings.

        Args:
            text (list of str): A list of text inputs.
            image (list of torch.Tensor): A list of image inputs.

        Returns:
            list: The computed metrics for the batch.
        """
                
        met = []
        for t,m in zip(text,image):
            met += self.compute(t,m)
        return met
    
    def compute_batch(self, text_1, text_2):
        """
        Computes the metric for a batch of text embeddings.

        Args:
            text_1 (list of str): A list of first text inputs.
            text_2 (list of str): A list of second text inputs.

        Returns:
            list: The computed metrics for the batch.
        """
                
        met = 0
        for t1,t2 in zip(text_1,text_2):
            met += self.compute(t1,t2)
        return met
        






