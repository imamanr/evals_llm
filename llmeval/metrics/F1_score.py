from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from .base import gtMetrics

class F1_score(gtMetrics):
    def compute(self, y_true, y_pred, average='weighted'):
        """
        Compute the F1 score for an intention classifier with scalable intentions.

        Parameters:
        true_labels (list): The true labels of the intentions.
        predicted_labels (list): The predicted labels of the intentions.
        average (str): The type of averaging to be performed on the data (default is 'weighted').

        Returns:
        float: The computed F1 score.
        """
        # Validate input
        if not y_true or not y_pred:
            raise ValueError("The true_labels and predicted_labels lists cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("The true_labels and predicted_labels lists must have the same length.")
        
        # Compute F1 score
        f1 = f1_score(y_true, y_pred, average=average)
        # print(f'Model {""} has follwing f1_scores:{f1}')
        return f1

if __name__ == '__main__':
    # Example usage
    true_labels = [0, 1, 2, 0, 1, 2]  # Example true labels
    predicted_labels = [0, 1, 2, 0, 2, 1]  # Example predicted labels
    
    f1 = F1_score.compute(true_labels, predicted_labels)
    print(f"F1 Score: {f1}")
