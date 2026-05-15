"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple feed-forward neural-network classifier.

    Compatible interface with sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_size: int | Sequence[int] = 100,
        activation: str = "logistic",
        learning_rate: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        early_stopping: bool = False,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        hidden_layer_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Initialize the SLP classifier.

        Parameters:
        -----------
        hidden_layer_size : int or sequence of int
            Number of neurons in the hidden layer, or layer sizes for a deep network
        activation : str
            Hidden-layer activation function ('logistic', 'tanh', 'relu'), default='logistic'
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        batch_size : int or None
            Size of mini-batches for stochastic gradient descent
        momentum : float
            Momentum factor in the range [0, 1)
        hidden_layer_sizes : sequence of int or None
            Sizes for one or more hidden layers
        """
        super().__init__(
            hidden_layer_size,
            activation,
            learning_rate,
            max_iter,
            random_state,
            tol,
            n_iter_no_change,
            early_stopping,
            batch_size,
            momentum,
            hidden_layer_sizes,
        )

        self.classes_: Optional[NDArray[np.int_]] = None
        self.n_outputs_: Optional[int] = None

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        hidden_zs, activations, z2, y_pred : tuple
            Hidden pre-activations, layer activations, logits, and probabilities
        """
        activation_func, _ = self._get_activation_function()
        hidden_zs = []
        activations = [X]

        for weight, bias in zip(self.weights_[:-1], self.biases_[:-1]):
            z = activations[-1] @ weight + bias
            hidden_zs.append(z)
            activations.append(activation_func(z))

        z2 = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        y_pred = softmax(z2)
        return hidden_zs, activations, z2, y_pred

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: list[NDArray[np.floating]],
        a1: list[NDArray[np.floating]],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
    ]:
        """
        Perform backpropagation to compute gradients.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        gradients_w, gradients_b : tuple of lists
            Gradients for each weight matrix and bias vector
        """
        _, activation_derivative = self._get_activation_function()
        gradients_w = [np.zeros_like(weight) for weight in self.weights_]
        gradients_b = [np.zeros_like(bias) for bias in self.biases_]

        delta = y_pred - y
        gradients_w[-1] = a1[-1].T @ delta
        gradients_b[-1] = np.sum(delta, axis=0)

        for layer_idx in range(len(self.hidden_layer_sizes) - 1, -1, -1):
            delta = (delta @ self.weights_[layer_idx + 1].T) * activation_derivative(
                z1[layer_idx]
            )
            gradients_w[layer_idx] = a1[layer_idx].T @ delta
            gradients_b[layer_idx] = np.sum(delta, axis=0)

        return gradients_w, gradients_b

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return float(loss)

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_outputs_ = len(self.classes_)
        y_one_hot = np.zeros((n_samples, self.n_outputs_))
        y_one_hot[np.arange(n_samples), y_encoded] = 1
        self.n_features_ = n_features
        self._initialize_weights()
        self._initialize_velocities()

        self.loss_curve_ = []
        self.n_iter_ = 0
        best_loss = np.inf
        no_improvement_count = 0
        batch_size = (
            n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        )
        for iteration in range(self.max_iter):
            z1, a1, z2, y_pred = self._forward_propagation(X)
            loss = self._compute_loss(y_one_hot, y_pred)
            self.loss_curve_.append(loss)
            self.n_iter_ = iteration + 1
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= self.n_iter_no_change:
                    break
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y_one_hot[batch_indices]
                z1, a1, z2, y_pred = self._forward_propagation(X_batch)
                gradients_w, gradients_b = self._backward_propagation(
                    X_batch, y_batch, z1, a1, z2, y_pred
                )
                for layer_idx in range(len(self.weights_)):
                    self.velocity_weights_[layer_idx] *= self.momentum
                    self.velocity_weights_[layer_idx] -= (
                        self.learning_rate * gradients_w[layer_idx]
                    )
                    self.velocity_biases_[layer_idx] *= self.momentum
                    self.velocity_biases_[layer_idx] -= (
                        self.learning_rate * gradients_b[layer_idx]
                    )
                    self.weights_[layer_idx] += self.velocity_weights_[layer_idx]
                    self.biases_[layer_idx] += self.velocity_biases_[layer_idx]

        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        if self.W1_ is None or self.W2_ is None:
            raise ValueError("This SimpleSLPClassifier instance is not fitted yet.")
        X = np.asarray(X, dtype=float)
        _, _, _, y_pred = self._forward_propagation(X)
        return y_pred

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        if self.classes_ is None:
            raise ValueError("This SimpleSLPClassifier instance is not fitted yet.")
        y_proba = self.predict_proba(X)
        class_indices = np.argmax(y_proba, axis=1)
        return self.classes_[class_indices]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
