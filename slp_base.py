"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators.
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
        Initialize the SLP estimator.

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
        tol : float
            Minimum loss improvement required for early stopping
        n_iter_no_change : int
            Number of iterations without improvement before stopping
        early_stopping : bool
            Whether to stop training when loss stops improving
        batch_size : int or None
            Size of mini-batches; None uses full-batch training
        momentum : float
            Momentum factor in the range [0, 1)
        hidden_layer_sizes : sequence of int or None
            Sizes for one or more hidden layers. If None, hidden_layer_size is used.
        """
        if hidden_layer_sizes is None and isinstance(
            hidden_layer_size, (list, tuple, np.ndarray)
        ):
            hidden_layer_sizes = hidden_layer_size

        if hidden_layer_sizes is None:
            hidden_layer_sizes_tuple = (int(hidden_layer_size),)
        else:
            hidden_layer_sizes_tuple = tuple(int(size) for size in hidden_layer_sizes)

        if not hidden_layer_sizes_tuple:
            raise ValueError("hidden_layer_sizes must contain at least one layer")
        if any(size <= 0 for size in hidden_layer_sizes_tuple):
            raise ValueError("hidden layer sizes must be positive")
        if momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in the range [0, 1)")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if activation not in {"relu", "tanh", "logistic"}:
            raise ValueError("activation must be 'relu', 'tanh', or 'logistic'")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive")

        self.momentum = momentum
        self.batch_size = batch_size
        self.hidden_layer_sizes: tuple[int, ...] = hidden_layer_sizes_tuple
        self.hidden_layer_size: int = hidden_layer_sizes_tuple[0]
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.tol: float = tol
        self.n_iter_no_change: int = n_iter_no_change
        self.early_stopping: bool = early_stopping
        self.n_iter_: int = 0
        # Backward-compatible aliases for the first and final layer parameters.
        self.W1_: Optional[NDArray[np.floating]] = None
        self.b1_: Optional[NDArray[np.floating]] = None
        self.W2_: Optional[NDArray[np.floating]] = None
        self.b2_: Optional[NDArray[np.floating]] = None
        self.weights_: Optional[list[NDArray[np.floating]]] = None
        self.biases_: Optional[list[NDArray[np.floating]]] = None
        self.velocity_weights_: Optional[list[NDArray[np.floating]]] = None
        self.velocity_biases_: Optional[list[NDArray[np.floating]]] = None
        self.loss_curve_: list[float] = []

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases.
        """
        layer_sizes = (
            self.n_features_,
            *self.hidden_layer_sizes,
            self.n_outputs_,
        )
        self.weights_ = []
        self.biases_ = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights_.append(np.random.randn(fan_in, fan_out) * 0.05)
            self.biases_.append(np.zeros(fan_out))

        self.W1_ = self.weights_[0]
        self.b1_ = self.biases_[0]
        self.W2_ = self.weights_[-1]
        self.b2_ = self.biases_[-1]

    def _initialize_velocities(self) -> None:
        """
        Initialize momentum velocity arrays for each weight and bias matrix.
        """
        self.velocity_weights_ = [np.zeros_like(weight) for weight in self.weights_]
        self.velocity_biases_ = [np.zeros_like(bias) for bias in self.biases_]
        self.vW1_ = self.velocity_weights_[0]
        self.vb1_ = self.velocity_biases_[0]
        self.vW2_ = self.velocity_weights_[-1]
        self.vb2_ = self.velocity_biases_[-1]

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        list[NDArray[np.floating]],
        list[NDArray[np.floating]],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
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

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
