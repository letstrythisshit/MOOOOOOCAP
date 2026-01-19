"""
Temporal filtering and smoothing for motion capture data.

Provides various filtering algorithms to reduce jitter and smooth
landmark trajectories over time:
- One Euro Filter (adaptive low-pass)
- Kalman Filter (optimal estimation)
- Exponential Smoothing
- Savitzky-Golay Filter (polynomial smoothing)
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


class TemporalFilter(ABC):
    """Abstract base class for temporal filters."""

    @abstractmethod
    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Apply filter to a single value/frame.

        Args:
            value: Input value (can be scalar, 1D, or 2D array)
            timestamp: Optional timestamp for time-aware filtering

        Returns:
            Filtered value with same shape as input
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the filter state."""
        pass

    @abstractmethod
    def filter_batch(self, values: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply filter to a batch of values.

        Args:
            values: Input values with shape (num_frames, ...)
            timestamps: Optional timestamps for each frame

        Returns:
            Filtered values with same shape as input
        """
        pass


class OneEuroFilter(TemporalFilter):
    """
    One Euro Filter - An adaptive low-pass filter.

    This filter adapts its cutoff frequency based on the speed of change,
    providing smooth results for slow movements while preserving quick movements.

    Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
    CHI '12.

    Attributes:
        min_cutoff: Minimum cutoff frequency (lower = more smoothing)
        beta: Speed coefficient (higher = less lag during fast movement)
        d_cutoff: Cutoff frequency for derivative calculation
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        """Initialize the One Euro Filter."""
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # State variables
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate smoothing factor alpha."""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(
        self,
        a: float,
        x: np.ndarray,
        x_prev: np.ndarray
    ) -> np.ndarray:
        """Apply exponential smoothing."""
        return a * x + (1 - a) * x_prev

    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Apply One Euro Filter to a value."""
        value = np.atleast_1d(value).astype(np.float64)

        # Handle first frame
        if self._x_prev is None:
            self._x_prev = value.copy()
            self._dx_prev = np.zeros_like(value)
            self._t_prev = timestamp if timestamp is not None else 0.0
            return value

        # Calculate time delta
        t = timestamp if timestamp is not None else self._t_prev + 1/30.0
        t_e = max(t - self._t_prev, 1e-6)

        # Calculate derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (value - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # Calculate cutoff frequency based on speed
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Apply smoothing
        a = self._smoothing_factor(t_e, np.mean(cutoff))
        x_hat = self._exponential_smoothing(a, value, self._x_prev)

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self):
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter_batch(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply filter to a batch of values."""
        self.reset()
        result = np.zeros_like(values)

        for i in range(len(values)):
            t = timestamps[i] if timestamps is not None else None
            result[i] = self.filter(values[i], t)

        return result


class KalmanFilter(TemporalFilter):
    """
    Kalman Filter for optimal state estimation.

    A linear Kalman filter that estimates both position and velocity,
    providing smooth trajectories with minimal lag.

    Attributes:
        process_noise: Process noise covariance (Q)
        measurement_noise: Measurement noise covariance (R)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dim: int = 1,
    ):
        """Initialize the Kalman Filter."""
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dim = dim

        # State: [position, velocity]
        self._x: Optional[np.ndarray] = None  # State estimate
        self._P: Optional[np.ndarray] = None  # Error covariance
        self._initialized = False

    def _initialize(self, value: np.ndarray):
        """Initialize filter state from first measurement."""
        value = value.flatten()
        self.dim = len(value)

        # State vector: [position (dim), velocity (dim)]
        self._x = np.zeros(2 * self.dim)
        self._x[:self.dim] = value

        # Error covariance
        self._P = np.eye(2 * self.dim) * 1.0

        self._initialized = True

    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Apply Kalman Filter to a value."""
        original_shape = value.shape
        value = np.atleast_1d(value).flatten().astype(np.float64)

        if not self._initialized:
            self._initialize(value)
            return value.reshape(original_shape)

        dt = 1/30.0  # Assume 30 FPS if no timestamp

        # State transition matrix
        F = np.eye(2 * self.dim)
        F[:self.dim, self.dim:] = np.eye(self.dim) * dt

        # Process noise
        Q = np.eye(2 * self.dim) * self.process_noise
        Q[:self.dim, :self.dim] *= dt**4 / 4
        Q[self.dim:, self.dim:] *= dt**2
        Q[:self.dim, self.dim:] *= dt**3 / 2
        Q[self.dim:, :self.dim] *= dt**3 / 2

        # Measurement matrix (we only observe position)
        H = np.zeros((self.dim, 2 * self.dim))
        H[:self.dim, :self.dim] = np.eye(self.dim)

        # Measurement noise
        R = np.eye(self.dim) * self.measurement_noise

        # Predict
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        # Update
        y = value - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = x_pred + K @ y
        self._P = (np.eye(2 * self.dim) - K @ H) @ P_pred

        return self._x[:self.dim].reshape(original_shape)

    def reset(self):
        """Reset filter state."""
        self._x = None
        self._P = None
        self._initialized = False

    def filter_batch(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply filter to a batch of values."""
        self.reset()
        result = np.zeros_like(values)

        for i in range(len(values)):
            t = timestamps[i] if timestamps is not None else None
            result[i] = self.filter(values[i], t)

        return result


class ExponentialFilter(TemporalFilter):
    """
    Simple exponential smoothing filter.

    A basic low-pass filter using exponential smoothing.
    Fast and efficient but not adaptive.

    Attributes:
        alpha: Smoothing factor (0-1, lower = more smoothing)
    """

    def __init__(self, alpha: float = 0.5):
        """Initialize the Exponential Filter."""
        self.alpha = np.clip(alpha, 0.01, 1.0)
        self._x_prev: Optional[np.ndarray] = None

    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Apply exponential smoothing."""
        value = np.atleast_1d(value).astype(np.float64)

        if self._x_prev is None:
            self._x_prev = value.copy()
            return value

        result = self.alpha * value + (1 - self.alpha) * self._x_prev
        self._x_prev = result
        return result

    def reset(self):
        """Reset filter state."""
        self._x_prev = None

    def filter_batch(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply filter to a batch of values using optimized scipy function."""
        # Use scipy's uniform filter for efficiency
        axis = 0
        return uniform_filter1d(
            values.astype(np.float64),
            size=int(1/self.alpha),
            axis=axis,
            mode='nearest'
        )


class SavitzkyGolayFilter(TemporalFilter):
    """
    Savitzky-Golay polynomial smoothing filter.

    Fits successive windows to a polynomial and uses the center value.
    Preserves higher moments (better at keeping peaks/features).

    Attributes:
        window_length: Length of the filter window (must be odd)
        poly_order: Order of the polynomial
    """

    def __init__(self, window_length: int = 7, poly_order: int = 2):
        """Initialize the Savitzky-Golay Filter."""
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        self.window_length = window_length
        self.poly_order = min(poly_order, window_length - 1)

        # Buffer for streaming mode
        self._buffer: list = []
        self._max_buffer_size = window_length * 2

    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """Apply Savitzky-Golay filter in streaming mode."""
        value = np.atleast_1d(value).astype(np.float64)

        self._buffer.append(value.copy())

        # Keep buffer size manageable
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer[-self._max_buffer_size:]

        # Need at least window_length frames
        if len(self._buffer) < self.window_length:
            return value

        # Apply filter to buffer
        buffer_array = np.array(self._buffer)
        filtered = savgol_filter(
            buffer_array,
            window_length=self.window_length,
            polyorder=self.poly_order,
            axis=0,
            mode='nearest'
        )

        return filtered[-1]

    def reset(self):
        """Reset filter state."""
        self._buffer = []

    def filter_batch(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter to a batch."""
        if len(values) < self.window_length:
            return values

        return savgol_filter(
            values.astype(np.float64),
            window_length=self.window_length,
            polyorder=self.poly_order,
            axis=0,
            mode='nearest'
        )


class LandmarkFilterBank:
    """
    A bank of filters for all landmarks.

    Manages individual filters for each landmark dimension,
    allowing for per-landmark filtering.
    """

    def __init__(
        self,
        num_landmarks: int,
        filter_class: type = OneEuroFilter,
        **filter_kwargs
    ):
        """
        Initialize filter bank.

        Args:
            num_landmarks: Number of landmarks to filter
            filter_class: Filter class to use
            **filter_kwargs: Arguments passed to filter constructor
        """
        self.num_landmarks = num_landmarks
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs

        # Create filters for x, y, z coordinates of each landmark
        self._filters: list[list[TemporalFilter]] = []
        for _ in range(num_landmarks):
            self._filters.append([
                filter_class(**filter_kwargs)
                for _ in range(3)  # x, y, z
            ])

    def filter(
        self,
        landmarks: np.ndarray,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter landmarks.

        Args:
            landmarks: Array of shape (num_landmarks, 3)
            timestamp: Optional timestamp

        Returns:
            Filtered landmarks with same shape
        """
        if landmarks is None:
            return None

        result = np.zeros_like(landmarks)

        for i in range(min(len(landmarks), self.num_landmarks)):
            for j in range(3):
                result[i, j] = self._filters[i][j].filter(
                    np.array([landmarks[i, j]]),
                    timestamp
                )[0]

        return result

    def reset(self):
        """Reset all filters."""
        for landmark_filters in self._filters:
            for f in landmark_filters:
                f.reset()


def create_filter(
    filter_type: str,
    **kwargs
) -> TemporalFilter:
    """
    Factory function to create a temporal filter.

    Args:
        filter_type: Type of filter ('one_euro', 'kalman', 'exponential', 'savgol')
        **kwargs: Filter-specific parameters

    Returns:
        Configured temporal filter instance
    """
    filter_map = {
        'one_euro': OneEuroFilter,
        'kalman': KalmanFilter,
        'exponential': ExponentialFilter,
        'savgol': SavitzkyGolayFilter,
        'savitzky_golay': SavitzkyGolayFilter,
    }

    filter_type = filter_type.lower().replace('-', '_')
    if filter_type not in filter_map:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filter_map[filter_type](**kwargs)
