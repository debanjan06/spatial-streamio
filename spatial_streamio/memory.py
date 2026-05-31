import os
import numpy as np


class MemoryMappedSpatialReader:
    """Handles out-of-core memory mapping for ultra-dense 3D point cloud datasets,

    allowing zero-copy data reads from binary disk files.
    """

    def __init__(self, file_path, num_features=4, dtype=np.float32):
        """Parameters:

        - file_path: Path to the raw uncompressed binary file (.npy or raw
        bytes)
        - num_features: Number of elements per point (e.g., 4 for X, Y, Z,
        Class_ID)
        - dtype: The data precision type stored on disk
        """
        self.file_path = file_path
        self.num_features = num_features
        self.dtype = np.dtype(dtype)
        self.element_size = self.dtype.itemsize

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target data file not found at: {self.file_path}")

        self.total_bytes = os.path.getsize(self.file_path)
        self.total_elements = self.total_bytes // self.element_size
        self.num_points = self.total_elements // self.num_features

        # Initialize the low-level virtual memory mapping handle
        self.mmap_array = np.memmap(
            self.file_path,
            dtype=self.dtype,
            mode="r",
            shape=(self.num_points, self.num_features),
        )
        print(f"Successfully mapped {self.file_path} to virtual memory.")
        print(f"Total structured points detected: {self.num_points:,}")

    def get_slice(self, start_idx, end_idx):
        """Retrieves a slice of spatial data without duplicating the underlying

        memory buffer block.
        """
        if start_idx < 0 or end_idx > self.num_points:
            raise IndexError(
                f"Slice range [{start_idx}:{end_idx}] exceeds data bounds [0:{self.num_points}]."
            )

        # NumPy slice over memmap returns a window view pointing straight to disk offsets
        return self.mmap_array[start_idx:end_idx]

    def close(self):
        """Closes the underlying memory map file descriptor handle."""
        if hasattr(self, "mmap_array"):
            del self.mmap_array
