import os
import pytest
import numpy as np
from spatial_streamio.memory import MemoryMappedSpatialReader
from spatial_streamio.pipeline import AsynchronousBufferPipeline

@pytest.fixture
def dummy_binary_file(tmp_path):
    """Generates a temporary dummy 6-feature binary file for isolated testing."""
    file_path = tmp_path / "test_points.bin"
    # Create 500 points with 6 features each (uniform float32 layout)
    num_points = 500
    num_features = 6
    mock_data = np.random.rand(num_points, num_features).astype(np.float32)
    
    with open(file_path, "wb") as f:
        f.write(mock_data.tobytes())
        
    return str(file_path), num_points, num_features


def test_memory_mapped_reader_initialization(dummy_binary_file):
    """Verifies that the MemoryMappedSpatialReader correctly maps file properties."""
    file_path, expected_points, expected_features = dummy_binary_file
    
    reader = MemoryMappedSpatialReader(file_path, num_features=expected_features, dtype=np.float32)
    
    assert reader.num_points == expected_points
    assert reader.num_features == expected_features
    assert reader.mmap_array.shape == (expected_points, expected_features)
    
    # Test safe slicing window integrity
    slice_data = reader.get_slice(0, 10)
    assert slice_data.shape == (10, expected_features)
    
    reader.close()


def test_asynchronous_pipeline_streaming_and_sentinel(dummy_binary_file):
    """Ensures the pipeline streams batches correctly and delivers the None epoch token."""
    file_path, expected_points, expected_features = dummy_binary_file
    batch_size = 100
    
    reader = MemoryMappedSpatialReader(file_path, num_features=expected_features, dtype=np.float32)
    pipeline = AsynchronousBufferPipeline(reader, batch_size=batch_size, queue_size=2)
    
    pipeline.start()
    
    # Track streamed point quantities
    total_sampled_batches = 0
    
    while True:
        batch = pipeline.next_batch()
        if batch is None:  # Check for your production epoch completion sentinel token
            break
            
        assert batch.shape == (batch_size, expected_features)
        total_sampled_batches += 1
        
    # 500 points total / 100 batch size = 5 full extraction batches
    assert total_sampled_batches == 5
    
    pipeline.stop()
    reader.close()