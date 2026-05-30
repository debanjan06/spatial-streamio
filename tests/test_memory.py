import os
import numpy as np
import pytest
from spatial_streamio.memory import MemoryMappedSpatialReader

@pytest.fixture
def dummy_spatial_file(tmp_path):
    """Generates a temporary binary mock dataset representing 1,000 production 3D points."""
    file_path = tmp_path / "mock_urban_scene.bin"
    # Updated to 6 features per point to match the production schema: 
    # (X, Y, Z, intensity, sem_class, ins_class)
    num_points = 1000
    num_features = 6
    mock_data = np.random.rand(num_points, num_features).astype(np.float32)
    mock_data.tofile(file_path) # Save as raw binary block
    return str(file_path), mock_data, num_points, num_features

def test_mmap_initialization(dummy_spatial_file):
    file_path, original_data, expected_points, expected_features = dummy_spatial_file
    
    # Explicitly pass num_features=6 to align with your production framework bounds
    reader = MemoryMappedSpatialReader(file_path, num_features=expected_features, dtype=np.float32)
    
    assert reader.num_points == expected_points
    assert reader.num_features == expected_features
    
    # Verify values match exactly without losing float precision
    np.testing.assert_array_almost_equal(reader.get_slice(0, 10), original_data[0:10])
    reader.close()