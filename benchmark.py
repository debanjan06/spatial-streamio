import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from spatial_streamio.memory import MemoryMappedSpatialReader
from spatial_streamio.pipeline import AsynchronousBufferPipeline

# Check if your virtual environment has an active CUDA device available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking Execution Device Target: {device}")


class SpatialFeatureProcessor(nn.Module):
    """Simulates a dense feature processing layer typical of 3D ML backbones."""

    def __init__(self):
        super().__init__()
        # Input: 6 features (X, Y, Z, intensity, sem_class, ins_class)
        # Projecting to a higher dimension to simulate model layer processing overhead
        self.linear1 = nn.Linear(6, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)  # Output coordinates/classes

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def extract_multiple_ply_files(ply_folder, output_bin_path, max_files_to_sample=3):
    """Scans the training directory and compiles raw binary files seamlessly."""
    if os.path.exists(output_bin_path):
        print(
            f"Target production binary already exists at {output_bin_path}. Skipping extraction."
        )
        return

    ply_files = glob.glob(os.path.join(ply_folder, "*.ply"))
    if not ply_files:
        raise FileNotFoundError(
            f"No .ply files found inside the target folder: {ply_folder}"
        )

    print(f"Found {len(ply_files)} total .ply files in the training directory.")
    files_to_process = ply_files[:max_files_to_sample]

    with open(output_bin_path, "wb") as bin_file:
        for idx, ply_path in enumerate(files_to_process):
            print(
                f"[{idx + 1}/{len(files_to_process)}] Parsing 'testing' elements from {os.path.basename(ply_path)}..."
            )
            try:
                plydata = PlyData.read(ply_path)
                vertex = plydata["testing"]

                X = np.array(vertex["x"], dtype=np.float32)
                Y = np.array(vertex["y"], dtype=np.float32)
                Z = np.array(vertex["z"], dtype=np.float32)
                intensity = np.array(vertex["intensity"], dtype=np.float32)
                sem_class = np.array(vertex["sem_class"], dtype=np.float32)
                ins_class = np.array(vertex["ins_class"], dtype=np.float32)

                matrix_block = np.stack(
                    (X, Y, Z, intensity, sem_class, ins_class), axis=1
                )
                bin_file.write(matrix_block.tobytes())

            except Exception as e:
                print(f"Failed to parse file {ply_path} due to error: {e}")
                continue


def run_standard_baseline(file_path, batch_size, model):
    """Simulates a conventional sequential blocking loader running on real GPU tensors."""
    print("\n--- Running Standard Sequential Baseline (PyTorch Heavy Compute) ---")
    model.eval()
    start_time = time.time()

    element_size = np.dtype(np.float32).itemsize
    row_stride = 6 * element_size

    total_bytes = os.path.getsize(file_path)
    total_points = total_bytes // row_stride
    num_batches = total_points // batch_size

    with open(file_path, "rb") as f:
        for i in range(num_batches):
            # Blocking I/O read step occurs directly in the main thread loop execution timeline
            f.seek(i * batch_size * row_stride)
            raw_bytes = f.read(batch_size * row_stride)
            if not raw_bytes:
                break

            # Convert raw bytes to numpy array, then transfer to GPU memory space
            np_batch = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 6)

            # Safe duplicate fallback to clear the read-only flag warning for PyTorch conversion
            if not np_batch.flags.writeable:
                np_batch = np_batch.copy()

            tensor_batch = torch.from_numpy(np_batch).to(device)

            # Execute actual tensor graph calculations on the GPU device hardware
            with torch.no_grad():
                _ = model(tensor_batch)

    total_time = time.time() - start_time
    print(f"Standard Baseline completed in: {total_time:.4f} seconds")
    return total_time


def run_spatial_streamio_pipeline(file_path, batch_size, model):
    """Runs the optimized Asynchronous Memory-Mapped Buffer Pipeline with Epoch Signaling."""
    print("\n--- Running Spatial-StreamIO Pipeline (PyTorch Heavy Compute) ---")
    model.eval()
    start_time = time.time()

    reader = MemoryMappedSpatialReader(file_path, num_features=6, dtype=np.float32)
    pipeline = AsynchronousBufferPipeline(reader, batch_size=batch_size, queue_size=2)
    pipeline.start()

    time.sleep(0.2)  # Give the prefetch thread a head start to populate lookahead slots

    while True:
        # Pull pre-staged memory arrays cleanly from the background thread loop queue
        batch = pipeline.next_batch()

        # Epoch Signaling Check: Break the pipeline loop when the worker hits the None sentinel token
        if batch is None:
            print(
                "   -> Epoch end signaling token received safely. Terminating epoch processing run."
            )
            break

        # Explicitly toggle the flag status to satisfy the PyTorch tensor casting check
        batch.flags.writeable = True
        tensor_batch = torch.from_numpy(batch).to(device)

        # Execute actual tensor graph calculations on the GPU device hardware
        with torch.no_grad():
            _ = model(tensor_batch)

    total_time = time.time() - start_time
    pipeline.stop()
    reader.close()

    print(f"Spatial-StreamIO Pipeline completed in: {total_time:.4f} seconds")
    return total_time


if __name__ == "__main__":
    training_ply_folder = "data"
    target_bin_file = os.path.join("data", "production_combined_urban_points.bin")

    try:
        extract_multiple_ply_files(
            training_ply_folder, target_bin_file, max_files_to_sample=3
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        exit(1)

    total_bytes = os.path.getsize(target_bin_file)
    total_points = total_bytes // (6 * np.dtype(np.float32).itemsize)

    # Run over large batches to maximize matrix computation pipelines
    BATCH_SIZE = 250000

    # Initialize the PyTorch model and ship it to your graphics hardware pipeline memory space
    spatial_model = SpatialFeatureProcessor().to(device)

    # 2. Run the actual heavy benchmarking runs
    baseline_dur = run_standard_baseline(target_bin_file, BATCH_SIZE, spatial_model)
    pipeline_dur = run_spatial_streamio_pipeline(
        target_bin_file, BATCH_SIZE, spatial_model
    )

    # 3. Calculate actual efficiency gains
    speedup = (baseline_dur - pipeline_dur) / baseline_dur * 100
    print("\n=============================================")
    print("      PRODUCTION WORKLOAD BENCHMARK RESULTS  ")
    print("=============================================")
    print(f"Total Points Sampled:             {total_points:,}")
    print(f"Sequential Loader Step Duration:  {baseline_dur:.4f}s")
    print(f"Spatial-StreamIO Step Duration:   {pipeline_dur:.4f}s")
    print(f"Data Pipeline Efficiency Gain:    {speedup:.2f}% improvement")
    print("=============================================")
