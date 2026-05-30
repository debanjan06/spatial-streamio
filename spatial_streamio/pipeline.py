import threading
import queue
import time
import numpy as np

class AsynchronousBufferPipeline:
    """
    Manages background prefetching threads to load spatial data slices 
    into memory ahead of execution, preventing hardware starvation loops.
    """
    def __init__(self, reader, batch_size, queue_size=2, timeout=10.0):
        """
        Parameters:
        - reader: An instance of MemoryMappedSpatialReader
        - batch_size: Number of points to stream per execution step
        - queue_size: Max number of pre-staged lookahead batches to keep in RAM
        - timeout: Maximum time (seconds) to wait for a batch before timing out
        """
        self.reader = reader
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.timeout = timeout
        
        self.running = False
        self.worker_thread = None
        
        # Calculate indices to partition the dataset into clean blocks
        self.indices = np.arange(0, self.reader.num_points, self.batch_size)
        self.current_step = 0

    def _prefetch_loop(self):
        """Background worker target loop that runs independently of the main execution loop."""
        while self.running:
            if self.current_step >= len(self.indices):
                # Epoch Completed: Put a None sentinel token into the queue to signal termination
                try:
                    self.queue.put(None, block=True, timeout=self.timeout)
                    self.current_step = 0 # Reset index step for the next potential epoch
                    break 
                except queue.Full:
                    continue # Try to insert the epoch end signal again if queue is temporarily blocked
                
            start_idx = self.indices[self.current_step]
            end_idx = min(start_idx + self.batch_size, self.reader.num_points)
            
            try:
                # Fetch memory view window from the reader
                data_slice = self.reader.get_slice(start_idx, end_idx)
                batch_buffer = np.copy(data_slice)
                
                # FIXED: block=True without a short timeout means the thread will sleep naturally
                # until the main training loop frees up a slot. No data is skipped.
                self.queue.put(batch_buffer, block=True, timeout=self.timeout)
                self.current_step += 1
                
            except queue.Full:
                # This block will now only hit if the main thread hung longer than the full timeout parameter
                print(f"Pipeline prefetch worker timed out waiting for an open queue slot.")
                time.sleep(0.1)
            except Exception as e:
                print(f"Pipeline prefetch worker encountered an error: {e}")
                self.running = False

    def start(self):
        """Spins up the background prefetch worker thread daemon."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.worker_thread.start()
            print("Asynchronous prefetch worker thread successfully running.")

    def next_batch(self):
        """
        Retrieves the next pre-staged data buffer block from the queue.
        Returns None when the epoch ends.
        """
        if not self.running:
            raise RuntimeWarning("Pipeline engine is not currently running. Call start() first.")
        try:
            return self.queue.get(block=True, timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("Data streaming starved: Prefetch thread failed to load buffers within timeline windows.")

    def stop(self):
        """Gracefully tears down execution loops and stops the background worker thread."""
        self.running = False
        if self.worker_thread:
            # Empty out remaining queue variables to break any blocking put() handles
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.worker_thread.join(timeout=2.0)
            print("Prefetch worker thread stopped cleanly.")
