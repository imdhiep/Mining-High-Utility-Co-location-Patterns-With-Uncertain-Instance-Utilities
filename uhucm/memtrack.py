"""Peak memory of a code block measured with tracemalloc (NumPy buffers are traced). Used in separate runs from the
timing runs because tracing slows execution."""
import tracemalloc
def start():
    tracemalloc.start(); tracemalloc.reset_peak()
def stop_mb():
    cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return pk / 2 ** 20
