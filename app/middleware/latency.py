# app/middleware/latency.py
import time

class LatencyMiddleware:
    def wrap(self, node_fn):
        def wrapper(state):
            start = time.time()
            result = node_fn(state)  # execute the node with full state
            end = time.time()

            print(f"[LATENCY] Node {node_fn.__name__} executed in {round(end - start, 3)} seconds")
            return result
        return wrapper
