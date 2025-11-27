class StepCounterMiddleware:
    step = 0

    def wrap(self, node_fn):
        def wrapper(state):
            StepCounterMiddleware.step += 1
            print(f"[STEP] Executing step {StepCounterMiddleware.step}")
            return node_fn(state)
        return wrapper
