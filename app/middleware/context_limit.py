class ContextLimitMiddleware:
    def __init__(self, max_chars=10000):
        self.max_chars = max_chars

    def wrap(self, node_fn):
        def wrapper(state):
            docs = state.get("documents", [])
            if docs:
                big_text = "\n\n".join([d.page_content for d in docs])
                if len(big_text) > self.max_chars:
                    print(f"[CONTEXT-LIMITER] Trimming context to {self.max_chars} chars")
                    # trim only first document to test
                    docs[0].page_content = big_text[:self.max_chars]

            return node_fn(state)  # pass the **full state**
        return wrapper
