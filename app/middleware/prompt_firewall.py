class PromptFirewall:
    blocked = [
        "ignore previous instructions",
        "forget previous instructions",
        "jailbreak",
        "system override",
    ]

    def wrap(self, node_fn):
        def wrapper(state):
            q = state.get("question", "")
            if any(bad in q.lower() for bad in self.blocked):
                print("[FIREWALL] Blocked malicious prompt")
                state["answer"] = "Unsafe or malicious query detected."
                return state
            # pass through without changing anything
            return node_fn(state)
        return wrapper
