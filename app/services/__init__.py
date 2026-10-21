"""Business logic, framework-agnostic.

Nothing in here imports FastAPI. The HTTP routers call the same functions,
which is the whole point: the rules about what may be rendered, by whom, at
what cost, must not live in a web layer that could be replaced.
"""
