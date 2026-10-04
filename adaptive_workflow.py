"""Three-arm adaptive compressor evaluation; historical protocols remain callable."""
from requirement_review import PROTOCOL
from graph_verified_lossless_workflow import run_pair as run_verified_pair, scenarios, MODEL


def run_pair(scenario, protocol=PROTOCOL, **kwargs):
    return run_verified_pair(scenario, adaptive=True, adaptive_protocol=protocol, **kwargs)
