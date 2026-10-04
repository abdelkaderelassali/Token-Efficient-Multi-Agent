"""Conservative preflight heuristic; estimates are never reported as measured tokens."""
import math


def compression_policy(source, compact_chars=None, consumers=2, mode="auto"):
    if mode not in ("auto", "always", "never"):
        raise ValueError("compression mode must be auto, always, or never")
    source_estimate = math.ceil(len(source) / 4)
    compact_estimate = math.ceil((compact_chars if compact_chars is not None else len(source) * .28) / 4)
    # Include estimated prompt overhead and generated summary in the compressor cost.
    overhead = source_estimate + 180 + 160
    net = consumers * (source_estimate - compact_estimate) - overhead
    worthwhile = len(source.split()) >= 300 and net >= 100
    apply = mode == "always" or (mode == "auto" and worthwhile)
    reason = ("Forced for an explicit experiment" if mode == "always" else
              "Disabled by policy" if mode == "never" else
              "Context is below 300 words" if len(source.split()) < 300 else
              "Estimated net saving is below 100 tokens" if net < 100 else
              "Context size and estimated net saving meet thresholds")
    return {"policy_version": "conditional-v1", "mode": mode, "should_compress": apply,
            "applied": False, "compressor_called": False, "reason": reason,
            "source_words": len(source.split()), "estimated_net_tokens_saved": net,
            "estimate_method": "Characters / 4, with prompt and output allowances; not measured token usage",
            "consumers": consumers}
