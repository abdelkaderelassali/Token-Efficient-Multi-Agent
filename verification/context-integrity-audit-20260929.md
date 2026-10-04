# Compressed-context audit — 2026-09-29

## Verdict

After this audit, the dashboard gained a **Complete source** default
(`validated-v8-lossless`). It copies the full brief, all carrier and method
names, every structured field and every supplier term into downstream prompts.
Its measured detailed Llama 3 pair saved **555 tokens (5.2%)** with both
decisions validated; `verification/lossless-context-audit-20260929.json`
passes every provenance and routing check with no omitted source terms.

The optional **Decision-focused** protocol (`validated-v5-graph`) preserves the structured
facts needed for the fictional quotation-selection decision. Its compressed
context is constructed exactly from Python's protected fact projection and
verbatim selected supplier caveats. The saved decisions passed all supplied-fact
checks. **It is not a lossless or fully semantically validated compression of
the original supplier documents.**

## Evidence

The reproducible audit in `context_integrity_audit.py` checked two detailed,
one medium and one short saved dashboard runs, plus two experimental
`validated-v6-projection` runs. Every run passed nine checks: unchanged source
and protected record, unchanged Python audit, exact source-backed context
reconstruction, required caveat coverage, exact selected caveats, context
routing, guarded agent handoffs and final-decision validation. The short run
skipped compression and retained the full source.

| Protocol | Scenario | Applied | Net reduction | Decision checks |
| --- | --- | --- | ---: | --- |
| v8 complete-source default | Detailed | Yes | 5.2% | Both passed |
| v5 decision-focused | Detailed, pilot | Yes | 27.4% | Both passed |
| v5 decision-focused | Detailed, held-out | Yes | 27.2% | Both passed |
| v5 decision-focused | Medium | Yes | 15.0% | Both passed |
| v5 decision-focused | Short | No | No compression saving | Both passed |
| v6 experimental | Detailed | Yes | 34.1% | Both passed |
| v6 experimental | Medium | Yes | 22.6% | Both passed |

The machine-readable selective and experimental audit is in
`verification/context-integrity-audit-20260929.json`; the new complete-source
audit is in `verification/lossless-context-audit-20260929.json`.

## What is not established

- The compressed v5 prompt omits carrier names and transport methods. The
  protected source record keeps them, but downstream reviewers do not see them
  in that branch. The v6 experiment includes these fields.
- Both compressed protocols shorten the full brief and supplier terms. All
  catalogued caveats of eligible offers are required, but some sentences for
  ineligible offers are deliberately omitted. The keyword-based caveat
  extractor is not a proof that every important sentence was catalogued.
- The claim guard verifies structured Python-supported comparisons and exact
  source caveats. It does not grade the completeness or writing quality of
  free-form prose. Passing a final decision checklist does not prove that
  every nuance of the original context survived.
- These saved runs use fictional quotation scenarios and Llama 3. They do not
  establish general correctness or a 30% minimum on arbitrary contexts.

## Wording suitable for the rapport

“For the tested structured quotation scenarios, the compressor did not alter
the protected numeric facts. Python reconstructed the downstream context from
those facts and exact selected source caveats, validated agent claims before
handoff, and checked the final decision. All audited paired decisions passed
the supplied-fact checks. The method does not claim lossless preservation of
all narrative details; omitted conditions and free-form prose require broader
semantic evaluation.”
