# Live verification — 2026-09-26

Model: llama3. Pipeline: paired-v7. One paired run per scenario, seed 42, temperature 0. All 58 automated tests passed. No prompts or model settings changed during this rerun.

| Scenario | Compression applied | Estimated net saving if compressed | Baseline tokens | Auto tokens | Measured difference | Baseline checks | Auto checks |
|---|---|---:|---:|---:|---:|---:|---:|
| demo_detailed | True | 830 | 8930 | 8473 | 457 (5.12%) | 10/12 | 12/12 |
| demo_medium | False | 12 | 5486 | 5328 | 158 (2.88%) | 11/12 | 12/12 |
| demo_short | False | -238 | 4000 | 3976 | 24 (0.60%) | 12/12 | 12/12 |

## Interpretation

- Detailed: predicted 830 net tokens saved; measured 457. Downstream input saving 2,179 + output saving 78 − compressor cost 1,800 = 457. Compression reduced Risk input by 66.01%, but whole-workflow tokens by only 5.12%. Baseline chose late D1; auto chose correct D2. Branch duration increased from 76.52s to 79.69s. Baseline Risk and compressor outputs reached generation limits.
- Medium: compression skipped (estimate 12, below 100-token threshold). The 158-token difference is generation variation, not compressor savings. Baseline chose M1 ($5,900), although eligible M3 costs $5,100. Auto chose M3.
- Short: compression skipped (estimate −238). Both decisions chose S1 correctly. The 24-token difference is generation variation, not compressor savings.

## Narrative errors observed

- Detailed Finance and compressed notes falsely state D2 at $12,800 exceeds the $15,000 budget; they recommend late D1 as feasible.
- Detailed compressed Risk falsely states 600 units is less than 500, all quotes include insurance (D4 does not), and recommends D1 despite its late arrival.
- Medium baseline Compliance calls M1 the lowest-cost feasible offer despite M3 being cheaper. Auto Risk omits its Impact and Actions sections and the late-arrival risk.
- Short Risk, in both branches, falsely states S2 ($540) is above the $900 budget.

Protected structured facts remained intact. Passing the final 12 checks does not establish correctness of all report prose. The short case has quality_preserving_savings=true in raw results because that field checks final decisions and net difference only; compression was skipped, so it must not be read as evidence of compressor effectiveness.

## Conclusion

There is measured token reduction on the one compressed case, smaller than predicted. The automatic branch has correct final quote selections in this sample, but generated reports are not fully reliable. This single run does not establish repeatable quality preservation. Recommended next work: strengthen report-level factual checks and prevent failed numeric/eligibility claims from reaching downstream agents, then repeat balanced-order runs before claiming a general saving.
