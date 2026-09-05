# End-to-End Option-Chain Analysis Pipeline

`src/report/pipeline.py` chains stages 1-8 (plus a lightweight liquidity
assessment) on one graded quote snapshot:

1. input frame check (quality/mid guards, empty protection);
2. implied forwards (parity regression);
3. five no-arbitrage checks;
4. Black-76 implied volatility (Newton + Brent, OTM-unified mids);
5. per-expiry skew (RR25/BF25);
6. ATM term structure + forward vol + calendar screen;
7. robust SVI calibration + Gatheral `g(k)` + cross-expiry calendar screen;
8. chain Greeks (r=4%, q=1.2% convention);
9. liquidity state (median relative spread, reliability, quality share);
10. structured report + figures.

Data-flow discipline: the full graded snapshot feeds cleaning/liquidity;
the quality-good implied-vol frame feeds skew/Greeks. Every stage is wrapped
so one failure degrades to a warning while later stages continue where
possible; all intermediate frames are saved next to `report.json`.

## Run

```bash
python scripts/run_analysis.py \
  --input outputs/real_option_chain/spy_quality_graded.csv \
  --output outputs/analysis_run
```

Real snapshot result (2026-09-04): spot 773.17, term structure `humped`
(front contango / back backwardation), SVI valid 4/6 with IV-unit mean RMSE
1.27 vol pts, cross-expiry calendar crossings 64, liquidity `moderate`,
zero pipeline warnings.
