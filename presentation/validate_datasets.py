"""
Cross-Dataset Validation Script
CS 4774 Final Project

Compares our primary dataset (annual, per-company) against the
event-level layoffs dataset (layoffs.csv) to validate data quality
and surface any major discrepancies.

Outputs:
  - validation_comparison.csv   : side-by-side comparison by company/year
  - validation_summary.csv      : per-company agreement statistics
"""

import pandas as pd
import numpy as np

# ── Load datasets ──────────────────────────────────────────────────────────────
primary = pd.read_csv("tech_employment_2000_2025.csv")
primary["layoff_pct"] = primary["layoffs"] / primary["employees_start"] * 100

events = pd.read_csv("layoffs.csv")
events["date"] = pd.to_datetime(events["date"], errors="coerce")
events["year"] = events["date"].dt.year

# ── Our 25 companies ───────────────────────────────────────────────────────────
our_companies = primary["company"].unique().tolist()

# ── Filter event dataset to only our companies ────────────────────────────────
events_matched = events[events["company"].isin(our_companies)].copy()

# ── Aggregate event-level data to annual totals per company ───────────────────
# Sum all documented layoff events within each company+year
# Use total_laid_off where available; fill missing with 0 for aggregation
events_matched["total_laid_off"] = events_matched["total_laid_off"].fillna(0)

annual_events = (
    events_matched
    .groupby(["company", "year"])
    .agg(
        event_count=("total_laid_off", "count"),          # number of layoff events
        total_laid_off_sum=("total_laid_off", "sum"),     # total headcount laid off
    )
    .reset_index()
)

# ── Merge with primary dataset ────────────────────────────────────────────────
merged = primary.merge(annual_events, on=["company", "year"], how="inner")

# Compute event-derived layoff % for comparison
merged["event_layoff_pct"] = merged["total_laid_off_sum"] / merged["employees_start"] * 100

# Absolute difference between the two sources
merged["pct_difference"] = abs(merged["layoff_pct"] - merged["event_layoff_pct"])

# Agreement flag: within 5 percentage points = "agree"
merged["sources_agree"] = merged["pct_difference"] <= 5.0

# ── Print results ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  CROSS-DATASET VALIDATION: Primary vs. Event-Level Data")
print("=" * 70)

comparison_cols = [
    "company", "year",
    "layoffs",           # raw count from primary dataset
    "layoff_pct",        # % from primary dataset
    "event_count",       # number of layoff events in second dataset
    "total_laid_off_sum",# sum of headcount from events
    "event_layoff_pct",  # % derived from events
    "pct_difference",
    "sources_agree"
]

print(f"\nOverlapping company-years found: {len(merged)}")
print(f"Sources agree (within 5%): {merged['sources_agree'].sum()} / {len(merged)}")
print(f"Agreement rate: {merged['sources_agree'].mean()*100:.1f}%\n")

# Show biggest discrepancies
print("── Largest discrepancies (top 10) ──────────────────────────────────")
top_disc = merged.sort_values("pct_difference", ascending=False).head(10)
print(top_disc[comparison_cols].to_string(index=False))

# Show cases where sources agree well
print("\n── Best agreements (pct_difference < 2%) ────────────────────────────")
good = merged[merged["pct_difference"] < 2].sort_values("pct_difference")
print(good[comparison_cols].to_string(index=False))

# ── Per-company summary ───────────────────────────────────────────────────────
print("\n── Per-company summary ──────────────────────────────────────────────")
summary = (
    merged.groupby("company")
    .agg(
        years_compared=("year", "count"),
        avg_primary_layoff_pct=("layoff_pct", "mean"),
        avg_event_layoff_pct=("event_layoff_pct", "mean"),
        avg_difference=("pct_difference", "mean"),
        pct_agreeing=("sources_agree", "mean"),
    )
    .reset_index()
)
summary["avg_primary_layoff_pct"] = summary["avg_primary_layoff_pct"].round(2)
summary["avg_event_layoff_pct"]   = summary["avg_event_layoff_pct"].round(2)
summary["avg_difference"]         = summary["avg_difference"].round(2)
summary["pct_agreeing"]           = (summary["pct_agreeing"] * 100).round(1)
summary = summary.sort_values("avg_difference")

print(summary.to_string(index=False))

# ── Notable real-world events check ───────────────────────────────────────────
print("\n── Notable layoff events captured in event dataset ─────────────────")
notable = events_matched[
    (events_matched["total_laid_off"] >= 5000) &
    (events_matched["company"].isin(our_companies))
][["company", "date", "total_laid_off", "percentage_laid_off"]].sort_values("total_laid_off", ascending=False)
print(notable.head(20).to_string(index=False))

# ── Save outputs ───────────────────────────────────────────────────────────────
merged[comparison_cols].to_csv("validation_comparison.csv", index=False)
summary.to_csv("validation_summary.csv", index=False)
print("\n✅ Saved: validation_comparison.csv, validation_summary.csv")