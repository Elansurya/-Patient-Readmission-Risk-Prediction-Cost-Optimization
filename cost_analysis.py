import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths 
BASE_DIR  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

SCORED_PATH  = os.path.join(DATA_DIR, "diabetic_scored.csv")
METRICS_PATH = os.path.join(OUT_DIR,  "model_metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("STEP 11 — COST OPTIMIZATION ANALYSIS")
print("=" * 65)

# ──  Guard 
if not os.path.exists(SCORED_PATH):
    raise FileNotFoundError(
        f"Scored dataset not found: {SCORED_PATH}\n"
        "Run Step 10 first."
    )

# ── Load scored dataset 
df = pd.read_csv(SCORED_PATH, low_memory=False)
print(f"\n  Scored dataset : {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Columns        : {df.columns.tolist()}")

# ── Validate required columns 
required_cols = ["readmitted_30d", "risk_score", "risk_tier"]
missing_cols  = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(
        f"Missing columns in scored dataset: {missing_cols}\n"
        "Re-run Step 10."
    )

# ── Cost Assumptions (literature-based, USD) 
# Source: Agency for Healthcare Research and Quality (AHRQ)
# Average diabetic readmission cost: ~$15,000-$17,500
# Intervention cost (care coordinator, follow-up): ~$1,200

COST_READMISSION   = 15000   # avg cost of one 30-day readmission ($)
COST_INTERVENTION  = 1200    # avg cost of care management per patient ($)
INTERVENTION_EFFICACY = 0.30 # 30% reduction in readmission for intervened patients

print(f"\n  Cost Assumptions:")
print(f"    Avg readmission cost     : ${COST_READMISSION:,}")
print(f"    Intervention cost        : ${COST_INTERVENTION:,}")
print(f"    Intervention efficacy    : {INTERVENTION_EFFICACY*100:.0f}% readmission reduction")

# ── Overall statistics 
total_patients  = len(df)
total_positives = int(df["readmitted_30d"].sum())
baseline_rate   = total_positives / total_patients

print(f"\n  Patient Statistics:")
print(f"    Total patients           : {total_patients:,}")
print(f"    Actual readmissions      : {total_positives:,}  ({baseline_rate*100:.1f}%)")
print(f"    Total readmission cost   : ${total_positives * COST_READMISSION:,.0f}")

# ANALYSIS 1 — Cost by Risk Tier

print(f"\n{'─'*65}")
print("ANALYSIS 1 — COST BREAKDOWN BY RISK TIER")
print(f"{'─'*65}")

tier_results = []
for tier in ["High", "Medium", "Low"]:
    subset        = df[df["risk_tier"] == tier]
    n_patients    = len(subset)
    n_actual_pos  = int(subset["readmitted_30d"].sum())
    readmit_rate  = n_actual_pos / n_patients if n_patients > 0 else 0

    # Cost WITHOUT intervention
    cost_no_intervention = n_actual_pos * COST_READMISSION

    # Cost WITH intervention (applied to all in tier)
    cost_intervention    = n_patients * COST_INTERVENTION
    readmissions_avoided = n_actual_pos * INTERVENTION_EFFICACY
    cost_savings         = readmissions_avoided * COST_READMISSION
    net_savings          = cost_savings - cost_intervention
    roi                  = (net_savings / cost_intervention * 100
                            if cost_intervention > 0 else 0)

    tier_results.append({
        "Tier"               : tier,
        "Patients"           : n_patients,
        "Actual Readmissions": n_actual_pos,
        "Readmit Rate"       : readmit_rate,
        "Cost No Intervention": cost_no_intervention,
        "Intervention Cost"  : cost_intervention,
        "Savings"            : cost_savings,
        "Net Savings"        : net_savings,
        "ROI %"              : roi,
        "Avoided Readmissions": readmissions_avoided,
    })

    print(f"\n  [{tier} Risk Tier]")
    print(f"    Patients             : {n_patients:,}")
    print(f"    Actual readmissions  : {n_actual_pos:,}  ({readmit_rate*100:.1f}%)")
    print(f"    Cost w/o intervention: ${cost_no_intervention:>12,.0f}")
    print(f"    Intervention cost    : ${cost_intervention:>12,.0f}")
    print(f"    Readmissions avoided : {readmissions_avoided:>12,.1f}")
    print(f"    Gross savings        : ${cost_savings:>12,.0f}")
    print(f"    Net savings          : ${net_savings:>12,.0f}")
    print(f"    ROI                  : {roi:>11.1f}%")

tier_df = pd.DataFrame(tier_results)


# ANALYSIS 2 — Threshold Sweep 

print(f"\n{'─'*65}")
print("ANALYSIS 2 — OPTIMAL INTERVENTION THRESHOLD")
print(f"{'─'*65}")

thresholds   = np.arange(0.05, 0.95, 0.01)
threshold_results = []

for thresh in thresholds:
    intervene_mask   = df["risk_score"] >= thresh
    n_intervene      = int(intervene_mask.sum())
    n_true_pos       = int((intervene_mask & (df["readmitted_30d"] == 1)).sum())
    n_true_neg       = int((~intervene_mask & (df["readmitted_30d"] == 0)).sum())
    n_false_neg      = int((~intervene_mask & (df["readmitted_30d"] == 1)).sum())

    # Costs
    intervention_cost   = n_intervene * COST_INTERVENTION
    readmissions_avoided = n_true_pos * INTERVENTION_EFFICACY
    gross_savings        = readmissions_avoided * COST_READMISSION
    missed_cost          = n_false_neg * COST_READMISSION
    net_savings          = gross_savings - intervention_cost

    precision = n_true_pos / n_intervene if n_intervene > 0 else 0
    recall    = n_true_pos / total_positives if total_positives > 0 else 0

    threshold_results.append({
        "threshold"          : round(float(thresh), 2),
        "n_intervene"        : n_intervene,
        "n_true_pos"         : n_true_pos,
        "precision"          : precision,
        "recall"             : recall,
        "intervention_cost"  : intervention_cost,
        "gross_savings"      : gross_savings,
        "net_savings"        : net_savings,
        "missed_cost"        : missed_cost,
    })

thresh_df = pd.DataFrame(threshold_results)

# Find optimal threshold (max net savings)
best_idx    = thresh_df["net_savings"].idxmax()
best_thresh = thresh_df.loc[best_idx]

print(f"\n  Optimal threshold (max net savings):")
print(f"    Threshold            : {best_thresh['threshold']:.2f}")
print(f"    Patients intervened  : {best_thresh['n_intervene']:,.0f}")
print(f"    True positives caught: {best_thresh['n_true_pos']:,.0f}")
print(f"    Precision            : {best_thresh['precision']:.3f}")
print(f"    Recall               : {best_thresh['recall']:.3f}")
print(f"    Net savings          : ${best_thresh['net_savings']:,.0f}")

# ANALYSIS 3 — Strategy Comparison

print(f"\n{'─'*65}")
print("ANALYSIS 3 — INTERVENTION STRATEGY COMPARISON")
print(f"{'─'*65}")

strategies = {
    "No Intervention"        : {"intervene_all": False, "tier": None,   "thresh": None},
    "Intervene All Patients" : {"intervene_all": True,  "tier": None,   "thresh": None},
    "High Risk Tier Only"    : {"intervene_all": False, "tier": "High", "thresh": None},
    "High + Medium Tier"     : {"intervene_all": False, "tier": "High+Medium", "thresh": None},
    "Optimal Threshold"      : {"intervene_all": False, "tier": None,   "thresh": float(best_thresh["threshold"])},
}

strategy_results = []
for strat_name, config in strategies.items():
    if config["intervene_all"]:
        mask = pd.Series([True] * len(df))
    elif config["tier"] == "High":
        mask = df["risk_tier"] == "High"
    elif config["tier"] == "High+Medium":
        mask = df["risk_tier"].isin(["High", "Medium"])
    elif config["thresh"] is not None:
        mask = df["risk_score"] >= config["thresh"]
    else:
        mask = pd.Series([False] * len(df))

    n_intervene          = int(mask.sum())
    n_true_pos           = int((mask & (df["readmitted_30d"] == 1)).sum())
    n_false_neg          = int((~mask & (df["readmitted_30d"] == 1)).sum())
    intervention_cost    = n_intervene * COST_INTERVENTION
    readmissions_avoided = n_true_pos * INTERVENTION_EFFICACY
    gross_savings        = readmissions_avoided * COST_READMISSION
    net_savings          = gross_savings - intervention_cost
    total_readmit_cost   = (n_false_neg + n_true_pos * (1 - INTERVENTION_EFFICACY)) * COST_READMISSION
    total_program_cost   = intervention_cost + total_readmit_cost
    recall               = n_true_pos / total_positives if total_positives > 0 else 0

    strategy_results.append({
        "Strategy"             : strat_name,
        "Intervened"           : n_intervene,
        "True Pos Caught"      : n_true_pos,
        "Recall"               : recall,
        "Intervention Cost ($)": intervention_cost,
        "Gross Savings ($)"    : gross_savings,
        "Net Savings ($)"      : net_savings,
        "Total Program Cost ($)": total_program_cost,
    })

    print(f"\n  [{strat_name}]")
    print(f"    Patients intervened  : {n_intervene:,}")
    print(f"    Readmissions caught  : {n_true_pos:,}  (recall={recall:.1%})")
    print(f"    Intervention cost    : ${intervention_cost:>12,.0f}")
    print(f"    Gross savings        : ${gross_savings:>12,.0f}")
    print(f"    Net savings          : ${net_savings:>12,.0f}")
    print(f"    Total program cost   : ${total_program_cost:>12,.0f}")

strategy_df = pd.DataFrame(strategy_results)

# FIGURE A — Net Savings by Risk Tier

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig 11a — Cost Analysis by Risk Tier",
             fontsize=13, fontweight="bold")

tier_colors = {"High": "#F44336", "Medium": "#FFC107", "Low": "#4CAF50"}
tiers       = tier_df["Tier"].tolist()
colors      = [tier_colors[t] for t in tiers]

# Net savings
bars = axes[0].bar(tiers,
                   tier_df["Net Savings"] / 1e6,
                   color=colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, tier_df["Net Savings"]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"${val/1e6:.2f}M",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_title("Net Savings by Tier")
axes[0].set_ylabel("Net Savings ($ Millions)")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].grid(axis="y", alpha=0.3)

# ROI %
bars2 = axes[1].bar(tiers,
                    tier_df["ROI %"],
                    color=colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars2, tier_df["ROI %"]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_title("Return on Investment by Tier")
axes[1].set_ylabel("ROI (%)")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].grid(axis="y", alpha=0.3)

# Readmissions avoided
bars3 = axes[2].bar(tiers,
                    tier_df["Avoided Readmissions"],
                    color=colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars3, tier_df["Avoided Readmissions"]):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 f"{val:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[2].set_title("Readmissions Avoided by Tier")
axes[2].set_ylabel("Readmissions Avoided")
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
path_a = os.path.join(OUT_DIR, "fig11a_cost_by_tier.png")
plt.savefig(path_a, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✔  Saved: fig11a_cost_by_tier.png")

# FIGURE B — Threshold Sweep: Net Savings vs Threshold

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 11b — Optimal Intervention Threshold Analysis",
             fontsize=13, fontweight="bold")

axes[0].plot(thresh_df["threshold"],
             thresh_df["net_savings"] / 1e6,
             color="#2196F3", linewidth=2)
axes[0].axvline(best_thresh["threshold"],
                color="red", linestyle="--", linewidth=1.5,
                label=f"Optimal ({best_thresh['threshold']:.2f})")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Risk Score Threshold")
axes[0].set_ylabel("Net Savings ($ Millions)")
axes[0].set_title("Net Savings vs Threshold")
axes[0].legend()
axes[0].grid(alpha=0.3)

ax2 = axes[0].twinx()
ax2.plot(thresh_df["threshold"],
         thresh_df["n_intervene"] / 1000,
         color="#FF9800", linewidth=1.5,
         linestyle="--", alpha=0.7,
         label="Patients intervened (K)")
ax2.set_ylabel("Patients Intervened (thousands)", color="#FF9800")
ax2.tick_params(axis="y", labelcolor="#FF9800")

axes[1].plot(thresh_df["threshold"],
             thresh_df["precision"],
             color="#4CAF50", linewidth=2, label="Precision")
axes[1].plot(thresh_df["threshold"],
             thresh_df["recall"],
             color="#F44336", linewidth=2, label="Recall")
axes[1].axvline(best_thresh["threshold"],
                color="blue", linestyle="--", linewidth=1.5,
                label=f"Optimal ({best_thresh['threshold']:.2f})")
axes[1].set_xlabel("Risk Score Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Precision & Recall vs Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
path_b = os.path.join(OUT_DIR, "fig11b_threshold_analysis.png")
plt.savefig(path_b, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig11b_threshold_analysis.png")

# FIGURE C — Strategy Comparison

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Fig 11c — Intervention Strategy Comparison",
             fontsize=13, fontweight="bold")

strat_names  = [s.replace(" ", "\n") for s in strategy_df["Strategy"]]
net_savings  = strategy_df["Net Savings ($)"].values / 1e6
prog_costs   = strategy_df["Total Program Cost ($)"].values / 1e6
bar_colors   = ["#9E9E9E", "#F44336", "#FF9800", "#FFC107", "#4CAF50"]

bars = axes[0].barh(strat_names, net_savings,
                    color=bar_colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, net_savings):
    xpos = val + 0.05 if val >= 0 else val - 0.05
    axes[0].text(xpos, bar.get_y() + bar.get_height()/2,
                 f"${val:.2f}M",
                 va="center", ha="left" if val >= 0 else "right",
                 fontsize=8, fontweight="bold")
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Net Savings ($ Millions)")
axes[0].set_title("Net Savings by Strategy")
axes[0].grid(axis="x", alpha=0.3)

bars2 = axes[1].barh(strat_names, prog_costs,
                     color=bar_colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars2, prog_costs):
    axes[1].text(val + 0.05,
                 bar.get_y() + bar.get_height()/2,
                 f"${val:.2f}M",
                 va="center", ha="left",
                 fontsize=8, fontweight="bold")
axes[1].set_xlabel("Total Program Cost ($ Millions)")
axes[1].set_title("Total Program Cost by Strategy")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
path_c = os.path.join(OUT_DIR, "fig11c_strategy_comparison.png")
plt.savefig(path_c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig11c_strategy_comparison.png")


# FIGURE D — Cost Waterfall: Best Strategy

best_strategy = strategy_df.loc[strategy_df["Net Savings ($)"].idxmax()]

waterfall_labels = [
    "Baseline\nReadmission Cost",
    "Intervention\nCost",
    "Gross\nSavings",
    "Net\nSavings"
]
baseline_cost   = total_positives * COST_READMISSION
intervention_c  = float(best_strategy["Intervention Cost ($)"])
gross_sav       = float(best_strategy["Gross Savings ($)"])
net_sav         = float(best_strategy["Net Savings ($)"])

waterfall_vals  = [baseline_cost, -intervention_c, gross_sav, net_sav]
waterfall_colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(waterfall_labels,
              [v / 1e6 for v in waterfall_vals],
              color=waterfall_colors, edgecolor="white",
              alpha=0.85, width=0.5)
for bar, val in zip(bars, waterfall_vals):
    ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
    ax.text(bar.get_x() + bar.get_width()/2,
            ypos, f"${val/1e6:.2f}M",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Amount ($ Millions)")
ax.set_title(f"Fig 11d — Cost Waterfall: '{best_strategy['Strategy']}' Strategy",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path_d = os.path.join(OUT_DIR, "fig11d_cost_waterfall.png")
plt.savefig(path_d, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig11d_cost_waterfall.png")

# ── Save results to CSV 
tier_df.to_csv(os.path.join(OUT_DIR, "cost_by_tier.csv"), index=False)
strategy_df.to_csv(os.path.join(OUT_DIR, "cost_by_strategy.csv"), index=False)
thresh_df.to_csv(os.path.join(OUT_DIR, "threshold_sweep.csv"), index=False)

print(f"\n  ✔  Saved: cost_by_tier.csv")
print(f"  ✔  Saved: cost_by_strategy.csv")
print(f"  ✔  Saved: threshold_sweep.csv")

# ── Final recommendation 
print(f"""
{'='*65}
COST OPTIMIZATION SUMMARY
{'='*65}
  Total patients          : {total_patients:,}
  Total readmissions      : {total_positives:,}  ({baseline_rate*100:.1f}%)
  Baseline readmit cost   : ${total_positives * COST_READMISSION:,.0f}

  RECOMMENDED STRATEGY    : {best_strategy['Strategy']}
  Patients to intervene   : {int(best_strategy['Intervened']):,}
  Recall (readmit caught) : {best_strategy['Recall']:.1%}
  Intervention cost       : ${best_strategy['Intervention Cost ($)']:,.0f}
  Gross savings           : ${best_strategy['Gross Savings ($)']:,.0f}
  Net savings             : ${best_strategy['Net Savings ($)']:,.0f}
  Total program cost      : ${best_strategy['Total Program Cost ($)']:,.0f}

  Optimal risk threshold  : {best_thresh['threshold']:.2f}
{'='*65}
""")