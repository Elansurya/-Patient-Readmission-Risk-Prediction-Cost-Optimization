import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ── Style 
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── Paths 
CSV_PATH   = r'C:\project\Patient Readmission Risk Prediction + Cost Optimization\data\cleaned_data.csv'
OUTPUT_DIR = r'C:\project\Patient Readmission Risk Prediction + Cost Optimization\outputs\eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load 
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Prepare columns 
age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
             '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
age_map   = {5:'[0-10)',15:'[10-20)',25:'[20-30)',35:'[30-40)',45:'[40-50)',
             55:'[50-60)',65:'[60-70)',75:'[70-80)',85:'[80-90)',95:'[90-100)'}

df['age_label']        = pd.Categorical(df['age'].map(age_map), categories=age_order, ordered=True)
df['readmitted_label'] = df['readmitted'].map({0: 'No Readmission', 1: 'Readmitted'})
df['total_visits']     = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

order_   = ['No Readmission', 'Readmitted']
colors_  = [PALETTE[2], PALETTE[3]]
colors_r = dict(zip(order_, colors_))

num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'total_visits']

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# EDA 1 – Dataset Overview & Demographics

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("EDA – Dataset Overview & Demographics", fontsize=16, fontweight='bold', y=1.01)

# 1-A: Summary stats text box
ax = axes[0, 0]
ax.axis('off')
stats = (
    f"Dataset Shape:  {df.shape[0]:,} rows × {df.shape[1]} cols\n\n"
    f"No missing values\n\n"
    f"Target: readmitted\n"
    f"  0 = No Readmission : {(df['readmitted']==0).sum():,}  ({(df['readmitted']==0).mean()*100:.1f}%)\n"
    f"  1 = Readmitted      : {(df['readmitted']==1).sum():,}  ({(df['readmitted']==1).mean()*100:.1f}%)\n\n"
    f"Numeric features : {df.select_dtypes(include='number').shape[1]}\n"
    f"Categorical feats: {df.select_dtypes(include='object').shape[1]}"
)
ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=12,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#EEF2FF', alpha=0.8))
ax.set_title("Dataset Summary", fontweight='bold')

# 1-B: Age distribution
ax = axes[0, 1]
age_counts = df['age_label'].value_counts().reindex(age_order, fill_value=0)
bars = ax.bar(age_order, age_counts, color=PALETTE[0], edgecolor='white')
ax.set_title("Age Group Distribution", fontweight='bold')
ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=45)
for b in bars:
    h = int(b.get_height())
    if h > 0:
        ax.text(b.get_x()+b.get_width()/2, h+150, f'{h:,}',
                ha='center', va='bottom', fontsize=7)

# 1-C: Gender pie
ax = axes[0, 2]
gc = df['gender'].value_counts()
ax.pie(gc, labels=gc.index, autopct='%1.1f%%',
       colors=[PALETTE[0], PALETTE[1]], startangle=90,
       wedgeprops=dict(edgecolor='white', linewidth=2))
ax.set_title("Gender Distribution", fontweight='bold')

# 1-D: Race distribution
ax = axes[1, 0]
rc = df['race'].value_counts()
ax.barh(rc.index, rc.values, color=PALETTE[2], edgecolor='white')
ax.set_title("Race Distribution", fontweight='bold')
ax.set_xlabel("Count")
for i, v in enumerate(rc.values):
    ax.text(v+100, i, f'{v:,}', va='center', fontsize=8)

# 1-E: Target distribution
ax = axes[1, 1]
rc2 = df['readmitted_label'].value_counts()
bars2 = ax.bar(order_, [rc2.get(o, 0) for o in order_], color=colors_, edgecolor='white')
ax.set_title("Readmission Distribution (Target)", fontweight='bold')
ax.set_ylabel("Count"); ax.tick_params(axis='x', rotation=10)
for b in bars2:
    pct = b.get_height() / len(df) * 100
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+400,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 1-F: Diabetes medication
ax = axes[1, 2]
dm = df['diabetesmed'].value_counts()
ax.bar(dm.index, dm.values, color=[PALETTE[4], PALETTE[1]], edgecolor='white', width=0.4)
ax.set_title("Diabetes Medication Prescribed", fontweight='bold')
ax.set_ylabel("Count")
for i, (k, v) in enumerate(dm.items()):
    ax.text(i, v+400, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
save(fig, 'eda1_overview_demographics.png')

# EDA 2 – Numeric Feature Distributions

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle("EDA – Numeric Feature Distributions", fontsize=16, fontweight='bold', y=1.01)

for i, col in enumerate(num_cols):
    ax = axes[i//3, i%3]
    data = df[col].clip(upper=df[col].quantile(0.99))
    ax.hist(data, bins=30, color=PALETTE[i%5], edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.5,
               label=f'Mean={df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='orange', linestyle='-',  linewidth=1.5,
               label=f'Median={df[col].median():.1f}')
    ax.legend(fontsize=7)

plt.tight_layout()
save(fig, 'eda2_numeric_distributions.png')

# EDA 3 – Numeric Features vs Readmission

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle("EDA – Numeric Features vs Readmission", fontsize=16, fontweight='bold', y=1.01)

for i, col in enumerate(num_cols):
    ax = axes[i//3, i%3]
    sns.boxplot(data=df, x='readmitted_label', y=col,
                order=order_, palette=colors_r, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel(""); ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=10)
    for j, grp in enumerate(order_):
        med = df[df['readmitted_label'] == grp][col].median()
        ax.text(j, med, f'{med:.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
save(fig, 'eda3_numeric_vs_readmission.png')

# EDA 4 – Correlation Heatmap

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("EDA – Correlations", fontsize=16, fontweight='bold')

corr_cols = num_cols + ['readmitted']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=axes[0],
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
axes[0].set_title("Correlation Heatmap (Numeric Features)", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=9)
axes[0].tick_params(axis='y', rotation=0,  labelsize=9)

ax = axes[1]
target_corr = corr['readmitted'].drop('readmitted').sort_values()
colors_bar  = [PALETTE[3] if v > 0 else PALETTE[0] for v in target_corr]
ax.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Feature Correlation with Readmission", fontweight='bold')
ax.set_xlabel("Pearson Correlation Coefficient")
for i, v in enumerate(target_corr.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), i, f'{v:.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=9)

plt.tight_layout()
save(fig, 'eda4_correlations.png')



# EDA 5 – Categorical Features vs Readmission Rate

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("EDA – Categorical Features vs Readmission Rate",
             fontsize=16, fontweight='bold', y=1.01)

cat_features = [
    ('age_label',   'Age Group',         age_order),
    ('gender',      'Gender',            None),
    ('race',        'Race',              None),
    ('change',      'Medication Change', None),
    ('diabetesmed', 'Diabetes Med',      None),
    ('insulin',     'Insulin Usage',     None),
]

for i, (col, title, order_cat) in enumerate(cat_features):
    ax = axes[i//3, i%3]
    rate = df.groupby(col, observed=True)['readmitted'].mean() * 100
    if order_cat:
        rate = rate.reindex(order_cat).fillna(0)
    else:
        rate = rate.sort_values(ascending=False)
    bars = ax.bar(range(len(rate)), rate.values,
                  color=sns.color_palette("Blues_d", len(rate)), edgecolor='white')
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(rate.index, rotation=35, ha='right', fontsize=8)
    ax.set_title(f"Readmission Rate by {title}", fontweight='bold')
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_ylim(0, rate.max() * 1.25 if rate.max() > 0 else 1)
    for b, v in zip(bars, rate.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
save(fig, 'eda5_categorical_vs_readmission.png')


# EDA 6 – Medication Analysis

med_cols = ['metformin', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'insulin', 'glimepiride', 'repaglinide']
med_cols = [m for m in med_cols if m in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("EDA – Medication Analysis", fontsize=16, fontweight='bold', y=1.01)

# 6-A: % patients on each medication
ax = axes[0, 0]
usage_s = pd.Series({m: (df[m] != 'No').mean()*100 for m in med_cols}).sort_values(ascending=False)
ax.bar(usage_s.index, usage_s.values, color=PALETTE[0], edgecolor='white')
ax.set_title("% Patients Prescribed Each Medication", fontweight='bold')
ax.set_ylabel("% Patients"); ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(usage_s.values):
    ax.text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')

# 6-B: Readmission rate with vs without medication
ax = axes[0, 1]
results = [{'Med': m,
            'Without': df[df[m]=='No']['readmitted'].mean()*100,
            'With':    df[df[m]!='No']['readmitted'].mean()*100}
           for m in med_cols]
med_df = pd.DataFrame(results).set_index('Med')
x = np.arange(len(med_df)); w = 0.35
ax.bar(x-w/2, med_df['Without'], width=w, label='Without', color=PALETTE[2], edgecolor='white')
ax.bar(x+w/2, med_df['With'],    width=w, label='With',    color=PALETTE[3], edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(med_df.index, rotation=30, ha='right', fontsize=8)
ax.set_title("Readmission Rate: With vs Without Medication", fontweight='bold')
ax.set_ylabel("Readmission Rate (%)"); ax.set_ylim(0, 60); ax.legend()

# 6-C: Insulin × readmission stacked bar
ax = axes[1, 0]
ins_pivot = (df.groupby(['insulin', 'readmitted_label'], observed=True)
               .size().unstack(fill_value=0))
for col in order_:
    if col not in ins_pivot.columns:
        ins_pivot[col] = 0
ins_pct = ins_pivot[order_].div(ins_pivot[order_].sum(axis=1), axis=0) * 100
ins_pct.plot(kind='bar', stacked=True, ax=ax, color=colors_, edgecolor='white', width=0.6)
ax.set_title("Insulin Usage × Readmission Breakdown", fontweight='bold')
ax.set_xlabel("Insulin Category"); ax.set_ylabel("Percentage (%)")
ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=9)

# 6-D: Medication change effect
ax = axes[1, 1]
change_rate = df.groupby('change', observed=True)['readmitted'].mean() * 100
ax.bar(change_rate.index, change_rate.values,
       color=[PALETTE[2], PALETTE[3]], edgecolor='white', width=0.4)
ax.set_title("Readmission Rate by Medication Change", fontweight='bold')
ax.set_ylabel("Readmission Rate (%)"); ax.set_ylim(0, 50)
for i, (k, v) in enumerate(change_rate.items()):
    ax.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
save(fig, 'eda6_medications.png')


# EDA 7 – Hospital Stay & Visit Patterns

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("EDA – Hospital Stay & Visit Patterns", fontsize=16, fontweight='bold', y=1.01)

# 7-A: Time in hospital density
ax = axes[0, 0]
for label, col in colors_r.items():
    subset = df[df['readmitted_label'] == label]['time_in_hospital']
    ax.hist(subset, bins=14, alpha=0.65, label=label, color=col, density=True, edgecolor='white')
ax.set_title("Time in Hospital Density by Readmission", fontweight='bold')
ax.set_xlabel("Days"); ax.set_ylabel("Density"); ax.legend(fontsize=9)

# 7-B: Avg hospital stay by age
ax = axes[0, 1]
age_stay = df.groupby('age_label', observed=True)['time_in_hospital'].mean().reindex(age_order).fillna(0)
ax.bar(age_order, age_stay, color=PALETTE[4], edgecolor='white')
ax.set_title("Avg Hospital Stay by Age Group", fontweight='bold')
ax.set_xlabel("Age Group"); ax.set_ylabel("Avg Days")
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(age_stay):
    ax.text(i, v+0.05, f'{v:.1f}', ha='center', fontsize=7.5)

# 7-C: Inpatient visits vs readmission rate
ax = axes[0, 2]
inpat_rate = (df[df['number_inpatient'] <= 6]
              .groupby('number_inpatient', observed=True)['readmitted'].mean() * 100)
ax.plot(inpat_rate.index, inpat_rate.values, marker='o',
        color=PALETTE[3], linewidth=2.5, markersize=8)
ax.fill_between(inpat_rate.index, inpat_rate.values, alpha=0.15, color=PALETTE[3])
ax.set_title("Readmission Rate by Prior Inpatient Visits", fontweight='bold')
ax.set_xlabel("# Prior Inpatient Visits"); ax.set_ylabel("Readmission Rate (%)")

# 7-D: Total visits histogram
ax = axes[1, 0]
ax.hist(df['total_visits'].clip(upper=15), bins=16, color=PALETTE[0], edgecolor='white')
ax.set_title("Total Prior Visits (capped 15)", fontweight='bold')
ax.set_xlabel("Total Visits"); ax.set_ylabel("Count")

# 7-E: Emergency visits vs readmission
ax = axes[1, 1]
emerg_rate = (df[df['number_emergency'] <= 5]
              .groupby('number_emergency', observed=True)['readmitted'].mean() * 100)
ax.plot(emerg_rate.index, emerg_rate.values, marker='s',
        color=PALETTE[2], linewidth=2.5, markersize=8)
ax.fill_between(emerg_rate.index, emerg_rate.values, alpha=0.15, color=PALETTE[2])
ax.set_title("Readmission Rate by Prior Emergency Visits", fontweight='bold')
ax.set_xlabel("# Prior Emergency Visits"); ax.set_ylabel("Readmission Rate (%)")

# 7-F: Num medications violin
ax = axes[1, 2]
sns.violinplot(data=df, x='readmitted_label', y='num_medications',
               order=order_, palette=colors_r, ax=ax, inner='quartile')
ax.set_title("Num Medications by Readmission", fontweight='bold')
ax.set_xlabel(""); ax.set_ylabel("Number of Medications")
ax.tick_params(axis='x', rotation=10)

plt.tight_layout()
save(fig, 'eda7_hospital_visits.png')

# EDA 8 – Summary Statistics Table + Age×Gender Heatmap

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("EDA – Summary Statistics", fontsize=16, fontweight='bold')

# 8-A: Descriptive stats table
ax = axes[0]
ax.axis('off')
desc = df[num_cols].describe().round(2).T
desc.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
desc['count'] = desc['count'].astype(int)
table = ax.table(
    cellText=desc.values,
    rowLabels=desc.index,
    colLabels=desc.columns,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False); table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EEF2FF')
ax.set_title("Descriptive Statistics – Numeric Features", fontweight='bold', pad=20)

# 8-B: Readmission rate age × gender heatmap
ax = axes[1]
pivot_hm = (df.groupby(['age_label', 'gender'], observed=True)['readmitted']
              .mean().unstack() * 100).reindex(age_order).fillna(0)
sns.heatmap(pivot_hm, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={"label": "Readmit %"}, annot_kws={"size": 10})
ax.set_title("Readmission Rate (%) by Age × Gender", fontweight='bold')
ax.set_xlabel("Gender"); ax.set_ylabel("Age Group")

plt.tight_layout()
save(fig, 'eda8_summary_stats.png')

print(f"\n✅ All 8 EDA figures saved to: {OUTPUT_DIR}")