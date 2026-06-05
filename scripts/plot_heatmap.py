import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

eval_dir = Path('plots/eval_overnight')
groups = ['ca_only', 'backbone', 'all_atom']

# We will collect a list of dicts for our dataframe
records = []
metrics_metadata = {
    # Metric Name: (HigherIsBetter, min_val, max_val_for_norm)
    'Bond Valid 0.2': (True, 0.0, 1.0),
    'Bond RMSE': (False, 0.0, 0.5), # 0 is best, >0.5 is worst
    'Angle RMSE': (False, 0.0, 0.5),
    'Clash Rate': (False, 0.0, 1.0),
    'MMD': (False, 0.0, 0.1),
    'Rg Diff': (False, 0.0, 1.0),
    'Novelty Valid': (True, 0.0, 1.0),
    'Coverage': (True, 0.0, 1.0),
    'Precision': (True, 0.0, 1.0),
    'NND Ratio': (None, 0.9, 1.1),
    'Test1 Eq Err': (False, -6.0, 0.0), # log10(error)
    'Isotropy': (False, 1.0, 3.0),
    'Rosetta Pass': (True, 0.0, 1.0),
    'Rosetta fa_rep': (False, 0.0, 100.0),
}

for group in groups:
    p = eval_dir / group / 'metrics.json'
    if not p.exists():
        continue
    with open(p) as f:
        data = json.load(f)
    
    per_model = data.get('per_model', {})
    sections = data.get('sections', {})
    phys_ref = sections.get('physics', {}).get('ref_metrics', {})
    ref_rg = phys_ref.get('rg_mean', 5.92)
    
    for lbl, m_data in per_model.items():
        rec = {'Model': f"{group}/{lbl}"}
        cm = m_data.get('ca_metrics', {})
        
        # Physics
        rec['Bond Valid 0.2'] = cm.get('bond_valid_02', np.nan)
        rec['Bond RMSE'] = cm.get('bond_rmse', np.nan)
        rec['Angle RMSE'] = cm.get('angle_rmse_cos', np.nan)
        rec['Clash Rate'] = cm.get('clash_rate', np.nan)
        rec['MMD'] = cm.get('mmd', np.nan)
        if 'rg_mean' in cm:
            rec['Rg Diff'] = abs(cm['rg_mean'] - ref_rg)
        else:
            rec['Rg Diff'] = np.nan
            
        # Equivariance
        eq = sections.get('equivariance', {}).get(lbl, {})
        if not eq.get('skipped', True) and 'test1' in eq:
            try:
                mean_prop = np.mean(eq['test1']['proper'])
                rec['Test1 Eq Err'] = np.log10(mean_prop) if mean_prop > 0 else -6
            except:
                rec['Test1 Eq Err'] = np.nan
        else:
            rec['Test1 Eq Err'] = np.nan
            
        if not eq.get('skipped', True) and 'test3' in eq:
            rec['Isotropy'] = eq['test3'].get('ratio', np.nan)
        else:
            rec['Isotropy'] = np.nan

        # Novelty
        nov = sections.get('novelty', {}).get(lbl, {})
        rec['Novelty Valid'] = nov.get('valid_fraction', np.nan)
        rec['Coverage'] = nov.get('coverage', np.nan)
        rec['Precision'] = nov.get('precision', np.nan)
        rec['NND Ratio'] = nov.get('nnd_ratio', np.nan)
        
        # PyRosetta
        ro = sections.get('pyrosetta', {}).get(lbl, {})
        if not ro.get('skipped', True):
            rec['Rosetta Pass'] = ro.get('pass_fraction', np.nan)
            agg = ro.get('scores_agg', {})
            rec['Rosetta fa_rep'] = agg.get('fa_rep', {}).get('mean', np.nan)
        else:
            rec['Rosetta Pass'] = np.nan
            rec['Rosetta fa_rep'] = np.nan
            
        records.append(rec)

df = pd.DataFrame(records).set_index('Model')

# Prepare normalized dataframe
df_norm = pd.DataFrame(index=df.index, columns=df.columns)
for col in df.columns:
    if col not in metrics_metadata:
        continue
    high_is_better, vmin, vmax = metrics_metadata[col]
    
    vals = df[col].values
    
    if high_is_better is True:
        n = (vals - vmin) / (vmax - vmin)
    elif high_is_better is False:
        n = 1.0 - (vals - vmin) / (vmax - vmin)
    else:
        # Custom for NND
        n = (vals - 0.9) / (1.0 - 0.9)
        # Handle nan gracefully when clipping
        n_fixed = []
        for i, val in enumerate(vals):
            if pd.isna(val):
                n_fixed.append(np.nan)
            else:
                if val > 1.0:
                    n_fixed.append(1.0)
                else:
                    n_fixed.append(n[i])
        n = np.array(n_fixed)
        
    # Clip and fill nans temporarily for norm
    n = np.clip(np.array(n, dtype=float), 0, 1)
    df_norm[col] = n

fig, ax = plt.subplots(figsize=(16, 12))

# Plot imshow
im = ax.imshow(df_norm.values.astype(float), cmap="RdYlGn", aspect='auto', interpolation='none', 
               vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.index)))
ax.set_xticklabels(df.columns, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticklabels(df.index)

# Loop over data dimensions and create text annotations
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        val = df.iloc[i, j]
        if not pd.isna(val):
            # Choose text color based on background luminance
            norm_val = df_norm.iloc[i, j]
            # Simple heuristic: extreme values are dark in diverging colormap, mid values are light
            # Actually RdYlGn is light in the middle.
            text_color = "black" if (0.2 < norm_val < 0.8) else "white"
            
            # Format value depending on scale
            if abs(val) < 0.01 and val != 0:
                txt = f"{val:.1e}"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=8)

plt.title("Comprehensive Model Evaluation Heatmap\n(Green = Better, Red = Worse, White/Blank = NaN/Not Evaluated)")
plt.xlabel("Metrics")
plt.ylabel("Models")

# Add a subtle grid
ax.set_xticks(np.arange(-.5, len(df.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(df.index), 1), minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig('plots/eval_overnight/comprehensive_heatmap.png', dpi=150)
print("Heatmap saved to plots/eval_overnight/comprehensive_heatmap.png")
