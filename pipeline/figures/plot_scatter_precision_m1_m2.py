#!/usr/bin/env python
"""
Plot: Scatter plot of m1 vs m2 color-coded by precision metric.

This script generates scatter plots for all available precision metrics,
showing the primary vs secondary mass space with color-coding indicating
measurement precision levels.

Cell 43 from degradation_analysis.ipynb
"""

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path and change to pipeline directory for data files
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(script_dir)
sys.path.insert(0, pipeline_dir)
os.chdir(pipeline_dir)

# Use the physrev style if available
try:
    plt.style.use('physrev.mplstyle')
except:
    pass

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------
spin_a = 0.99  # Prograde spin
tpl_val = 0.25  # Plunge time
degradation = 1.0
tolerance = 1e-6
selected_run_type = 'circular'

# Label mapping for precision metrics
ylabel_map = {
    "relative_precision_m1_det": r"$\sigma_{m_{ 1,\mathrm{det} } }/m_{ 1,\mathrm{det} }$",
    "relative_precision_m1": r"$\sigma_{m_{1} }/m_{1}$",
    "relative_precision_m2_det": r"$\sigma_{m_{ 2,\mathrm{det} } }/m_{ 2,\mathrm{det} }$",
    "relative_precision_m2": r"$\sigma_{m_{2} }/m_{2}$",
    "relative_precision_dist": r"$\sigma_{d_L}/d_L$",
    "relative_precision_e0": r"$\sigma_{e_0}/e_0$",
    "absolute_precision_a": r"$\sigma_{a}$",
    "relative_precision_a": r"$\sigma_{a}/a$",
    "absolute_precision_OmegaS": r"Sky Localization $\Delta \Omega_S [\mathrm{deg}^2]$",
    "snr": "SNR",
}

# -----------------------------------------------------------------------------
# Load inference data
# -----------------------------------------------------------------------------
inference_files = sorted(glob.glob('inference_*/inference.h5'))
print(f"Found {len(inference_files)} inference.h5 files")

inference_metadata = {}
inference_precision_data = {}

for idx, inf_file in enumerate(inference_files):
    source_id = int(inf_file.split('_')[1].split('/')[0])
    
    with h5py.File(inf_file, 'r') as f:
        for run_type in ['circular', 'eccentric']:
            if run_type not in f.keys():
                continue
            
            run_group = f[run_type]
            source_key = (source_id, run_type)
            
            inference_metadata[source_key] = {
                'm1': float(np.round(run_group['m1'][()], decimals=5)),
                'm2': float(np.round(run_group['m2'][()], decimals=5)),
                'a': float(run_group['a'][()]),
                'p0': float(run_group['p0'][()]),
                'e0': float(run_group['e0'][()]),
                'e_f': float(run_group['e_f'][()]),
                'dist': float(run_group['dist'][()]),
                'T': float(np.round(run_group['Tpl'][()], decimals=5)),
                'redshift': float(run_group['redshift'][()]),
                'snr': run_group['snr'][()],
                'run_type': run_type,
            }
            
            detector_precision = run_group['detector_measurement_precision'][()]
            source_precision = run_group['source_measurement_precision'][()]
            param_names = run_group['param_names'][()]
            param_names = np.array(param_names, dtype=str).tolist()
            inference_metadata[source_key].update({"param_names": param_names})

            # populate precision dict for this source
            if source_key not in inference_precision_data:
                inference_precision_data[source_key] = {}

            for ii, name in enumerate(param_names):
                if name == 'M':
                    inference_precision_data[source_key].update({
                        "relative_precision_m1_det": detector_precision[:, param_names.index(name)] / (inference_metadata[source_key]['m1'] * (1 + inference_metadata[source_key]['redshift'])),
                        "relative_precision_m1": source_precision[:, param_names.index(name)] / inference_metadata[source_key]['m1']
                    })
                elif name == 'mu':
                    inference_precision_data[source_key].update({
                        "relative_precision_m2_det": detector_precision[:, param_names.index(name)] / (inference_metadata[source_key]['m2'] * (1 + inference_metadata[source_key]['redshift'])),
                        "relative_precision_m2": source_precision[:, param_names.index(name)] / inference_metadata[source_key]['m2']
                    })
                elif name == 'e0':
                    inference_precision_data[source_key].update({
                        "relative_precision_e0": detector_precision[:, param_names.index(name)] / inference_metadata[source_key]['e0']
                    })
                else:
                    inference_precision_data[source_key].update({
                        "absolute_precision_" + name: detector_precision[:, param_names.index(name)]
                    })
                
                if name == 'dist':
                    inference_precision_data[source_key].update({
                        "relative_precision_" + name: detector_precision[:, param_names.index(name)] / inference_metadata[source_key][name]
                    })
                if name == 'a':
                    inference_precision_data[source_key].update({
                        "relative_precision_" + name: detector_precision[:, param_names.index(name)] / inference_metadata[source_key][name]
                    })

            inference_precision_data[source_key].update({"snr": run_group['snr'][()]})

print(f"Loaded metadata for {len(inference_metadata)} sources")

# Add SNR to ylabel_map
ylabel_map['snr'] = 'SNR'

# -----------------------------------------------------------------------------
# Generate plots for each precision metric
# -----------------------------------------------------------------------------
for precision_metric in list(ylabel_map.keys()):
    scatter_data = {'m1': [], 'm2': [], 'precision': [], 'source_id': []}

    for src_key in sorted(inference_metadata.keys()):
        source_id, run_type = src_key
        
        # Filter by spin and Tpl
        src_a = inference_metadata[src_key]['a']
        src_tpl = inference_metadata[src_key]['T']
        if abs(src_a - spin_a) > tolerance or abs(src_tpl - tpl_val) > tolerance:
            continue
        
        m1 = inference_metadata[src_key]['m1']
        m2 = inference_metadata[src_key]['m2']
        
        # Check if this precision metric exists for this source
        if precision_metric not in inference_precision_data[src_key]:
            continue
        
        # Get precision array and compute median (across 100 realizations)
        precision_array = inference_precision_data[src_key][precision_metric]
        precision_median = np.median(precision_array)
        precision_deg = precision_median * np.sqrt(degradation)
        
        # Only add selected run type to avoid duplication
        if run_type == selected_run_type:
            scatter_data['m1'].append(m1)
            scatter_data['m2'].append(m2)
            scatter_data['precision'].append(precision_deg)
            scatter_data['source_id'].append(source_id)

    if not scatter_data['m1']:
        print(f"No data found for spin a={spin_a:.2f}, Tpl={tpl_val:.2f} and metric {precision_metric}")
        continue
    
    fig, ax = plt.subplots(figsize=(3.25, 1.5*2.0))
    
    # Create scatter plot
    m1_arr = np.array(scatter_data['m1'])
    m2_arr = np.array(scatter_data['m2'])
    precision_arr = np.abs(np.array(scatter_data['precision']))
    
    # Create scatter plot with color mapping
    prec_min = float(precision_arr.min())
    prec_max = float(precision_arr.max())
    vmin_decade = prec_min * 0.7
    vmax_decade = prec_max * 1.5
    if 'dist' in precision_metric:
        norm = plt.matplotlib.colors.Normalize(vmin=vmin_decade, vmax=vmax_decade)
    else:
        norm = plt.matplotlib.colors.LogNorm(vmin=vmin_decade, vmax=vmax_decade)
        
    scatter = ax.scatter(m1_arr, m2_arr, c=precision_arr, s=200, cmap='cividis',
                         alpha=0.7, edgecolors='black', linewidth=0.5,
                         norm=norm)
    
    ax.set_xlabel(r'Primary mass $m_1 [M_\odot]$')
    ax.set_ylabel(r'Secondary mass $m_2 [M_\odot]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 3e4)
    
    # Add colorbar above the plot
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', location='top')
    cbar.set_label(ylabel_map[precision_metric], fontsize=12, labelpad=10)
    
    # Set colorbar ticks to the unique precision values
    unique_precision_values = np.sort(np.unique(precision_arr))

    # Add full-width tick lines across the colorbar for min/max values
    cbar_ax = cbar.ax
    for val in [unique_precision_values.min(), unique_precision_values.max()]:
        cbar_ax.axvline(x=val, color='green', linewidth=1.5, alpha=1.0, linestyle='-')
        
    for val in [unique_precision_values.min(), unique_precision_values.max()]:
        cbar_ax.axvline(x=val * 1.4, color='red', linewidth=1.5, alpha=1.0, linestyle='--')

    plt.tight_layout()
    output_filename = f'scatter_{precision_metric}_m1_vs_m2_spin_a{spin_a}.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved: figures/{output_filename}")
    # plt.show()

    # Generate markdown table for this plot's data
    sorted_data = sorted(zip(scatter_data['source_id'], scatter_data['m1'], scatter_data['m2'], scatter_data['precision']))
    header = "| Source ID | m1 | m2 | Precision |"
    divider = "|---|---|---|---|"
    markdown_content = f"# {precision_metric} Scatter Plot Data\n\n"
    markdown_content += f"Data for spin a={spin_a:.2f}, Tpl={tpl_val:.2f}, run_type={selected_run_type}\n\n"
    markdown_content += header + "\n" + divider + "\n"
    for sid, m1, m2, prec in sorted_data:
        markdown_content += f"| {sid} | {m1:.2e} | {m2:.2e} | {prec:.2e} |\n"
    
    output_filename_md = f'scatter_{precision_metric}_m1_vs_m2_spin_a{spin_a}_data.md'
    with open(os.path.join(script_dir, output_filename_md), "w") as f:
        f.write(markdown_content)
    print(f"Data table saved: figures/{output_filename_md}")

# -----------------------------------------------------------------------------
# Generate markdown and LaTeX table summary (including redshift)
# -----------------------------------------------------------------------------
print("\nGenerating precision summary table (markdown + LaTeX)...")

# Use only the requested metrics (in this specific order)
desired_metrics = [
    'absolute_precision_OmegaS',
    'relative_precision_dist',
    'relative_precision_a',
    'relative_precision_e0',
    'relative_precision_m1',
    'relative_precision_m1_det',
    'relative_precision_m2',
    'relative_precision_m2_det',
]

# Prepare markdown table header (add redshift column)
all_metrics = desired_metrics
header = "| m1 | m2 | a | e0 | z | " + " | ".join(all_metrics) + " |"
divider = "|---" * (5 + len(all_metrics)) + "|"

# Group sources by unique Tpl values and create separate tables per Tpl
tpl_values = sorted({round(inference_metadata[k]['T'], 8) for k in inference_metadata.keys()})

for tpl_group in tpl_values:
    rows = []
    latex_rows = []
    for src_key in sorted(inference_metadata.keys()):
        src_tpl = inference_metadata[src_key]['T']
        if abs(src_tpl - tpl_group) > tolerance:
            continue

        m1 = inference_metadata[src_key]['m1']
        m2 = inference_metadata[src_key]['m2']
        a = inference_metadata[src_key]['a']
        e0 = inference_metadata[src_key]['e0']
        z = inference_metadata[src_key]['redshift']
        row = [f"{m1:.0e}", f"{m2:.0e}", f"{a:.2f}", f"{e0:.3f}", f"{z:.3f}"]
        latex_row = [f"{m1:.0e}", f"{m2:.0e}", f"{a:.2f}", f"{e0:.3f}", f"{z:.3f}"]
        for metric in all_metrics:
            if metric in inference_precision_data.get(src_key, {}):
                val = np.median(inference_precision_data[src_key][metric])
                row.append(f"{val:.2e}")
                latex_row.append(f"{val:.2e}")
            else:
                row.append("-")
                latex_row.append("-")
        rows.append("| " + " | ".join(row) + " |")
        latex_rows.append(" & ".join(latex_row) + " \\\\ \n")

    # Write to markdown file for this Tpl group
    markdown_content = "# Precision Summary Table\n\n"
    markdown_content += f"This table summarizes the median precision for each metric and each source (Tpl={tpl_group}).\n\n"
    markdown_content += header + "\n"
    markdown_content += divider + "\n"
    for row in rows:
        markdown_content += row + "\n"

    md_path = os.path.join(script_dir, f"precision_summary_Tpl{tpl_group:.2f}.md")
    with open(md_path, "w") as f:
        f.write(markdown_content)

    # Also write a LaTeX file with a tabular of the same information for this Tpl
    tex_header_cols = "c" * (5 + len(all_metrics))
    tex_path = os.path.join(script_dir, f"precision_summary_Tpl{tpl_group:.2f}.tex")
    latex_label_map = {
        'm1': r"$m_1$",
        'm2': r"$m_2$",
        'a': r"$a$",
        'e0': r"$e_0$",
        'redshift': r"$z$",
        'absolute_precision_OmegaS': r"$\Delta \Omega_S [\mathrm{deg}^2]$",
        'absolute_precision_dist': r"$\sigma_{d_L}$",
        'relative_precision_a': ylabel_map.get('relative_precision_a', r"$\sigma_a/a$"),
        'relative_precision_dist': ylabel_map.get('relative_precision_dist', r"$\sigma_{d_L}/d_L$"),
        'relative_precision_e0': ylabel_map.get('relative_precision_e0', r"$\sigma_{e_0}/e_0$"),
        'relative_precision_m1': ylabel_map.get('relative_precision_m1', r"$\sigma_{m_1}/m_1$"),
        'relative_precision_m1_det': ylabel_map.get('relative_precision_m1_det', r"$\sigma_{m_{1,\mathrm{det}}}/m_{1,\mathrm{det}}$"),
        'relative_precision_m2': ylabel_map.get('relative_precision_m2', r"$\sigma_{m_2}/m_2$"),
        'relative_precision_m2_det': ylabel_map.get('relative_precision_m2_det', r"$\sigma_{m_{2,\mathrm{det}}}/m_{2,\mathrm{det}}$"),
        'snr': ylabel_map.get('snr', 'SNR')
    }

    with open(tex_path, "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Precision summary table (median values)}\n")
        f.write('\\begin{tabular}{%s}\n' % tex_header_cols)
        f.write("\\toprule\n")
        # LaTeX header row (use math-formatted labels)
        tex_header = [latex_label_map['m1'], latex_label_map['m2'], latex_label_map['a'], latex_label_map['e0'], latex_label_map['redshift']] + [latex_label_map.get(m, m.replace('_','\\_')) for m in all_metrics]
        f.write(" & ".join(tex_header) + " \\\\ \n")
        f.write("\\midrule\n")
        # Rows
        for lr in latex_rows:
            f.write(lr)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

    print(f"Precision summary saved: {md_path}")
    print(f"LaTeX precision summary saved: {tex_path}")
