#!/usr/bin/env python3
"""
Generate LaTeX tables from source files
"""
import ast

def parse_file(filename):
    """Parse a file containing Python dictionaries and extract specified columns"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = ast.literal_eval(line.strip())
                    data.append({
                        'M': entry['M'],
                        'mu': entry['mu'],
                        'a': entry['a'],
                        'e_f': entry['e_f'],
                        'T': entry['T'],
                        'z': entry['z']
                    })
                except:
                    pass
    return data

def create_latex_table(data, caption, label):
    """Create a LaTeX table from the data"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{cccccc}")
    lines.append(r"\hline")
    lines.append(r"$m_{1,\mathrm{det}} [M_\odot]$ & $m_{2,\mathrm{det}} [M_\odot]$ & $a$ & $e_f$ & $T$ [year] & $z$ \\")
    lines.append(r"\hline")
    
    for entry in data:
        # Format numbers appropriately
        M_str = f"{entry['M']:.6e}"# if entry['M'] >= 1000 else f"{entry['M']:.2f}"
        mu_str = f"{entry['mu']:.6e}"# if entry['mu'] >= 1000 else f"{entry['mu']:.2f}"
        a_str = f"{entry['a']:.2f}"
        ef_str = f"{entry['e_f']:.2f}"
        T_str = f"{entry['T']:.2f}"
        z_str = f"{entry['z']:.3f}"
        
        lines.append(f"{M_str} & {mu_str} & {a_str} & {ef_str} & {T_str} & {z_str} \\\\")
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)

def main():
    # Parse SNR and inference files
    snr_data = parse_file('./snr_sources_snr.txt')
    inference_data = parse_file('./inference_sources_pe.txt')

    # Split inference data by eccentricity
    inference_ef0 = [entry for entry in inference_data if entry['e_f'] == 0.0]
    inference_ef01 = [entry for entry in inference_data if entry['e_f'] == 0.01]

    # Create LaTeX document
    latex_doc = []
    latex_doc.append(r"\documentclass{article}")
    latex_doc.append(r"\usepackage{booktabs}")
    latex_doc.append(r"\usepackage{longtable}")
    latex_doc.append(r"\usepackage{geometry}")
    latex_doc.append(r"\geometry{margin=1in}")
    latex_doc.append(r"")
    latex_doc.append(r"\begin{document}")
    latex_doc.append(r"")
    latex_doc.append(r"\section*{Detection SNR Source Parameters}")
    latex_doc.append(r"")

    # Break SNR table into chunks (max 50 entries per table)
    max_entries = 50
    for i in range(0, len(snr_data), max_entries):
        chunk = snr_data[i:i+max_entries]
        latex_doc.append(create_latex_table(
            chunk,
            f"Detection SNR Sources Parameters (entries {i+1}-{min(i+max_entries, len(snr_data))})",
            f"tab:snr_sources_{i//max_entries+1}"
        ))
        latex_doc.append(r"")
        latex_doc.append(r"\clearpage")
        latex_doc.append(r"")

    latex_doc.append(r"\section*{Inference Source Parameters}")
    latex_doc.append(r"")
    # Break inference_ef0 into chunks
    for i in range(0, len(inference_ef0), max_entries):
        chunk = inference_ef0[i:i+max_entries]
        latex_doc.append(create_latex_table(
            chunk,
            f"Inference Sources Parameters ($e_f=0.0$) (entries {i+1}-{min(i+max_entries, len(inference_ef0))})",
            f"tab:inference_sources_ef0_{i//max_entries+1}"
        ))
        latex_doc.append(r"")
        latex_doc.append(r"\clearpage")
        latex_doc.append(r"")
    # Break inference_ef01 into chunks
    for i in range(0, len(inference_ef01), max_entries):
        chunk = inference_ef01[i:i+max_entries]
        latex_doc.append(create_latex_table(
            chunk,
            f"Inference Sources Parameters ($e_f=0.01$) (entries {i+1}-{min(i+max_entries, len(inference_ef01))})",
            f"tab:inference_sources_ef01_{i//max_entries+1}"
        ))
        latex_doc.append(r"")
        latex_doc.append(r"\clearpage")
        latex_doc.append(r"")
    latex_doc.append(r"\end{document}")

    # Write to file
    output_filename = 'source_tables.tex'
    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_doc))

    print(f"LaTeX document created: {output_filename}")
    print(f"Detection SNR sources: {len(snr_data)} entries")
    print(f"Inference sources (e_f=0.0): {len(inference_ef0)} entries")
    print(f"Inference sources (e_f=0.01): {len(inference_ef01)} entries")

if __name__ == "__main__":
    main()
