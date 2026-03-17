# -*- coding: utf-8 -*-
r"""从 reference_values 与运行结果生成完整 LaTeX 表格，供稿件直接使用或 \input。"""

import sys
import os
# 允许从仓库根或 experiments 目录运行
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, os.path.dirname(_here))
try:
    from experiments import reference_values as ref
except ImportError:
    import reference_values as ref


def table_memory_gap(overrides=None):
    """Table 1: Memory-only vs DualGraph 四域。overrides: dict 同 MEMORY_GAP 结构，可部分覆盖。"""
    data = ref.MEMORY_GAP.copy()
    if overrides:
        for d, rows in overrides.items():
            if d in data:
                data[d].update(rows)
    lines = [
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Condition} & \textbf{Turns}",
        r"  & \textbf{Acq.\ (\%)} & \textbf{Conc.\ (\%)}",
        r"  & \textbf{Un-om.\ (\%)} & $p$ \\",
        r"\midrule",
    ]
    for domain in ref.DOMAINS:
        mo = data[domain]["Memory-only"]
        dg = data[domain]["DualGraph"]
        t_mo = f"${mo['turns'][0]:.1f} \\pm {mo['turns'][1]:.1f}$"
        t_dg = f"${dg['turns'][0]:.1f} \\pm {dg['turns'][1]:.1f}$"
        lines.append(r"\multirow{2}{*}{" + domain + "}")
        lines.append(f"  & Memory-only & ${t_mo}$  & {mo['acq']} & {mo['conc']} & {mo['un_om']} & {mo['p']} \\\\")
        lines.append(f"  & DualGraph   & ${t_dg}$ & {dg['acq']} & {dg['conc']} & {dg['un_om']} & {dg['p']} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1]  # drop last addlinespace
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_baselines(overrides=None):
    """Table 2: 七方法四域 Acq, Conc, Trn。"""
    data = ref.BASELINES if overrides is None else {**ref.BASELINES, **overrides}
    domains = ref.DOMAINS
    methods = list(data.keys())
    lines = [
        r"\begin{tabular}{@{}lccccccccccccc@{}}",
        r"\toprule",
        "& " + " & ".join([r"\multicolumn{3}{c}{\textbf{" + d + "}}" for d in domains]) + " \\\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"\textbf{Method} & " + " & ".join(["Acq. & Conc. & Trn"] * 4) + " \\\\",
        r"\midrule",
    ]
    for method in methods:
        row = [method]
        for dom in domains:
            a, c, t = data[method][dom]
            row.extend([str(a), str(c), f"{t:.1f}"])
        lines.append(" & ".join(row) + " \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_evolving(overrides=None):
    """Table 3: Evolving DAG Pre-specified / Emergent Acq. Hyp. & D3。"""
    data = ref.EVOLVING if overrides is None else {**ref.EVOLVING, **overrides}
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{Pre-specified Acq.\ (\%)}}",
        r"& \multicolumn{2}{c}{\textbf{Emergent Acq.\ (\%)}}",
        r"& \textbf{Graph} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Method} & Hyp. & D3 & Hyp. & D3 & \textbf{Growth} \\",
        r"\midrule",
    ]
    for method, row in data.items():
        m = method.replace("DualGraph (static)", r"\textbf{DualGraph (static)}").replace("DualGraph (evolving)", r"\textbf{DualGraph (evolving)}")
        lines.append(f"{m} & {row['pre_hyp']} & {row['pre_d3']} & {row['em_hyp']} & {row['em_d3']} & {row['growth']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_ncs_validation(overrides=None):
    """Table 4: NCS validation。"""
    d = ref.NCS_VALIDATION if overrides is None else {**ref.NCS_VALIDATION, **overrides}
    lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{NCS mode} & \textbf{Global recompute} \\",
        r"\midrule",
        f"Score match rate vs.\\ global  & {d['score_match_rate']} & --- \\\\",
        f"NCS prediction match rate     & {d['NCS_prediction_match_rate_pct']}\\% & --- \\\\",
        f"Mean $|\\mathcal{{U}}_t|$ per turn & {d['mean_U_t_per_turn']} & {d['global_recompute_per_turn']} \\\\",
        f"Mean score oscillations per entity & {d['mean_oscillations_ncs']} & {d['mean_oscillations_global']} \\\\",
        f"Mean per-turn score-update time (ms) & {d['mean_update_time_ms_ncs']} & {d['mean_update_time_ms_global']} \\\\",
        f"Entity acquisition (\\%)       & {d['entity_acquisition_pct_ncs']} & {d['entity_acquisition_pct_global']} \\\\",
        f"Concordance (\\%)              & {d['concordance_pct_ncs']} & {d['concordance_pct_global']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_ablation(overrides=None):
    """Table 5: Ablation。"""
    rows = overrides if overrides is not None else ref.ABLATION
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\textbf{Condition} & \textbf{Acq.\ (\%)} & $\Delta$\textbf{Acq.}",
        r"  & \textbf{Conc.\ (\%)} & \textbf{Turns} & \textbf{Coherence} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['condition']} & {r['acq']} & {r['delta_acq']} & {r['conc']} & {r['turns']} & {r['coherence']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_graph_construction(overrides=None):
    """Table 6: Graph construction 四域。"""
    data = ref.GRAPH_CONSTRUCTION if overrides is None else {**ref.GRAPH_CONSTRUCTION, **overrides}
    lines = [
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Edge Agreement (F1)}}",
        r"& \multicolumn{3}{c}{\textbf{Downstream Acq.\ (\%)}}",
        r"& \textbf{Review} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Domain} & Prereq. & Assoc. & Overall",
        r"  & Auto & Auto+Rev. & Expert & \textbf{(min)} \\",
        r"\midrule",
    ]
    for domain in ref.DOMAINS:
        r = data[domain]
        lines.append(f"{domain} & {r['prereq_f1']} & {r['assoc_f1']} & {r['overall_f1']} & {r['auto_acq']} & {r['autorev_acq']} & {r['expert_acq']} & {r['review_min']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_downstream_hypertension_diabetes():
    """Table 7 临床部分：Hypertension 与 Diabetes。"""
    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Method}",
        r"  & \textbf{Classif.\ (\%)} & \textbf{Risk (\%)}",
        r"  & \textbf{Med.\ (1--5)} & \textbf{Follow-up (1--5)} \\",
        r"\midrule",
    ]
    for domain in ["Hypertension", "Diabetes"]:
        for r in ref.DOWNSTREAM[domain]:
            method = r["method"]
            if "classif" in r:
                lines.append(f"{domain} & {method}  & {r['classif']} & {r['risk']} & {r['med']} & {r['followup']} \\\\")
            else:
                break
        lines.append(r"\addlinespace")
    return "\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}"])


def table_locomo(overrides=None):
    """Appendix LoCoMo 表。overrides 可提供 DualGraph 实际运行结果。"""
    data = ref.LOCOMO_TABLE.copy()
    if overrides:
        data.update(overrides)
    methods = ["A-mem", "Mem0", "MemGPT", "memg", "zep", "DualGraph"]
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Overall} & \textbf{Single-hop}",
        r"  & \textbf{Multi-hop} & \textbf{Temporal} \\",
        r"\midrule",
    ]
    for m in methods:
        r = data[m]
        bold = "\\textbf{" if m == "DualGraph" else ""
        bold_end = "}" if m == "DualGraph" else ""
        lines.append(f"{m} & {bold}{r['overall']}{bold_end} & {bold}{r['single_hop']}{bold_end} & {bold}{r['multi_hop']}{bold_end} & {bold}{r['temporal']}{bold_end} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_graph_structure():
    """图结构表 (Methods)。"""
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Domain} & $|V|$ & $|E_P|$ & $|E_A|$ & Depth $L$",
        r"  & $\Delta_{\max}$ & Communities \\",
        r"\midrule",
    ]
    for domain in ref.DOMAINS:
        r = ref.GRAPH_STRUCTURE[domain]
        lines.append(f"{domain} & {r['|V|']} & {r['|E_P|']} & {r['|E_A|']} & {r['depth_L']} & {r['delta_max']} & {r['communities']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
