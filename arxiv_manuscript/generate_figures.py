#!/usr/bin/env python3
"""Generate all figures for MOSAIC arXiv manuscript."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# --- Nature-style settings (Times New Roman) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titlepad': 12,
    'axes.labelsize': 9,
    'axes.labelpad': 6,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'figure.constrained_layout.use': False,
})

# Color palette (Nature-style, colorblind-safe)
COLORS = {
    'mosaic': '#E64B35',    # Red (our method)
    'memos': '#4DBBD5',     # Cyan
    'mem0': '#00A087',      # Teal
    'mem0g': '#3C5488',     # Dark blue
    'zep': '#F39B7F',       # Salmon
    'memgpt': '#8491B4',    # Blue-grey
    'amem': '#91D1C2',      # Light teal
    'langmem': '#B09C85',   # Tan
    'openai': '#7E6148',    # Brown
    'supermem': '#DC9FB4',   # Pink
    'memobase': '#CCCCCC',  # Grey
}

BASEDIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASEDIR, 'images')


def save_fig(fig, folder, name):
    """Save figure as both PDF and PNG."""
    path = os.path.join(IMG_DIR, folder)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'{name}.pdf'))
    fig.savefig(os.path.join(path, f'{name}.png'))
    plt.close(fig)
    print(f'  Saved {folder}/{name}.pdf + .png')


# ============================================================================
# FIGURE 2: LoCoMo Results
# ============================================================================
def gen_figure2():
    print('Generating Figure 2 (LoCoMo)...')

    methods = ['A-Mem', 'LangMem', 'Zep', 'OpenAI', 'Mem0', 'MemGPT', 'MOSAIC']
    colors = [COLORS['amem'], COLORS['langmem'], COLORS['zep'], COLORS['openai'],
              COLORS['mem0'], COLORS['memgpt'], COLORS['mosaic']]

    # --- 2a: Overall accuracy bar chart ---
    overall = [38.95, 52.08, 56.32, 51.10, 62.14, 61.36, 79.68]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.bar(methods, overall, color=colors, edgecolor='white', linewidth=0.5, width=0.7)
    # Highlight MOSAIC bar
    bars[-1].set_edgecolor('#B03020')
    bars[-1].set_linewidth(1.5)
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('LoCoMo Benchmark \u2014 Overall', fontweight='bold')
    ax.set_ylim(0, 100)
    # Add value labels
    for bar, val in zip(bars, overall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=35)
    fig.subplots_adjust(bottom=0.18)
    # Add delta annotation
    ax.annotate('+17.5 pp', xy=(6, 79.68), xytext=(5.2, 88),
                fontsize=8, fontweight='bold', color=COLORS['mosaic'],
                arrowprops=dict(arrowstyle='->', color=COLORS['mosaic'], lw=1.2))
    save_fig(fig, 'figure2', 'locomo_overall')

    # --- 2b: Per-category grouped bar chart ---
    categories = ['Single-hop\n(841)', 'Multi-hop\n(282)', 'Open-domain\n(96)', 'Temporal\n(321)']
    data = {
        'A-Mem':   [39.79, 18.85, 54.05, 49.91],
        'LangMem': [62.23, 47.92, 71.12, 23.43],
        'Zep':     [61.70, 41.35, 76.60, 49.31],
        'OpenAI':  [63.79, 42.92, 62.29, 21.71],
        'Mem0':    [67.13, 51.15, 72.93, 55.51],
        'MemGPT':  [65.71, 47.19, 75.71, 58.13],
        'MOSAIC':  [84.78, 76.95, 61.46, 74.14],
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    n_methods = len(methods)
    n_cats = len(categories)
    width = 0.11
    x = np.arange(n_cats)

    for i, (method, vals) in enumerate(data.items()):
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=colors[i],
                      edgecolor='white', linewidth=0.3)
        if method == 'MOSAIC':
            for bar in bars:
                bar.set_edgecolor('#B03020')
                bar.set_linewidth(1.0)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LoCoMo \u2014 Per-Category Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              frameon=False, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.25, top=0.90)
    save_fig(fig, 'figure2', 'locomo_category')

    # --- 2c: Per-conversation line chart ---
    convs = [f'conv{i}' for i in range(10)]
    conv_overall = [79.61, 80.25, 84.21, 75.88, 75.28, 82.93, 83.33, 74.87, 80.77, 83.54]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(convs, conv_overall, 'o-', color=COLORS['mosaic'], linewidth=1.5,
            markersize=6, markeredgecolor='white', markeredgewidth=0.8, zorder=5)
    ax.axhline(y=79.68, color=COLORS['mosaic'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(9.4, 79.68, 'Mean\n79.68%', fontsize=6.5, color=COLORS['mosaic'],
            va='center', alpha=0.7)

    # Shade best baseline range
    ax.axhspan(38.95, 62.14, alpha=0.08, color='grey')
    ax.text(0.1, 50, 'Baseline range\n(38.9–62.1%)', fontsize=6.5, color='grey',
            va='center', alpha=0.8)

    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('LoCoMo \u2014 Per-Conversation Performance', fontweight='bold')
    ax.set_ylim(30, 98)
    ax.tick_params(axis='x', rotation=35)

    for i, v in enumerate(conv_overall):
        ax.text(i, v + 2.0, f'{v:.1f}', ha='center', fontsize=5.5, color=COLORS['mosaic'])

    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, 'figure2', 'locomo_conversation')


# ============================================================================
# FIGURE 3: HaluMem Results
# ============================================================================
def gen_figure3():
    print('Generating Figure 3 (HaluMem)...')

    systems_ext = ['MemOS', 'Mem0-G', 'Mem0', 'SuperM', 'Memobase', 'MOSAIC']
    colors_ext = [COLORS['memos'], COLORS['mem0g'], COLORS['mem0'],
                  COLORS['supermem'], COLORS['memobase'], COLORS['mosaic']]

    # --- 3a: Memory Extraction ---
    recall =   [0.741, 0.433, 0.429, 0.415, 0.146, 0.891]
    f1 =       [0.797, 0.579, 0.573, 0.569, 0.251, 0.893]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(systems_ext))
    width = 0.35
    b1 = ax.bar(x - width/2, recall, width, label='Recall', color=colors_ext,
                edgecolor='white', linewidth=0.5, alpha=0.7)
    b2 = ax.bar(x + width/2, f1, width, label='F1', color=colors_ext,
                edgecolor='white', linewidth=0.5)
    # Highlight MOSAIC
    b1[-1].set_edgecolor('#B03020'); b1[-1].set_linewidth(1.2)
    b2[-1].set_edgecolor('#B03020'); b2[-1].set_linewidth(1.2)

    ax.set_ylabel('Score')
    ax.set_title('Memory Extraction', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(systems_ext, rotation=35)
    ax.set_ylim(0, 1.12)

    # Custom legend for Recall vs F1
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='grey', alpha=0.5, label='Recall'),
                       Patch(facecolor='grey', alpha=1.0, label='F1')]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)

    for i, (r, f) in enumerate(zip(recall, f1)):
        ax.text(i - width/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=5.5, rotation=90, va='bottom')
        ax.text(i + width/2, f + 0.02, f'{f:.2f}', ha='center', fontsize=5.5, rotation=90, va='bottom')

    fig.subplots_adjust(bottom=0.20)
    save_fig(fig, 'figure3', 'halumem_extraction')

    # --- 3b: Memory Updating (stacked: correctness, hallucination, omission) ---
    systems_upd = ['MemOS', 'Zep', 'Mem0-G', 'Mem0', 'SuperM', 'Memobase', 'MOSAIC']
    colors_upd = [COLORS['memos'], COLORS['zep'], COLORS['mem0g'], COLORS['mem0'],
                  COLORS['supermem'], COLORS['memobase'], COLORS['mosaic']]

    correct = [0.621, 0.473, 0.245, 0.255, 0.164, 0.052, 0.699]
    halluc =  [0.004, 0.004, 0.003, 0.005, 0.012, 0.006, 0.014]
    omiss =   [0.375, 0.523, 0.752, 0.740, 0.825, 0.943, 0.280]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(systems_upd))
    b1 = ax.bar(x, correct, 0.6, label='Correct \u2191', color='#2CA02C', alpha=0.85)
    b2 = ax.bar(x, halluc, 0.6, bottom=correct, label='Hallucination \u2193', color='#D62728', alpha=0.85)
    b3 = ax.bar(x, omiss, 0.6, bottom=[c+h for c,h in zip(correct, halluc)],
                label='Omission \u2193', color='#BBBBBB', alpha=0.85)

    ax.set_ylabel('Proportion')
    ax.set_title('Memory Updating', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(systems_upd, rotation=35)
    ax.set_ylim(0, 1.10)
    fig.subplots_adjust(bottom=0.18)
    ax.legend(loc='upper right', frameon=False, fontsize=7)

    # Annotate MOSAIC correctness
    ax.text(6, correct[6]/2, f'{correct[6]:.3f}', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white')

    save_fig(fig, 'figure3', 'halumem_updating')

    # --- 3c: Type-wise accuracy ---
    systems_tw = ['Mem0', 'Mem0-G', 'Memobase', 'MemOS', 'SuperM', 'Zep', 'MOSAIC']
    colors_tw = [COLORS['mem0'], COLORS['mem0g'], COLORS['memobase'], COLORS['memos'],
                 COLORS['supermem'], COLORS['zep'], COLORS['mosaic']]
    event =  [0.297, 0.300, 0.051, 0.634, 0.287, 0.448, 0.744]
    persona = [0.337, 0.337, 0.134, 0.598, 0.321, 0.498, 0.811]
    relationship = [0.278, 0.266, 0.068, 0.624, 0.207, 0.388, 0.674]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(systems_tw))
    width = 0.22
    b1 = ax.bar(x - width, event, width, label='Event', color='#4C72B0')
    b2 = ax.bar(x, persona, width, label='Persona', color='#55A868')
    b3 = ax.bar(x + width, relationship, width, label='Relationship', color='#C44E52')

    # Highlight MOSAIC group
    for b in [b1[-1], b2[-1], b3[-1]]:
        b.set_edgecolor('#B03020')
        b.set_linewidth(1.2)

    ax.set_ylabel('Accuracy')
    ax.set_title('Type-wise Memory Accuracy', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(systems_tw, rotation=35)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', frameon=False)
    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, 'figure3', 'halumem_typewise')

    # --- 3d: QA evaluation ---
    systems_qa = ['MemOS', 'Zep', 'Mem0-G', 'Mem0', 'SuperM', 'Memobase', 'MOSAIC']
    colors_qa = [COLORS['memos'], COLORS['zep'], COLORS['mem0g'], COLORS['mem0'],
                 COLORS['supermem'], COLORS['memobase'], COLORS['mosaic']]

    qa_correct = [0.672, 0.555, 0.547, 0.530, 0.541, 0.353, 0.878]
    qa_halluc =  [0.152, 0.219, 0.193, 0.192, 0.222, 0.300, 0.079]
    qa_omiss =   [0.176, 0.226, 0.261, 0.278, 0.237, 0.347, 0.043]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5), sharey=True)
    titles = ['Correctness \u2191', 'Hallucination \u2193', 'Omission \u2193']
    data_sets = [qa_correct, qa_halluc, qa_omiss]

    for ax, title, data in zip(axes, titles, data_sets):
        bars = ax.barh(systems_qa, data, color=colors_qa, edgecolor='white', linewidth=0.5)
        bars[-1].set_edgecolor('#B03020'); bars[-1].set_linewidth(1.2)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.05)
        for i, v in enumerate(data):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=6)

    fig.suptitle('Question Answering Evaluation', fontweight='bold', y=0.98, fontsize=10)
    fig.subplots_adjust(wspace=0.08, top=0.88, bottom=0.08, left=0.14, right=0.97)
    save_fig(fig, 'figure3', 'halumem_qa')


# ============================================================================
# FIGURE 4: Error Compounding
# ============================================================================
def gen_figure4():
    print('Generating Figure 4 (Error Compounding)...')

    methods = ['A-Mem', 'LangMem', 'MemGPT', 'OpenAI', 'Zep', 'Mem0', 'MOSAIC']
    colors_err = [COLORS['amem'], COLORS['langmem'], COLORS['memgpt'], COLORS['openai'],
                  COLORS['zep'], COLORS['mem0'], COLORS['mosaic']]

    # --- 4a: By error type ---
    numerical = [0.071, 0.000, 0.071, 0.000, 0.000, 0.000, 0.643]
    semantic =  [0.231, 0.077, 0.154, 0.308, 0.077, 0.077, 0.692]
    logical =   [0.130, 0.000, 0.000, 0.130, 0.000, 0.040, 0.652]
    overall =   [0.14,  0.02,  0.06,  0.14,  0.02,  0.04,  0.66]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(methods))
    width = 0.18

    b1 = ax.bar(x - 1.5*width, numerical, width, label='Numerical (14)', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x - 0.5*width, semantic, width, label='Semantic (13)', color='#55A868', alpha=0.85)
    b3 = ax.bar(x + 0.5*width, logical, width, label='Logical (23)', color='#C44E52', alpha=0.85)
    b4 = ax.bar(x + 1.5*width, overall, width, label='Overall (50)', color='#333333', alpha=0.9)

    # Highlight MOSAIC
    for b in [b1[-1], b2[-1], b3[-1], b4[-1]]:
        b.set_edgecolor('#B03020')
        b.set_linewidth(1.2)

    ax.set_ylabel('Detection Rate')
    ax.set_title('Conflict Detection by Error Type', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35)
    ax.set_ylim(0, 0.90)
    fig.subplots_adjust(bottom=0.18)
    ax.legend(loc='upper left', frameon=False, fontsize=7)

    # Annotate 4.7x
    ax.annotate('4.7×', xy=(6 + 1.5*width, 0.66), xytext=(5.5, 0.78),
                fontsize=10, fontweight='bold', color=COLORS['mosaic'],
                arrowprops=dict(arrowstyle='->', color=COLORS['mosaic'], lw=1.5))

    save_fig(fig, 'figure4', 'error_by_type')

    # --- 4b: By visibility ---
    methods_vis = ['A-Mem', 'LangMem', 'MemGPT', 'OpenAI', 'Zep', 'Mem0', 'MOSAIC']
    colors_vis = [COLORS['amem'], COLORS['langmem'], COLORS['memgpt'],
                  COLORS['openai'], COLORS['zep'], COLORS['mem0'], COLORS['mosaic']]
    implicit = [0.000, 0.000, 0.091, 0.273, 0, 0, 0.727]
    explicit = [0.179, 0.026, 0.051, 0.103, 0.026, 0.051, 0.641]

    fig, ax = plt.subplots(figsize=(7, 4))  # 稍微增加宽度以适应更多的柱子
    x = np.arange(len(methods_vis))
    width = 0.32

    b1 = ax.bar(x - width / 2, implicit, width, label='Implicit (11)',
                color='#C44E52', alpha=0.85)
    b2 = ax.bar(x + width / 2, explicit, width, label='Explicit (39)',
                color='#4C72B0', alpha=0.85)

    # 为MOSAIC柱子添加边框（现在索引是6）
    b1[6].set_edgecolor('#B03020');
    b1[6].set_linewidth(1.2)
    b2[6].set_edgecolor('#B03020');
    b2[6].set_linewidth(1.2)

    ax.set_ylabel('Detection Rate')
    ax.set_title('Conflict Detection by Visibility', fontweight='bold')
    ax.set_xticks(x);
    ax.set_xticklabels(methods_vis, rotation=35)
    ax.set_ylim(0, 0.95)
    fig.subplots_adjust(bottom=0.18)  # 调整底部边距以适应旋转的标签
    ax.legend(loc='upper left', frameon=False)

    # 更新MOSAIC柱子的值标签位置（现在索引是6）
    mosaic_x = 6
    ax.text(mosaic_x - width / 2, 0.727 + 0.02, '72.7%', ha='center', fontsize=7,
            fontweight='bold', color=COLORS['mosaic'])
    ax.text(mosaic_x + width / 2, 0.641 + 0.02, '64.1%', ha='center', fontsize=7,
            fontweight='bold', color=COLORS['mosaic'])

    save_fig(fig, 'figure4', 'error_by_visibility')


# ============================================================================
# FIGURE 5: Efficiency
# ============================================================================
def gen_figure5():
    print('Generating Figure 5 (Efficiency)...')

    systems = ['SuperM', 'Memobase', 'MOSAIC', 'MemOS', 'Mem0', 'Mem0-G']
    colors_eff = [COLORS['supermem'], COLORS['memobase'], COLORS['mosaic'],
                  COLORS['memos'], COLORS['mem0'], COLORS['mem0g']]

    add_time =  [273.21, 293.30, 346.57, 1028.84, 2768.14, 2840.07]
    ret_time =  [95.53, 139.95, 0.46, 20.52, 41.66, 54.65]

    # --- 5a: Stacked bar total time ---
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(systems))
    b1 = ax.barh(x, add_time, 0.6, label='Dialogue Addition', color='#4C72B0', alpha=0.85)
    b2 = ax.barh(x, ret_time, 0.6, left=add_time, label='Retrieval', color='#C44E52', alpha=0.85)

    # Highlight MOSAIC
    b1[2].set_edgecolor('#B03020'); b1[2].set_linewidth(1.2)
    b2[2].set_edgecolor('#B03020'); b2[2].set_linewidth(1.2)

    ax.set_xlabel('Time (minutes)')
    ax.set_title('HaluMem Execution Time', fontweight='bold')
    ax.set_yticks(x); ax.set_yticklabels(systems)
    ax.legend(loc='lower right', frameon=False)

    # Add total time labels
    for i, (a, r) in enumerate(zip(add_time, ret_time)):
        total = a + r
        ax.text(total + 30, i, f'{total:.0f} min', va='center', fontsize=7)

    save_fig(fig, 'figure5', 'efficiency_time')

    # --- 5b: Retrieval time comparison (log scale) ---
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.barh(systems, ret_time, 0.6, color=colors_eff, edgecolor='white', linewidth=0.5)
    bars[2].set_edgecolor('#B03020'); bars[2].set_linewidth(1.5)

    ax.set_xlabel('Retrieval Time (minutes, log scale)')
    ax.set_title('Memory Retrieval Speed', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 300)

    for i, v in enumerate(ret_time):
        ax.text(v * 1.3, i, f'{v:.2f}' if v < 1 else f'{v:.1f}',
                va='center', fontsize=7, fontweight='bold' if i == 2 else 'normal')

    # 45x annotation
    ax.annotate('45× faster\nthan MemOS', xy=(0.46, 2), xytext=(5, 4.5),
                fontsize=8, fontweight='bold', color=COLORS['mosaic'],
                arrowprops=dict(arrowstyle='->', color=COLORS['mosaic'], lw=1.2),
                ha='center')

    save_fig(fig, 'figure5', 'efficiency_retrieval')


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print(f'Output directory: {IMG_DIR}')
    gen_figure2()
    gen_figure3()
    gen_figure4()
    gen_figure5()
    print('\nAll figures generated successfully!')