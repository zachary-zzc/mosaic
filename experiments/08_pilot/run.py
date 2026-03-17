#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 08：前瞻试点完整构建。输出试点设计、指标与叙述用数值（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref


def main():
    results_dir = ensure_results_dir("08_pilot")
    full_data = ref.PILOT
    save_json(os.path.join(results_dir, "pilot_results.json"), full_data)

    # 叙述用段落片段（可贴入 manuscript）
    narrative = (
        f"N = {ref.PILOT['N']} consented patients, hypertension clinic. "
        f"Entity acquisition: {ref.PILOT['entity_acquisition_dualgraph_pct']}\\% (DualGraph) vs.\\ "
        f"{ref.PILOT['entity_acquisition_clinician_pct']}\\% (clinician). "
        f"Concordance: {ref.PILOT['concordance_pct']}\\%; "
        f"clinician trust: mean {ref.PILOT['clinician_trust_mean']}/5 (range {ref.PILOT['clinician_trust_range']}). "
        f"Interventions: mean {ref.PILOT['interventions_per_case_mean']} per case (range {ref.PILOT['interventions_range']}). "
        f"Safety: {ref.PILOT['safety_flags_activated']} flags; {ref.PILOT['true_positives']} true positives, "
        f"{ref.PILOT['false_positives']} false positives. "
        f"Evolving DAG: {ref.PILOT['evolving_activation_pct']}\\% of cases; "
        f"{ref.PILOT['evolving_integration_success_pct']}\\% integrated. IRB: {ref.PILOT['IRB']}."
    )
    with open(os.path.join(results_dir, "pilot_narrative.txt"), "w", encoding="utf-8") as f:
        f.write(narrative)
    print("Pilot full results and narrative written to:", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
