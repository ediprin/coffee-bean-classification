from __future__ import annotations
import argparse,csv,json
from pathlib import Path
ARMS=("R0","C0","F0","W0")
def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def generate_report(oof_summary_path,bootstrap_path,analysis_summary_path,efficiency_path,output_dir):
    oof=_json(oof_summary_path); bootstrap=_json(bootstrap_path); analysis=_json(analysis_summary_path); efficiency=_json(efficiency_path) if efficiency_path is not None else None; output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arm in ARMS:
        metrics=oof["arms"][arm]; delta=bootstrap["point_delta_vs_R0"].get(arm); ci=bootstrap["paired_bootstrap_delta_vs_R0"][arm]["macro_f1"] if arm!="R0" else None
        rows.append({"arm":arm,"macro_f1":metrics["macro_f1"],"delta_macro_f1_vs_R0":None if arm=="R0" else delta["macro_f1"],"macro_f1_ci95_low":None if ci is None else ci["ci95_low"],"macro_f1_ci95_high":None if ci is None else ci["ci95_high"],"balanced_accuracy":metrics["balanced_accuracy"],"accuracy":metrics["accuracy"],"worst_class_f1":metrics["worst_class_f1"]})
    with (output_dir/"primary_summary.csv").open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={"format":"bilinear_lmmd.preprocessing.final_report.v1","protocol":"coffee17-preprocessing-primary-v1","sample_count":oof["sample_count"],"primary_metric":"macro_f1","arms":rows,"bootstrap_scope":bootstrap["scope"],"analysis":analysis,"efficiency":efficiency,"interpretation_rule":"Report arm-minus-R0 effects and uncertainty without selecting a winner; positive, null and negative effects are all valid outcomes."}; (output_dir/"final_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=["# Final Coffee17 Preprocessing Study Report","",f"Primary OOF population: {oof['sample_count']} clean identities","","| Arm | Macro-F1 | Δ vs R0 | Paired bootstrap 95% CI | Balanced Acc. | Worst-F1 |","|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["arm"]=="R0": delta="—"; interval="—"
        else: delta=f"{row['delta_macro_f1_vs_R0']:+.2%}"; interval=f"[{row['macro_f1_ci95_low']:+.2%}, {row['macro_f1_ci95_high']:+.2%}]"
        lines.append(f"| {row['arm']} | {row['macro_f1']:.2%} | {delta} | {interval} | {row['balanced_accuracy']:.2%} | {row['worst_class_f1']:.2%} |")
    lines += ["","## Interpretation boundary","","The paired bootstrap quantifies OOF sample uncertainty conditional on the frozen seed-42 trained models. It is not a full estimate of optimization-seed uncertainty.","","No primary arm is promoted or rejected after outer-test opening. Positive, null, and negative preprocessing effects are retained."]
    if efficiency is not None: lines += ["","## Efficiency","","See `preprocessing_efficiency.json` for pre-model, CNN-only, and end-to-end timings. Frontend execution devices are reported explicitly because C0 uses canonical CPU OpenCV while F0/W0 use tensor operators on the selected device."]
    (output_dir/"FINAL_PREPROCESSING_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print("\n".join(lines),flush=True); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--oof-summary",required=True,type=Path); p.add_argument("--bootstrap",required=True,type=Path); p.add_argument("--analysis-summary",required=True,type=Path); p.add_argument("--efficiency",type=Path); p.add_argument("--output-dir",required=True,type=Path); a=p.parse_args(); generate_report(a.oof_summary,a.bootstrap,a.analysis_summary,a.efficiency,a.output_dir)
if __name__=="__main__": main()
