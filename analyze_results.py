# -*- coding: utf-8 -*-
"""汇总并核验实验八的所有结构化结果，生成 results/summary.json 与 summary.md。

用法：
  .venv/bin/python analyze_results.py
  .venv/bin/python analyze_results.py --check   # 只跑核验，不写文件
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analytical_params(vocab, cfg):
    C, L, T = cfg["n_embd"], cfg["n_layer"], cfg["block_size"]
    return vocab * C + T * C + L * (12 * C * C + 13 * C) + 2 * C + vocab * C


def ppl(loss):
    return math.exp(loss)


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": str(detail)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    checks = []
    summary = {"artifacts": {}}

    # ---------- 必做：正式结果 ----------
    all_path = RESULTS / "formal" / "results_all.json"
    all_res = load(all_path)
    data = all_res["data"]
    teacher = all_res["teacher"]
    scratch = all_res["scratch"]
    kd = all_res["kd"]
    # 正式 all 运行复用 teacher.pt，teacher 行本身 time=0、curve=[]；
    # 从同一次 teacher 训练的 results_teacher.json 补齐真实耗时与曲线。
    tr_path = RESULTS / "formal" / "results_teacher.json"
    if tr_path.exists():
        tr = load(tr_path)["teacher"]
        if tr.get("params") == teacher.get("params"):
            teacher["time_s"] = tr.get("time_s", teacher.get("time_s", 0.0))
            teacher["curve"] = tr.get("curve", teacher.get("curve", []))
    summary["data"] = data
    summary["teacher"] = teacher
    summary["scratch"] = scratch
    summary["kd"] = kd
    summary["comparison"] = all_res["comparison"]
    summary["samples"] = all_res["samples"]
    summary["env"] = all_res["env"]

    checks.append(check("vocab_322", data["vocab"] == 322, data["vocab"]))
    checks.append(check("train_val_disjoint_poems",
                        data["val_contains_fixed_poems"] is False))
    checks.append(check("train_chars", data["train_chars"] > 50000, data["train_chars"]))
    checks.append(check("val_chars", data["val_chars"] > 4000, data["val_chars"]))
    checks.append(check("scratch_kd_same_params",
                        scratch["params"] == kd["params"], scratch["params"]))
    checks.append(check("params_5x",
                        abs(teacher["params"] / kd["params"] - 5.0) < 0.1,
                        teacher["params"] / kd["params"]))
    for name, row in (("teacher", teacher), ("scratch", scratch), ("kd", kd)):
        checks.append(check(name + "_loss_finite", math.isfinite(row["val_loss"]),
                            row["val_loss"]))
        checks.append(check(name + "_ppl_consistent",
                            abs(row["ppl"] - ppl(row["val_loss"])) < 1e-6,
                            row["ppl"]))
        pa = analytical_params(data["vocab"], row["cfg"])
        err = abs(pa - row["params"]) / row["params"]
        checks.append(check(name + "_param_formula", err < 0.01,
                            "formula=%d actual=%d err=%.4f%%" %
                            (pa, row["params"], err * 100)))
    checks.append(check("scratch_agree_range", 0 <= scratch["agree"] <= 1,
                        scratch["agree"]))
    checks.append(check("kd_agree_range", 0 <= kd["agree"] <= 1, kd["agree"]))
    checks.append(check("teacher_time_positive", teacher["time_s"] > 0,
                        teacher["time_s"]))
    checks.append(check("teacher_curve_200", len(teacher.get("curve", [])) == 200,
                        len(teacher.get("curve", []))))
    checks.append(check("kd_better_than_scratch",
                        kd["val_loss"] < scratch["val_loss"],
                        "%.4f < %.4f" % (kd["val_loss"], scratch["val_loss"])))
    rel = 100 * (scratch["val_loss"] - kd["val_loss"]) / scratch["val_loss"]
    checks.append(check("relative_drop_recomputed",
                        abs(rel - all_res["comparison"]["relative_drop_pct"]) < 1e-9,
                        rel))
    for key in ("teacher", "scratch", "kd"):
        txt = all_res["samples"]["samples"][key]
        checks.append(check("sample_len_" + key, len(txt) >= 60, len(txt)))

    meta_all = load(RESULTS / "formal" / "run_meta.json")
    checks.append(check("formal_run_success", meta_all["status"] == "success",
                        meta_all["status"]))
    checks.append(check("teacher_iters_200",
                        meta_all["config"]["teacher_iters"] == 200,
                        meta_all["config"]["teacher_iters"]))
    checks.append(check("student_iters_120",
                        meta_all["config"]["student_iters"] == 120,
                        meta_all["config"]["student_iters"]))
    checks.append(check("default_T4_alpha05",
                        meta_all["config"]["T"] == 4.0 and
                        meta_all["config"]["alpha"] == 0.5,
                        "%s/%s" % (meta_all["config"]["T"], meta_all["config"]["alpha"])))

    # ---------- 消融 ----------
    abl_res = load(RESULTS / "ablate" / "results_ablate.json")
    abl = abl_res["ablation"]
    rows = abl["rows"]
    summary["ablation"] = abl
    combos = {(r["T"], r["alpha"]) for r in rows}
    checks.append(check("ablation_16_groups", len(rows) == 16 and len(combos) == 16,
                        len(rows)))
    checks.append(check("ablation_all_finite",
                        all(math.isfinite(r["val_loss"]) for r in rows)))
    checks.append(check("ablation_best_consistent",
                        min(r["val_loss"] for r in rows) == abl["best"]["val_loss"],
                        abl["best"]))
    checks.append(check("ablation_80_iters", abl["abl_iters"] == 80, abl["abl_iters"]))
    # 热力图 / 曲线存在
    for f in ("curves.png", "heatmap.png", "ablation_curves.png"):
        p = RESULTS / "figures" / f
        checks.append(check("figure_" + f, p.exists() and p.stat().st_size > 5000,
                            p.stat().st_size if p.exists() else 0))

    # ---------- 120 步消融复测（补充实验） ----------
    abl120_path = RESULTS / "ablate120" / "results_ablate.json"
    if abl120_path.exists():
        abl120_res = load(abl120_path)
        abl120 = abl120_res["ablation"]
        summary["ablation120"] = abl120
        checks.append(check("ablation120_16_groups",
                            len(abl120["rows"]) == 16, len(abl120["rows"])))
        checks.append(check("ablation120_best_T4_alpha05",
                            abl120["best"]["T"] == 4 and abl120["best"]["alpha"] == 0.5,
                            "%s/%s" % (abl120["best"]["T"], abl120["best"]["alpha"])))

    # ---------- 最优组 120 步复测 ----------
    best_path = RESULTS / "best" / "results_all.json"
    if best_path.exists():
        best = load(best_path)
        summary["best_120"] = best
        checks.append(check("best_120_T2_alpha03",
                            best["kd"]["T"] == 2 and best["kd"]["alpha"] == 0.3,
                            "%s/%s" % (best["kd"]["T"], best["kd"]["alpha"])))
        checks.append(check("best_120_finite",
                            math.isfinite(best["kd"]["val_loss"]),
                            best["kd"]["val_loss"]))

    # ---------- 拓展 ----------
    feat_path = RESULTS / "ext_feat" / "results_feat.json"
    if feat_path.exists():
        feat = load(feat_path)
        summary["feat"] = feat
        gain = 100 * (kd["val_loss"] - feat["kd_feat"]["val_loss"]) / kd["val_loss"]
        summary["feat"]["gain_vs_kd_pct"] = gain
        checks.append(check("feat_0.1", feat["kd_feat"]["feat"] == 0.1,
                            feat["kd_feat"]["feat"]))
        checks.append(check("feat_finite",
                            math.isfinite(feat["kd_feat"]["val_loss"]),
                            feat["kd_feat"]["val_loss"]))

    quant_path = RESULTS / "ext_quant" / "results_quant.json"
    if quant_path.exists():
        q = load(quant_path)
        summary["quant"] = q
        checks.append(check("quant_size_smaller",
                            q["int8"]["file_size_bytes"] < q["fp32"]["file_size_bytes"],
                            q["size_ratio_int8_over_fp32"]))
        checks.append(check("quant_loss_close",
                            abs(q["int8"]["val_loss"] - q["fp32"]["val_loss"]) < 0.01,
                            q["int8"]["val_loss"] - q["fp32"]["val_loss"]))

    self_path = RESULTS / "ext_self" / "results_self.json"
    if self_path.exists():
        s = load(self_path)
        summary["self"] = s
        checks.append(check("self_teacher_same_arch",
                            s["teacher"]["cfg"]["n_embd"] == 64,
                            s["teacher"]["cfg"]["n_embd"]))
        checks.append(check("self_teacher_longer",
                            s["teacher"]["iters"] == 300,
                            s["teacher"]["iters"]))

    scan_path = RESULTS / "ext_scan" / "results_scan.json"
    if scan_path.exists():
        sc = load(scan_path)
        summary["scan"] = sc
        checks.append(check("scan_three_teachers", len(sc["rows"]) == 3,
                            [r["teacher_embd"] for r in sc["rows"]]))
        for r in sc["rows"]:
            checks.append(check("scan_C%d_finite" % r["teacher_embd"],
                                math.isfinite(r["kd"]["val_loss"]),
                                r["kd"]["val_loss"]))

    # ---------- 汇总表 ----------
    summary["tables"] = {
        "table6": [
            {"model": "Teacher", "params": teacher["params"],
             "iters": 200, "time_s": teacher["time_s"],
             "val_loss": teacher["val_loss"], "ppl": teacher["ppl"],
             "agree": None},
            {"model": "Scratch", "params": scratch["params"],
             "iters": 120, "time_s": scratch["time_s"],
             "val_loss": scratch["val_loss"], "ppl": scratch["ppl"],
             "agree": scratch["agree"]},
            {"model": "KD", "params": kd["params"],
             "iters": 120, "time_s": kd["time_s"],
             "val_loss": kd["val_loss"], "ppl": kd["ppl"],
             "agree": kd["agree"]},
        ],
        "table7": [{"T": r["T"], "alpha": r["alpha"], "val_loss": r["val_loss"]}
                   for r in rows],
    }

    # 控制台打印
    print("=" * 78)
    print("实验八结果核验")
    print("=" * 78)
    for c in checks:
        print("[%s] %-36s %s" % ("PASS" if c["pass"] else "FAIL", c["name"], c["detail"]))
    n_fail = sum(1 for c in checks if not c["pass"])
    print("-" * 78)
    print("共 %d 项检查，PASS %d，FAIL %d" %
          (len(checks), len(checks) - n_fail, n_fail))
    print("\n表 6：")
    for r in summary["tables"]["table6"]:
        print("  %-8s params=%d iters=%d time=%.1fs loss=%.4f ppl=%.2f agree=%s"
              % (r["model"], r["params"], r["iters"], r["time_s"],
                 r["val_loss"], r["ppl"],
                 "—" if r["agree"] is None else "%.3f" % r["agree"]))
    print("\n表 7 消融（val loss）：")
    for T in [1, 2, 4, 8]:
        vals = [next(r["val_loss"] for r in rows if r["T"] == T and abs(r["alpha"] - a) < 1e-9)
                for a in [0.3, 0.5, 0.7, 0.9]]
        print("  T=%d  " % T + "  ".join("%.4f" % v for v in vals))
    print("\n最优：T=%d alpha=%.1f val=%.4f" %
          (abl["best"]["T"], abl["best"]["alpha"], abl["best"]["val_loss"]))
    if "ablation120" in summary:
        print("80 步最优：T=%d alpha=%.1f val=%.4f；120 步复测最优：T=%d alpha=%.1f val=%.4f" %
              (abl["best"]["T"], abl["best"]["alpha"], abl["best"]["val_loss"],
               summary["ablation120"]["best"]["T"],
               summary["ablation120"]["best"]["alpha"],
               summary["ablation120"]["best"]["val_loss"]))
    if "best_120" in summary:
        print("最优 120 步复测：T=%.1f alpha=%.1f val=%.4f" %
              (summary["best_120"]["kd"]["T"], summary["best_120"]["kd"]["alpha"],
               summary["best_120"]["kd"]["val_loss"]))
    if "feat" in summary:
        print("特征蒸馏：feat=0.1 val=%.4f（相对默认 KD 再降 %.2f%%）" %
              (summary["feat"]["kd_feat"]["val_loss"],
               summary["feat"]["gain_vs_kd_pct"]))
    if "quant" in summary:
        print("动态量化：fp32 %.1f KB / int8 %.1f KB（%.2fx），loss %.4f -> %.4f" %
              (summary["quant"]["fp32"]["file_size_bytes"] / 1024,
               summary["quant"]["int8"]["file_size_bytes"] / 1024,
               summary["quant"]["size_ratio_int8_over_fp32"],
               summary["quant"]["fp32"]["val_loss"],
               summary["quant"]["int8"]["val_loss"]))
    if "scan" in summary:
        print("教师规模扫描：" + " | ".join(
            "C=%d gain=%.2f%%" % (r["teacher_embd"], r["relative_drop_pct"])
            for r in summary["scan"]["rows"]))
    print("=" * 78)

    if args.check:
        return 0 if n_fail == 0 else 1

    summary["checks"] = checks
    summary["n_fail"] = n_fail
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Markdown 版本
    lines = ["# 实验八结果汇总", "", "## 表 6 三组模型", "",
             "| 模型 | 参数量 | 训练步数 | 耗时/s | 留出集 loss | PPL | 一致率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["tables"]["table6"]:
        lines.append("| %s | %d | %d | %.1f | %.4f | %.2f | %s |" %
                     (r["model"], r["params"], r["iters"], r["time_s"],
                      r["val_loss"], r["ppl"],
                      "—" if r["agree"] is None else "%.3f" % r["agree"]))
    lines += ["", "## 表 7 T × α 消融（留出集 loss）", "",
              "| T \\ α | 0.3 | 0.5 | 0.7 | 0.9 |", "|---|---:|---:|---:|---:|"]
    for T in [1, 2, 4, 8]:
        vals = [next(r["val_loss"] for r in rows if r["T"] == T and abs(r["alpha"] - a) < 1e-9)
                for a in [0.3, 0.5, 0.7, 0.9]]
        lines.append("| %d | " % T + " | ".join("%.4f" % v for v in vals) + " |")
    lines += ["", "最优：T=%d，α=%.1f，val=%.4f；最差：T=%d，α=%.1f，val=%.4f。" %
              (abl["best"]["T"], abl["best"]["alpha"], abl["best"]["val_loss"],
               abl["worst"]["T"], abl["worst"]["alpha"], abl["worst"]["val_loss"])]
    if "ablation120" in summary:
        a120 = summary["ablation120"]
        lines += ["", "## 120 步消融复测（补充）", "",
                  "| T \\ α | 0.3 | 0.5 | 0.7 | 0.9 |", "|---|---:|---:|---:|---:|"]
        for T in [1, 2, 4, 8]:
            vals = [next(r["val_loss"] for r in a120["rows"]
                         if r["T"] == T and abs(r["alpha"] - a) < 1e-9)
                    for a in [0.3, 0.5, 0.7, 0.9]]
            lines.append("| %d | " % T + " | ".join("%.4f" % v for v in vals) + " |")
        lines += ["", "120 步最优：T=%d，α=%.1f，val=%.4f。" %
                  (a120["best"]["T"], a120["best"]["alpha"], a120["best"]["val_loss"])]
    lines += ["", "## 核验", ""]
    for c in checks:
        lines.append("- [%s] %s：%s" % ("x" if c["pass"] else " ", c["name"], c["detail"]))
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n[write] %s" % (RESULTS / "summary.json"))
    print("[write] %s" % (RESULTS / "summary.md"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
