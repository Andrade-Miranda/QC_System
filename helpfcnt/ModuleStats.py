"""
ModuleStats — Statistical Report for PanTS Pipeline.

Reads the best available cohort CSV (cropped > selected) and produces:
  - a console summary
  - a self-contained HTML report (auto-opened in the browser)

Compatible with the output of Module 2 (selected_cohort_*.csv) and
Module 4 (cropped_dataset/cropped_cohort.csv).
Columns used: volume_tumor_cc, volume_pancreas_cc, tumor_pancreas_ratio,
tumor_attenuation, tumor_location, spacing_x/y/z, has_tumor.
"""

import os
import pathlib
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# DATA LOADING
# =============================================================================

def load_best_csv():
    """
    Tries CSV sources in order of preference — most processed first.
    Stops at the first file that actually exists.
    """
    candidates = []

    # Module 4 output (cropped + balanced)
    cropped = os.path.join("cropped_dataset", "cropped_cohort.csv")
    candidates.append((cropped, "Module 4 (cropped cohort)"))

    # Module 2 output (selected 50/50, not yet cropped)
    for f in sorted(os.listdir("."), reverse=True):
        if f.startswith("selected_cohort_") and f.endswith(".csv"):
            candidates.append((f, "Module 2 (selected cohort)"))
            break

    for path, label in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[INFO] CSV loaded from {label}: {path} ({len(df)} patients)")
            return df, label

    raise SystemExit("[FATAL] No pipeline CSV found. Run Module 2 first.")


# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(df):
    """
    Computes all statistics used in the console and HTML reports.
    Columns that are absent from the CSV are skipped gracefully so the
    module works whether or not cropping / bbox steps have been run.
    """
    s = {}

    # --- Population ---
    s["n_total"]   = len(df)
    is_sick        = df["tumor_volume_cc"].notnull() & (df["tumor_volume_cc"] > 0)
    s["n_sick"]    = int(is_sick.sum())
    s["n_healthy"] = s["n_total"] - s["n_sick"]
    s["pct_sick"]  = round(100 * s["n_sick"] / s["n_total"], 1) if s["n_total"] > 0 else 0

    df_sick = df[is_sick]

    # --- Crop reduction (only if bbox columns are present) ---
    bbox_cols = ["bbox_x_min", "bbox_x_max", "bbox_y_min",
                 "bbox_y_max", "bbox_z_min", "bbox_z_max"]
    if all(c in df.columns for c in bbox_cols):
        df["bbox_vol"] = (
            (df["bbox_x_max"] - df["bbox_x_min"]) *
            (df["bbox_y_max"] - df["bbox_y_min"]) *
            (df["bbox_z_max"] - df["bbox_z_min"])
        )
        original_vol_est     = 512 * 512 * df["bbox_z_max"].mean()
        s["crop_reduction"]  = round(100 * (1 - df["bbox_vol"].mean() / original_vol_est), 1)
    else:
        s["crop_reduction"] = None

    # --- Spacing ---
    for ax in ["x", "y", "z"]:
        col = f"spacing_{ax}"
        if col in df.columns:
            s[f"spacing_{ax}"] = {
                "mean"  : round(df[col].mean(), 3),
                "median": round(df[col].median(), 3),
                "min"   : round(df[col].min(), 3),
                "max"   : round(df[col].max(), 3),
            }

    # --- Pancreas volume ---
    if "volume_pancreas_cc" in df.columns:
        vp = df["volume_pancreas_cc"].dropna()
        s["volume_pancreas"] = {
            "mean"  : round(vp.mean(), 2),
            "median": round(vp.median(), 2),
            "min"   : round(vp.min(), 2),
            "max"   : round(vp.max(), 2),
        }

    # --- Tumor volume ---
    if "tumor_volume_cc" in df.columns:
        vt = df_sick["tumor_volume_cc"].dropna()
        vt = vt[vt > 0]
        if not vt.empty:
            s["volume_tumor"] = {
                "mean"  : round(vt.mean(), 2),
                "median": round(vt.median(), 2),
                "min"   : round(vt.min(), 2),
                "max"   : round(vt.max(), 2),
                "n"     : len(vt),
                "small" : int((vt < 2).sum()),
                "medium": int(((vt >= 2) & (vt <= 10)).sum()),
                "large" : int((vt > 10).sum()),
            }

    # --- Tumor/pancreas ratio ---
    if "tumor_pancreas_ratio" in df.columns:
        r = df_sick["tumor_pancreas_ratio"].dropna()
        r = r[r > 0]
        if not r.empty:
            s["ratio"] = {
                "mean"  : round(r.mean(), 4),
                "median": round(r.median(), 4),
                "max"   : round(r.max(), 4),
            }

    # --- Tumor attenuation (Hypo / Iso / Hyper) ---
    if "tumor_attenuation" in df.columns:
        att = df_sick["tumor_attenuation"].dropna()
        if not att.empty:
            s["attenuation"]     = att.value_counts().to_dict()
            total                = len(att)
            s["attenuation_pct"] = {
                k: round(100 * v / total, 1)
                for k, v in s["attenuation"].items()
            }

    # --- Tumor location (head / body / tail) ---
    if "tumor_location" in df.columns:
        loc = df_sick["tumor_location"].dropna()
        loc = loc[loc.isin(["head", "body", "tail"])]
        if not loc.empty:
            s["location"]     = loc.value_counts().to_dict()
            total             = len(loc)
            s["location_pct"] = {
                k: round(100 * v / total, 1)
                for k, v in s["location"].items()
            }

    # --- Curriculum levels ---
    # Combines attenuation and location to classify difficulty:
    #   Level 1 — most visible  (Hypo/Hyper, body/tail)
    #   Level 4 — most subtle   (Iso, head)
    if "tumor_attenuation" in df.columns and "tumor_location" in df.columns:
        att_s = df_sick["tumor_attenuation"].fillna("").str.capitalize()
        loc_s = df_sick["tumor_location"].fillna("").str.lower()

        niv1 = int(((att_s.isin(["Hypo", "Hyper"])) & (loc_s.isin(["body", "tail"]))).sum())
        niv2 = int(((att_s.isin(["Hypo", "Hyper"])) & (loc_s == "head")).sum())
        niv3 = int(((att_s == "Iso") & (loc_s.isin(["body", "tail"]))).sum())
        niv4 = int(((att_s == "Iso") & (loc_s == "head")).sum())
        total_class = niv1 + niv2 + niv3 + niv4

        def pct(n): return round(100 * n / total_class, 1) if total_class > 0 else 0

        s["curriculum"] = {
            "Level 1 (Hypo/Hyper, Body/Tail)": {"count": niv1, "pct": pct(niv1)},
            "Level 2 (Hypo/Hyper, Head)"     : {"count": niv2, "pct": pct(niv2)},
            "Level 3 (Iso, Body/Tail)"        : {"count": niv3, "pct": pct(niv3)},
            "Level 4 (Iso, Head)"             : {"count": niv4, "pct": pct(niv4)},
            "total_class"                     : total_class,
        }

    return s


# =============================================================================
# HTML REPORT
# =============================================================================

COULEUR_ROSE   = "#D291BC"
COULEUR_MARRON = "#704241"


def pct_bar(pct, color=COULEUR_ROSE):
    return (
        f'<div style="background:#E9ECEF;border-radius:4px;height:8px;margin:4px 0 8px;">'
        f'<div style="background:{color};width:{min(pct,100)}%;height:100%;'
        f'border-radius:4px;transition:width .6s;"></div></div>'
    )


def stat_card(title, rows, title_color=COULEUR_MARRON, bar_color=COULEUR_ROSE):
    rows_html = ""
    for label, value, pct in rows:
        bar = pct_bar(pct, bar_color) if pct is not None else ""
        rows_html += (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:13px;color:#6C757D;">'
            f'<span>{label}</span>'
            f'<span style="font-weight:600;color:#212529;">{value}</span>'
            f'</div>{bar}</div>'
        )
    return (
        f'<div style="background:white;border:1px solid #E9ECEF;border-radius:12px;'
        f'padding:20px;margin-bottom:16px;">'
        f'<h3 style="margin:0 0 14px;font-size:15px;color:{title_color};">{title}</h3>'
        f'{rows_html}</div>'
    )


def big_number(value, label, color=COULEUR_MARRON):
    return (
        f'<div style="text-align:center;background:white;border:1px solid #E9ECEF;'
        f'border-radius:12px;padding:20px;">'
        f'<div style="font-size:36px;font-weight:bold;color:{color};">{value}</div>'
        f'<div style="font-size:12px;opacity:0.7;margin-top:4px;">{label}</div>'
        f'</div>'
    )


def generate_html_report(s, source_label, output_path):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Big numbers row
    crop_str = f"{s['crop_reduction']}%" if s.get("crop_reduction") is not None else "N/A"
    bignums  = (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);'
        'gap:12px;margin-bottom:24px;">'
        + big_number(s["n_total"],                      "total patients",       COULEUR_MARRON)
        + big_number(f"{s['n_sick']} ({s['pct_sick']}%)", "sick (50/50 cohort)", COULEUR_ROSE)
        + big_number(s["n_healthy"],                    "healthy",              COULEUR_MARRON)
        + big_number(crop_str,                          "image reduction (crop)", COULEUR_ROSE)
        + '</div>'
    )

    cards = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'

    if "location" in s:
        loc_rows = [
            (k, f"{v} ({s['location_pct'].get(k, 0)}%)", s["location_pct"].get(k, 0))
            for k, v in s["location"].items()
        ]
        cards += stat_card("Tumor Location", loc_rows)

    if "attenuation" in s:
        att_rows = [
            (k, f"{v} ({s['attenuation_pct'].get(k, 0)}%)", s["attenuation_pct"].get(k, 0))
            for k, v in s["attenuation"].items()
        ]
        cards += stat_card("Tumor Attenuation", att_rows)

    if "volume_tumor" in s:
        vt = s["volume_tumor"]
        n  = vt["n"]
        cards += stat_card(
            f"Tumor Size ({n} patients)",
            [
                ("Small  (<2 cc)",   vt["small"],  round(100 * vt["small"]  / n, 1)),
                ("Medium (2–10 cc)", vt["medium"], round(100 * vt["medium"] / n, 1)),
                ("Large  (>10 cc)",  vt["large"],  round(100 * vt["large"]  / n, 1)),
            ]
        )

    spacing_rows = []
    for ax in ["x", "y", "z"]:
        if f"spacing_{ax}" in s:
            sp = s[f"spacing_{ax}"]
            spacing_rows.append((f"Mean {ax.upper()}", f"{sp['mean']} mm", None))
    if spacing_rows:
        cards += stat_card("Voxel Spacing", spacing_rows)

    if "curriculum" in s:
        curr = s["curriculum"]
        curr_rows = [
            (k, f"{v['count']} cases", v["pct"])
            for k, v in curr.items() if k != "total_class"
        ]
        cards += stat_card(f"Curriculum Levels ({curr['total_class']} tumors)", curr_rows)

    cards += '</div>'

    vol_pan = s.get("volume_pancreas", {}).get("mean", "-")
    vol_tum = s.get("volume_tumor",    {}).get("mean", "-")
    ratio   = round(s.get("ratio",     {}).get("mean", 0) * 100, 1)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>
  body {{ font-family: sans-serif; background: #F9F5F6; color: #4A2C2B;
         padding: 32px; max-width: 1100px; margin: 0 auto; }}
  h1   {{ font-size: 24px; color: #704241;
         border-bottom: 3px solid #D291BC; padding-bottom: 8px; }}
  h2   {{ font-size: 18px; margin: 24px 0 12px; color: #704241; }}
</style>
</head><body>
  <h1>PanTS — Statistical Report</h1>
  <p style="font-size:12px;opacity:0.6;margin-bottom:24px;">
    Generated {now} · Source: {source_label}
  </p>
  <h2>Overview</h2>
  {bignums}
  {cards}
  <div style="background:white;padding:20px;border-radius:12px;
              margin-top:16px;border:1px solid #E9ECEF;">
    <h3 style="margin-top:0;color:#704241;">Mean Volumes &amp; Ratio</h3>
    Pancreas: <b>{vol_pan} cc</b> &nbsp;|&nbsp;
    Tumor: <b>{vol_tum} cc</b> &nbsp;|&nbsp;
    Ratio: <b>{ratio}%</b>
  </div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {output_path}")


# =============================================================================
# CONSOLE REPORT
# =============================================================================

def print_console_report(s, source_label):
    sep = "=" * 60
    print(f"\n{sep}\n  STATISTICAL REPORT — PanTS PIPELINE\n  Source: {source_label}\n{sep}")

    print(f"\n  POPULATION")
    print(f"    Total   : {s['n_total']} patients")
    print(f"    Sick    : {s['n_sick']} ({s['pct_sick']}%)")
    print(f"    Healthy : {s['n_healthy']}")
    if s.get("crop_reduction") is not None:
        print(f"    Crop    : {s['crop_reduction']}% volume reduction")

    for ax in ["x", "y", "z"]:
        if f"spacing_{ax}" in s:
            sp = s[f"spacing_{ax}"]
            print(f"\n  SPACING {ax.upper()}  "
                  f"mean={sp['mean']} | median={sp['median']} "
                  f"| [{sp['min']} – {sp['max']}] mm")

    if "volume_pancreas" in s:
        vp = s["volume_pancreas"]
        print(f"\n  PANCREAS VOLUME")
        print(f"    Mean   : {vp['mean']} cc")
        print(f"    Median : {vp['median']} cc")
        print(f"    Range  : {vp['min']} – {vp['max']} cc")

    if "volume_tumor" in s:
        vt = s["volume_tumor"]
        print(f"\n  TUMOR VOLUME  ({vt['n']} tumors)")
        print(f"    Mean   : {vt['mean']} cc")
        print(f"    Median : {vt['median']} cc")
        print(f"    Range  : {vt['min']} – {vt['max']} cc")
        print(f"    Small (<2 cc)   : {vt['small']}")
        print(f"    Medium (2-10)   : {vt['medium']}")
        print(f"    Large (>10)     : {vt['large']}")

    if "attenuation" in s:
        print(f"\n  TUMOR ATTENUATION")
        for k, v in sorted(s["attenuation"].items(), key=lambda x: -x[1]):
            print(f"    {k:<8}: {v} ({s['attenuation_pct'].get(k, 0)}%)")

    if "location" in s:
        print(f"\n  TUMOR LOCATION")
        for k, v in sorted(s["location"].items(), key=lambda x: -x[1]):
            print(f"    {k:<8}: {v} ({s['location_pct'].get(k, 0)}%)")

    if "curriculum" in s:
        curr = s["curriculum"]
        print(f"\n  CURRICULUM LEVELS  ({curr['total_class']} classified tumors)")
        for k, v in curr.items():
            if k != "total_class":
                print(f"    {k:<35}: {v['count']} ({v['pct']}%)")

    print(f"\n{sep}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUTPUT_HTML = "stats_report.html"

    df, source_label = load_best_csv()
    s = compute_stats(df)

    print_console_report(s, source_label)
    generate_html_report(s, source_label, OUTPUT_HTML)

    url = pathlib.Path(os.path.abspath(OUTPUT_HTML)).as_uri()
    webbrowser.open(url)

    print(f"=== MODULE STATS DONE ===\n"
          f"    HTML report : {OUTPUT_HTML}\n"
          f"    Opened automatically in browser.\n")