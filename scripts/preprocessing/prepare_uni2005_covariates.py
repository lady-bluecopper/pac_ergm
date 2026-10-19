"""
prepare_uni2005_covariates.py — build the University 2005 edge covariates.

Reads the original Sailer & McCulloh (2012) data (the authors' zip) and writes the
four 67x67 matrices used by the published ERGM in Clark & Handcock (2022), Table 2:

    use     usefulness of j as perceived by i   (raw 7/8/9 recoded to 1/2/3)
    team    same team,  0/1
    floor   same floor, 0/1
    metric  shortest walking distance, in meters

The raw data are proprietary and are NOT stored in this repository; the four derived
matrices are (data/office_sailer/uni2005_cov/), and this script regenerates them from
the authors' zip.

    python scripts/prepare_uni2005_covariates.py --zip "C:/path/Data_SailerMcCulloh2012.zip"
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_ZIP_DIR = "Data/01_Uni_2005/01_"

# The affiliation matrices cover 69 people, the network only 67. These two rows/columns
# are the ones the network does not contain. Row 67 is identified with confidence; the
# second is the best fit among several near-equivalent candidates -> to be confirmed
# with the authors. Changing it moves the team coefficient by < 0.1.
DROP = (66, 67)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(Path.home() / "Downloads" / "Data_SailerMcCulloh2012.zip"))
    ap.add_argument("--out", default=str(ROOT / "data" / "office_sailer" / "uni2005_cov"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    rd = lambda name: pd.read_csv(z.open(f"{IN_ZIP_DIR}{name}.csv"), header=None).values.astype(float)

    # --- the observed network: ties are stored as 5 ("daily"), plus a few self-loops ---
    A = (rd("soc_freqdaily") > 0).astype(int)
    np.fill_diagonal(A, 0)
    n = A.shape[0]

    # --- the four covariates -------------------------------------------------
    use = rd("soc_use")
    use = np.where(use > 0, use - 6, 0.0)        # 7/8/9 -> 1/2/3 (see README)
    keep = [k for k in range(69) if k not in DROP]
    team = rd("affi_sameteam")[np.ix_(keep, keep)]
    floor = rd("affi_samefloor")[np.ix_(keep, keep)]
    metric = rd("dist_metric")
    cov = {"use": use, "team": team, "floor": floor, "metric": metric}
    for M in cov.values():
        np.fill_diagonal(M, 0.0)

    # --- checks --------------------------------------------------------------
    print(f"network: {n} nodes, {int(A.sum())} directed edges")
    ref = pd.read_csv(ROOT / "data" / "office_sailer" / "office_uni2005_edges.csv")
    A_ref = np.zeros_like(A)
    A_ref[ref["from"].values, ref["to"].values] = 1
    print("  identical to office_uni2005_edges.csv:", np.array_equal(A, A_ref))

    print(f"\n{'covariate':8s} {'shape':>10s} {'min':>8s} {'max':>8s}   sum over the observed edges")
    for name, M in cov.items():
        assert M.shape == (n, n), f"{name}: shape {M.shape}, expected ({n},{n})"
        print(f"  {name:6s} {str(M.shape):>10s} {M.min():8.2f} {M.max():8.2f}   {float((A * M).sum()):12.2f}")
        pd.DataFrame(M).to_csv(out / f"{name}.csv", header=False, index=False)

    print(f"\nwritten to {out}")
    print("published ERGM (Clark & Handcock, Table 2, University 2005):")
    print("  edges -3.4 | reciprocity 0.38 | gwesp(0.5) 1.36 | usefulness 0.7")
    print("  team 0.78  | floor 0.15       | metric -0.04")


if __name__ == "__main__":
    main()