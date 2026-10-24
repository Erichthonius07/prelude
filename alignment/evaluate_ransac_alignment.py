"""
evaluate_ransac_alignment.py -- evaluate alignment/ransac_alignment.py (the script, unmodified).

That script has no functions: it reads ./burst/ with the reference hard-coded as frame_11.jpeg,
prints inlier count + confidence per frame, and discards the homography. So this evaluator

  1. builds, per synthetic burst, a temp folder  burst/  with the chosen reference saved
     (copied byte-for-byte, no re-encoding) as frame_11.jpeg and the other frames as frame_NN.png
     (cv2.imread decodes by content, so a PNG named .jpeg loads fine)
  2. RUNS THE REAL SCRIPT in that folder and parses its stdout
  3. recomputes the homography with the SAME settings (ORB 2000, crossCheck, RANSAC 5.0 px,
     2000 iters, 0.995) to get H, and checks it reproduces the script's printed inliers/confidence
     EXACTLY. If it does not, the correctness numbers are void and the script says so.
  4. scores H against the TRUE motion stored in manifest.json (synth_bursts.py)

Reports: equivalence check, failure breakdown, success rate at 0.5/1/2 px (95% Wilson CI),
whether the printed confidence tracks correctness, results by noise level and by motion size,
run-to-run repeatability, runtime.

  python evaluate_ransac_alignment.py --bursts data/synthetic_bursts_clean ^
         --script alignment/ransac_alignment.py [--limit 60] [--out ransac_eval]
"""
import argparse, csv, json, math, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import cv2
import numpy as np

EXT = (".png", ".jpg", ".jpeg")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def replica(ref_g, cur_g):
    """Same estimation as ransac_alignment.py. Returns (status, H, n_matches, inliers, confidence)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kr, dr = orb.detectAndCompute(clahe.apply(ref_g), None)
    kc, dc = orb.detectAndCompute(clahe.apply(cur_g), None)
    if dr is None or dc is None or len(kc) < 4:
        return "no_keypoints", None, 0, 0, 0.0
    ms = sorted(bf.match(dr, dc), key=lambda m: m.distance)
    if len(ms) < 4:
        return "lt4_matches", None, len(ms), 0, 0.0
    src = np.float32([kr[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
    dst = np.float32([kc[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.995)
    if H is None or mask is None:
        return "ransac_failed", None, len(ms), 0, 0.0
    k = int(np.sum(mask.ravel() > 0))
    conf = float(k / len(ms))
    return "ok", H, len(ms), k, conf


def parse_stdout(text):
    """-> {filename: dict(status, matches, inliers, confidence)} from the script's prints."""
    out, cur = {}, None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^(frame_\w+\.(?:jpeg|jpg|png))$", s)
        if m:
            cur = out.setdefault(m.group(1), dict(status="unknown", matches=None, inliers=None, confidence=None)); continue
        m = re.match(r"^(frame_\w+\.(?:jpeg|jpg|png)): Could not load image", s)
        if m:
            out[m.group(1)] = dict(status="load_failed", matches=None, inliers=None, confidence=0.0); cur = None; continue
        if cur is None:
            continue
        if s.startswith("Total matches:"):
            cur["matches"] = int(s.split(":")[1])
        elif s.startswith("Not enough keypoints"):
            cur["status"] = "no_keypoints"; cur["confidence"] = 0.0
        elif "fewer than 4 matches" in s:
            cur["status"] = "lt4_matches"; cur["confidence"] = 0.0
        elif s.startswith("RANSAC: FAILED"):
            cur["status"] = "ransac_failed"; cur["confidence"] = 0.0
        elif s.startswith("RANSAC inliers:"):
            cur["inliers"] = int(s.split(":")[1]); cur["status"] = "ok"
        elif s.startswith("RANSAC confidence:") and "(%)" not in s:
            cur["confidence"] = float(s.split(":")[1])
    return out


def corner_err(M, Mt, size):
    P = np.array([[.1, .1, 1], [.88, .12, 1], [.12, .88, 1], [.88, .88, 1.0]]).T * np.array([[size], [size], [1]])
    a = M @ P; a = a[:2] / a[2]; b = Mt @ P; b = b[:2] / b[2]
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True, type=Path)
    ap.add_argument("--script", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--split", type=str, choices=["all", "train", "val"], default="all")
    ap.add_argument("--out", type=Path, default=Path("ransac_eval"))
    a = ap.parse_args()
    man = json.loads((a.bursts / "manifest.json").read_text())
    margin, size = man["margin"], man["out"]
    T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1.0]]); TI = np.linalg.inv(T)
    items = list(man["bursts"].items())
    if a.split != "all":
        items = [(bid, b) for bid, b in items if b.get("split") == a.split]
    if a.limit < len(items):
        items = items[::max(1, len(items) // a.limit)][:a.limit]

    rows, crashes, script_secs, replica_secs, equiv_bad, equiv_n, repeat_diff = [], [], 0.0, 0.0, 0, 0, 0
    for bid, b in items:
        bdir = a.bursts / b["split"] / bid
        files = sorted(f for f in os.listdir(bdir) if f.lower().endswith(EXT))
        ims = [cv2.imread(str(bdir / f)) for f in files]
        grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in ims]
        ref = int(np.argmax([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))   # sharpest, as the pipeline does
        Hs = [np.array(h) for h in b["homographies"]]
        with tempfile.TemporaryDirectory() as tmp:
            bd = Path(tmp) / "burst"; bd.mkdir()
            names = {}
            for j, f in enumerate(files):
                nm = "frame_11.jpeg" if j == ref else f"frame_{j:02d}.png"
                shutil.copyfile(bdir / f, bd / nm); names[nm] = j
            t0 = time.time()
            proc = subprocess.run([sys.executable, str(a.script.resolve())], cwd=tmp, capture_output=True, text=True)
            script_secs += time.time() - t0
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            kp_ref = len(cv2.ORB_create(nfeatures=2000).detect(grays[ref], None))
            crashes.append(dict(burst=bid, reason=tail, ref_keypoints=kp_ref))
            if len(crashes) == 1:
                print(f"script crashed on {bid}: {tail}  (ORB keypoints in the reference frame: {kp_ref})")
            for nm, j in names.items():                         # a crash fails every frame in the burst
                if j == ref:
                    continue
                Mt = TI @ Hs[ref] @ np.linalg.inv(Hs[j]) @ T
                rows.append(dict(burst=bid, frame=j, status="script_crash", matches=None, inliers=None, confidence=0.0,
                                 err_px=None, motion_px=corner_err(np.eye(3), Mt, size),
                                 psnr=float(np.mean(b["frame_psnr"])), equivalent=None))
            continue
        printed = parse_stdout(proc.stdout)
        for nm, j in names.items():
            if j == ref:
                continue
            p = printed.get(nm, dict(status="missing", matches=None, inliers=None, confidence=None))
            t1 = time.time(); st, H, nm_, k, conf = replica(grays[ref], grays[j]); replica_secs += time.time() - t1
            st2, H2, _, k2, _ = replica(grays[ref], grays[j])                    # repeat: is RANSAC deterministic?
            if (H is None) != (H2 is None) or (H is not None and not np.allclose(H, H2)):
                repeat_diff += 1
            same = (p["status"] == st) and (st != "ok" or (p["inliers"] == k and abs((p["confidence"] or 0) - round(conf, 4)) < 1.5e-4))
            equiv_n += 1; equiv_bad += (not same)
            err = None
            if H is not None:
                Mt = TI @ Hs[ref] @ np.linalg.inv(Hs[j]) @ T
                err = corner_err(np.linalg.inv(H), Mt, size)
                disp = corner_err(np.eye(3), Mt, size)
            else:
                Mt = TI @ Hs[ref] @ np.linalg.inv(Hs[j]) @ T; disp = corner_err(np.eye(3), Mt, size)
            rows.append(dict(burst=bid, frame=j, status=p["status"], matches=p["matches"], inliers=p["inliers"],
                             confidence=p["confidence"] if p["confidence"] is not None else 0.0, err_px=err, motion_px=disp,
                             psnr=float(np.mean(b["frame_psnr"])), equivalent=same))
    if not rows:
        raise SystemExit("no frames evaluated (did the script run? try it by hand in a folder named burst/)")

    n = len(rows)
    print(f"{len(items)} bursts, {n} frame pairs (reference = sharpest frame, saved as frame_11.jpeg)\n")
    print(f"EQUIVALENCE (script's printed inliers/confidence vs my recomputation): {equiv_n - equiv_bad}/{equiv_n} identical")
    if equiv_bad:
        print("  !! MISMATCH: my recomputation is not faithful for these frames; correctness numbers below are NOT valid.")
    print(f"REPEATABILITY: identical inputs run twice gave different homographies in {repeat_diff}/{n} frames "
          f"({'RANSAC is deterministic here' if repeat_diff == 0 else 'RANSAC varies run to run'})")
    if crashes:
        kps = sorted({c["ref_keypoints"] for c in crashes})
        print(f"\nSCRIPT CRASHES: {len(crashes)} of {len(items)} bursts ({len(crashes) / len(items):.0%}) stopped with an uncaught error; "
              f"every frame in them counts as a failure below.")
        print(f"  cause: {sorted({c['reason'] for c in crashes})}")
        print(f"  ORB keypoints found in those reference frames: {kps}   (0 = the frame has no detectable features)")
    print("\nFAILURE BREAKDOWN (as printed by the script):")
    for st in ("ok", "script_crash", "no_keypoints", "lt4_matches", "ransac_failed", "load_failed", "missing"):
        c = sum(r["status"] == st for r in rows)
        if c:
            print(f"  {st:14s} {c:4d}  ({c / n:.0%})")
    err = np.array([r["err_px"] if r["err_px"] is not None else np.inf for r in rows]); conf = np.array([r["confidence"] for r in rows])
    print("\nSUCCESS RATE (frames whose warp lands within X px of the truth; failures count as failures)")
    for px in (0.5, 1.0, 2.0):
        k = int((err < px).sum()); lo, hi = wilson(k, n)
        print(f"  < {px:3.1f} px: {k / n:5.1%}   (95% CI {lo:.1%} - {hi:.1%})")
    ran = np.array([r["status"] != "script_crash" for r in rows])
    if crashes and ran.any():
        print(f"  (only bursts where the script ran: < 1.0 px = {(err[ran] < 1.0).mean():.1%} of {ran.sum()} frames)")
    ok = np.isfinite(err)
    print(f"  median error among frames that got a homography: {np.median(err[ok]):.2f} px   "
          f"(doing nothing would be {np.median([r['motion_px'] for r in rows]):.2f} px)")
    print("\nDOES THE PRINTED CONFIDENCE TRACK CORRECTNESS?  (share within 1 px, by confidence bin)")
    good = err < 1.0
    for lo_, hi_ in ((0, .3), (.3, .5), (.5, .75), (.75, 1.01)):
        m = (conf >= lo_) & (conf < hi_)
        if m.any():
            print(f"  confidence {lo_:.2f}-{min(hi_, 1):.2f}: n={m.sum():4d}  within 1px={good[m].mean():5.0%}")
    if good.std() > 0 and conf.std() > 0:
        print(f"  correlation(confidence, within-1px): {np.corrcoef(conf, good.astype(float))[0, 1]:+.2f}")
    psn = np.array([r["psnr"] for r in rows]); mot = np.array([r["motion_px"] for r in rows])
    print("\nBY NOISE LEVEL (mean frame PSNR of the burst):  within 1 px")
    for lo_, hi_, lab in ((0, 24, "noisy  < 24 dB"), (24, 28, "medium 24-28 dB"), (28, 99, "clean  > 28 dB")):
        m = (psn >= lo_) & (psn < hi_)
        if m.any():
            print(f"  {lab:16s} n={m.sum():4d}  {good[m].mean():5.0%}")
    print("BY HOW FAR THE FRAME IS FROM THE REFERENCE (true corner displacement):  within 1 px")
    for lo_, hi_ in ((0, 6), (6, 12), (12, 999)):
        m = (mot >= lo_) & (mot < hi_)
        if m.any():
            print(f"  {lo_:3d}-{hi_ if hi_ < 999 else '+':>3} px     n={m.sum():4d}  {good[m].mean():5.0%}")
    print(f"\nRUNTIME: replica estimation {1000 * replica_secs / (2 * n):.0f} ms per pair "
          f"(ORB+match+RANSAC, this PC); whole script {script_secs / len(items):.2f} s per burst incl. python start-up")

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "ransac_eval_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out / "ransac_eval.json").write_text(json.dumps(dict(
        pairs=n, equivalent=equiv_n - equiv_bad, equivalence_total=equiv_n, repeat_diff=repeat_diff,
        within_0p5=float((err < .5).mean()), within_1=float((err < 1).mean()), within_2=float((err < 2).mean())), indent=2))
    print(f"\nwrote {a.out / 'ransac_eval.json'} and ransac_eval_frames.csv")


if __name__ == "__main__":
    main()