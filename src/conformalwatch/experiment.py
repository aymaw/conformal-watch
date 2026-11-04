import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import evaluate, fit, mmd_permutation, predict, prediction_sets, synthetic, threshold
from .visual import save, scatter_svg, table


def run(out, quick=False):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_x, train_y = synthetic(2500, 11)
    model = fit(train_x, train_y, updates=30 if quick else 800)
    cal_x, cal_y = synthetic(1000, 22)
    cutoff = threshold(predict(model, cal_x), cal_y)
    records = []
    for name, shift in [("IID", 0.0), ("covariate shift", 2.5)]:
        x, y = synthetic(2000, 33, shift)
        p = predict(model, x)
        sets = prediction_sets(p, cutoff)
        records.append({"slice": name, **evaluate(p, y, sets)})
    # A fixed horizon permits Bonferroni control under the no-shift null.
    reference, _ = synthetic(128, 44)
    windows = []
    for t in range(6):
        shift = 0.0 if t < 3 else 2.5
        x, y = synthetic(128, 100 + t, shift)
        result = mmd_permutation(reference, x, 800 + t, 19 if quick else 499)
        p = predict(model, x)
        windows.append(
            {
                "window": t + 1,
                "shift": shift,
                **result,
                "alarm": result["p_value"] <= 0.05 / 6,
                **evaluate(p, y, prediction_sets(p, cutoff)),
            }
        )
    # Repeated independent calibrations quantify finite-sample behavior without choosing alpha on test.
    trials = []
    for seed in range(2 if quick else 100):
        cx, cy = synthetic(300, 1000 + seed)
        tx, ty = synthetic(500, 2000 + seed)
        q = threshold(predict(model, cx), cy)
        trials.append(
            evaluate(predict(model, tx), ty, prediction_sets(predict(model, tx), q))["coverage"]
        )
    xx, yy = np.meshgrid(np.linspace(-3, 3, 35), np.linspace(-3, 3, 25))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    gp = predict(model, grid)
    payload = {
        "quick": quick,
        "alpha": 0.1,
        "cutoff": cutoff,
        "slices": records,
        "windows": windows,
        "alarm_alpha": 0.05 / 6,
        "horizon": 6,
        "permutations": 19 if quick else 499,
        "coverage_trials": trials,
        "mean_trial_coverage": float(np.mean(trials)),
        "seconds": time.perf_counter() - start,
        "grid": grid.tolist(),
        "grid_probabilities": gp.tolist(),
        "calibration_probabilities": predict(model, cal_x).tolist(),
        "calibration_labels": cal_y.tolist(),
    }
    body = (
        '<div class="card"><label>Miscoverage α <input id="alpha" type="range" '
        'min="1" max="40" value="10"></label><p id="info"></p>'
        '<canvas id="plot" width="840" height="500" role="img" '
        'aria-label="Prediction set sizes across a feature plane"></canvas>'
        "<p>Teal: singleton · purple: multiple labels · orange: empty set. "
        "The slider recomputes the finite-sample rank on the fixed calibration split. "
        "It does not rerun the statistical experiment.</p></div>"
        + "<h2>Coverage under a fixed target of 90%</h2>"
        + table(
            [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                for r in records
            ],
            ["slice", "coverage", "mean_size", "singleton_fraction", "singleton_error"],
        )
        + "<h2>Label-free stream diagnostic</h2>"
        + table(windows, ["window", "shift", "p_value", "alarm", "coverage"])
        + "<p>RBF MMD² uses fixed bandwidth 2.0 and permutation p-values. "
        "Six planned comparisons share a 0.05 family-wise budget. "
        "An alarm does not repair coverage. The guarantee requires exchangeable calibration "
        "and test examples and is marginal, not per-example or per-class.</p>"
    )
    script = """
    const c=document.getElementById('plot'),ctx=c.getContext('2d'),slider=document.getElementById('alpha');
    function draw(){const alpha=Number(slider.value)/100;
      const s=DATA.calibration_probabilities.map((p,i)=>1-p[DATA.calibration_labels[i]]).sort((a,b)=>a-b);
      const rank=Math.ceil((s.length+1)*(1-alpha)),q=rank>s.length?Infinity:s[rank-1];
      ctx.clearRect(0,0,840,500);const counts=[0,0,0,0];
      DATA.grid.forEach(([x,y],i)=>{const n=DATA.grid_probabilities[i].filter(p=>1-p<=q).length;
        counts[n]++;ctx.fillStyle=n===1?'#66e4cc':n===0?'#ffb86c':'#bc9bff';
        ctx.fillRect((x+3)/6*822,(3-y)/6*482,24,20)});
      document.getElementById('info').textContent=`α=${alpha.toFixed(2)} · q=${q.toFixed(3)} · grid singleton fraction ${(counts[1]/DATA.grid.length).toFixed(3)}`;
    } slider.oninput=draw;draw();
    """
    save(
        out,
        payload,
        "Conformal Watch",
        "Explore set-valued predictions, abstention and a shift alarm. Watch where a 90% "
        "marginal coverage guarantee stops applying.",
        body,
        script,
    )
    image = scatter_svg(
        [("Reference", reference), ("Shifted", synthetic(128, 103, 2.5)[0])],
        "When calibration stops matching the stream",
        limit=5.0,
    )
    (out / "preview.svg").write_text(image)
    print(json.dumps({"slices": records, "mean_trial_coverage": payload["mean_trial_coverage"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/reference")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run(args.out, args.quick)
