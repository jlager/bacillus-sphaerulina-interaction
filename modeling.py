"""Fit logistic growth curves to replicate and median observations."""

import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from config import DATA, microbe_label, microbe_name, organism_names

def logistic(t, u0, rate, capacity):
    return capacity / (1 + (capacity / u0 - 1) * np.exp(-rate * t))


def fit_growth(t, values):
    """Fit distance in cm with the reference workflow's logistic model and bounds."""
    t, values = np.asarray(t, float), np.asarray(values, float)
    valid = np.isfinite(t) & np.isfinite(values)
    t, values = t[valid], values[valid]
    if len(t) < 4 or len(np.unique(t)) < 4:
        raise ValueError("Logistic fitting needs at least four distinct time points")
    tau = t - t.min()  # u0 is growth at the first observation, not inoculation.
    # Match ColonyTracker: capacity is constrained above the observed maximum.
    lower = np.array([1e-8, 1e-8, min(1.001 * values.max(), 8.5 - 1e-8)])
    upper = np.array([4.25, 2., 8.5])
    capacity = min(max(1.1 * np.percentile(values, 90), 1.01 * values.max()), 8.5)
    usable = (values > 0) & (values < capacity)
    rate = (np.polyfit(tau[usable], np.log(values[usable] / (capacity - values[usable])), 1)[0]
            if usable.sum() >= 2 else 0.1)
    start = np.clip([values[0], max(rate, 0.05), capacity], lower + 1e-10, upper - 1e-10)
    fit = least_squares(lambda p: logistic(tau, *p) - values, start,
                        bounds=(lower, upper), max_nfev=10000,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    sse = np.sum(fit.fun**2)
    total = np.sum((values - values.mean())**2)
    grid = np.linspace(t.min(), t.max(), 301)
    prediction = logistic(grid - t.min(), *fit.x)
    speed = fit.x[1] * prediction * (1 - prediction / fit.x[2])
    peak = np.argmax(speed)
    result = dict(u0_cm=fit.x[0], rate_per_day=fit.x[1], capacity_cm=fit.x[2],
                  first_dpi=t.min(), n=len(t), success=fit.success, status=fit.status,
                  message=fit.message, nfev=fit.nfev, optimality=fit.optimality,
                  rmse_cm=np.sqrt(sse / len(t)), r2=1 - sse / total if total > 1e-12 else np.nan,
                  boundary_parameters=";".join(name + ("_lower" if x - lo < hi - x else "_upper")
                      for name, x, lo, hi in zip(
                      ["u0", "rate", "capacity"], fit.x, lower, upper)
                      if min(x - lo, hi - x) < 1e-5 * (hi - lo)),
                  jacobian_condition=np.linalg.cond(fit.jac),
                  peak_dpi=grid[peak], peak_growth_cm_per_day=speed[peak])
    return result, grid, prediction


def model_growth():
    groups = defaultdict(lambda: defaultdict(list))
    with (DATA / "features.csv").open() as file:
        for row in csv.DictReader(file):
            key = tuple(row[k] for k in ("fungus", "bacterium", "microbe"))
            groups[key][row["replicate"]].append((float(row["dpi"]), float(row["distance_cm"])))
    models, curves = [], []
    for key, replicates in groups.items():
        by_day = defaultdict(list)
        for series in replicates.values():
            for day, value in series:
                if np.isfinite(value):
                    by_day[day].append(value)
        replicates["median"] = [(day, np.median(values)) for day, values in by_day.items()]
        for replicate, series in replicates.items():
            t, values = np.array(sorted(series)).T
            result, grid, prediction = fit_growth(t, values)
            metadata = dict(zip(("fungus", "bacterium", "microbe"), key), replicate=replicate)
            metadata.update(organism_names(key[0], key[1]),
                            microbe_name=microbe_name(key[2]), microbe_label=microbe_label(key[2]))
            models.append({**metadata, **result})
            curves.extend({**metadata, "dpi": day, "distance_cm": value}
                          for day, value in zip(grid, prediction))
    for name, rows in (("modeling.csv", models), ("model_curves.csv", curves)):
        with (DATA / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Fitted {len(models)} logistic models; {sum(not m['success'] for m in models)} failed.")


if __name__ == "__main__":
    model_growth()
