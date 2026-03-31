#!/usr/bin/env python3
"""K-Means + PCA 2D visualization demo (pure Python, no third-party deps)."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Vec = List[float]
Mat = List[List[float]]


@dataclass
class DemoResult:
    x_raw: Mat
    x_std: Mat
    labels: List[int]
    centers_std: Mat
    z_scaled: Mat
    z_centers_scaled: Mat
    sample_idx: int
    s_i1: float
    s_i2: float


# ---------- basic linear algebra helpers ----------
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def mat_vec(m: Mat, v: Sequence[float]) -> Vec:
    return [dot(row, v) for row in m]


def transpose(m: Mat) -> Mat:
    return [list(col) for col in zip(*m)]


def mean_col(x: Mat) -> Vec:
    n = len(x)
    d = len(x[0])
    return [sum(xi[j] for xi in x) / n for j in range(d)]


def cov_3x3(x: Mat) -> Mat:
    n = len(x)
    mu = mean_col(x)
    centered = [[xi[j] - mu[j] for j in range(3)] for xi in x]
    c = [[0.0] * 3 for _ in range(3)]
    for r in centered:
        for i in range(3):
            for j in range(3):
                c[i][j] += r[i] * r[j]
    for i in range(3):
        for j in range(3):
            c[i][j] /= n
    return c


def power_iteration(a: Mat, iters: int = 200) -> Tuple[float, Vec]:
    v = [1.0, 0.5, -0.2]
    nv = norm(v)
    v = [x / nv for x in v]
    for _ in range(iters):
        w = mat_vec(a, v)
        nw = norm(w)
        if nw == 0:
            break
        v = [x / nw for x in w]
    av = mat_vec(a, v)
    lam = dot(v, av)
    return lam, v


def deflate(a: Mat, lam: float, v: Vec) -> Mat:
    out = [[a[i][j] - lam * v[i] * v[j] for j in range(3)] for i in range(3)]
    return out


# ---------- simulation & algorithms ----------
def simulate_features(n_samples: int, seed: int) -> Mat:
    random.seed(seed)
    t = [i / (n_samples - 1) for i in range(n_samples)]

    mode = [1.0 if math.sin(2 * math.pi * 3 * ti) >= 0 else -1.0 for ti in t]
    u_sub = [
        750 + 120 * mode[i] + 12 * math.sin(2 * math.pi * 4 * t[i]) + random.gauss(0, 8)
        for i in range(n_samples)
    ]

    dt = t[1] - t[0]
    du_dt = [0.0] * n_samples
    du_dt[0] = (u_sub[1] - u_sub[0]) / dt
    for i in range(1, n_samples - 1):
        du_dt[i] = (u_sub[i + 1] - u_sub[i - 1]) / (2 * dt)
    du_dt[-1] = (u_sub[-1] - u_sub[-2]) / dt

    i_sub = [
        280 * mode[i] + 20 * math.cos(2 * math.pi * 5 * t[i]) + 0.3 * du_dt[i] + random.gauss(0, 15)
        for i in range(n_samples)
    ]

    return [[u_sub[i], du_dt[i], i_sub[i]] for i in range(n_samples)]


def standardize(x: Mat) -> Tuple[Mat, Vec, Vec]:
    n = len(x)
    d = len(x[0])
    mu = [sum(x[i][j] for i in range(n)) / n for j in range(d)]
    sigma = []
    for j in range(d):
        var = sum((x[i][j] - mu[j]) ** 2 for i in range(n)) / n
        s = math.sqrt(var)
        sigma.append(s if s > 0 else 1.0)
    out = [[(x[i][j] - mu[j]) / sigma[j] for j in range(d)] for i in range(n)]
    return out, mu, sigma


def kmeans(x: Mat, k: int = 2, max_iter: int = 100, tol: float = 1e-7, seed: int = 0) -> Tuple[List[int], Mat]:
    random.seed(seed)
    n = len(x)
    d = len(x[0])

    centers = [x[random.randrange(n)][:]]
    while len(centers) < k:
        d2 = []
        for xi in x:
            best = min(sum((xi[j] - c[j]) ** 2 for j in range(d)) for c in centers)
            d2.append(best)
        s = sum(d2)
        if s == 0:
            centers.append(x[random.randrange(n)][:])
            continue
        r = random.random() * s
        acc = 0.0
        idx = 0
        for i, val in enumerate(d2):
            acc += val
            if acc >= r:
                idx = i
                break
        centers.append(x[idx][:])

    labels = [0] * n
    for _ in range(max_iter):
        for i, xi in enumerate(x):
            dist2 = [sum((xi[j] - c[j]) ** 2 for j in range(d)) for c in centers]
            labels[i] = min(range(k), key=lambda c: dist2[c])

        new_centers = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for xi, li in zip(x, labels):
            counts[li] += 1
            for j in range(d):
                new_centers[li][j] += xi[j]
        for c in range(k):
            if counts[c] == 0:
                new_centers[c] = x[random.randrange(n)][:]
            else:
                new_centers[c] = [v / counts[c] for v in new_centers[c]]

        shift = math.sqrt(sum((new_centers[c][j] - centers[c][j]) ** 2 for c in range(k) for j in range(d)))
        centers = new_centers
        if shift < tol:
            break

    return labels, centers


def pca_2d(x: Mat) -> Tuple[Mat, Mat]:
    cov = cov_3x3(x)
    lam1, v1 = power_iteration(cov)

    c2 = deflate(cov, lam1, v1)
    _, v2 = power_iteration(c2)

    # orthogonalize v2 against v1
    proj = dot(v2, v1)
    v2 = [v2[i] - proj * v1[i] for i in range(3)]
    nv2 = norm(v2)
    if nv2 == 0:
        v2 = [0.0, 1.0, 0.0]
        nv2 = 1.0
    v2 = [x / nv2 for x in v2]

    w2 = [[v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]]]

    z = []
    for xi in x:
        z1 = xi[0] * w2[0][0] + xi[1] * w2[1][0] + xi[2] * w2[2][0]
        z2 = xi[0] * w2[0][1] + xi[1] * w2[1][1] + xi[2] * w2[2][1]
        z.append([z1, z2])
    return z, w2


def project_centers(centers: Mat, w2: Mat) -> Mat:
    out = []
    for c in centers:
        z1 = c[0] * w2[0][0] + c[1] * w2[1][0] + c[2] * w2[2][0]
        z2 = c[0] * w2[0][1] + c[1] * w2[1][1] + c[2] * w2[2][1]
        out.append([z1, z2])
    return out


def minmax_scale_to_08(points: Mat) -> Mat:
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)

    dx = x_max - x_min if x_max > x_min else 1.0
    dy = y_max - y_min if y_max > y_min else 1.0

    return [[0.8 * (p[0] - x_min) / dx, 0.8 * (p[1] - y_min) / dy] for p in points]


def run_demo(n_samples: int, seed: int) -> DemoResult:
    x_raw = simulate_features(n_samples, seed)
    x_std, _, _ = standardize(x_raw)
    labels, centers_std = kmeans(x_std, k=2, seed=seed)

    z, w2 = pca_2d(x_std)
    z_centers = project_centers(centers_std, w2)

    all_pts = z + z_centers
    all_scaled = minmax_scale_to_08(all_pts)
    z_scaled = all_scaled[:-2]
    z_centers_scaled = all_scaled[-2:]

    sample_idx = len(z_scaled) // 3
    xi = z_scaled[sample_idx]
    s_i1 = math.dist(xi, z_centers_scaled[0])
    s_i2 = math.dist(xi, z_centers_scaled[1])

    return DemoResult(
        x_raw=x_raw,
        x_std=x_std,
        labels=labels,
        centers_std=centers_std,
        z_scaled=z_scaled,
        z_centers_scaled=z_centers_scaled,
        sample_idx=sample_idx,
        s_i1=s_i1,
        s_i2=s_i2,
    )


# ---------- visualization output (SVG) ----------
def svg_map(x: float, y: float, w: int, h: int, m: int) -> Tuple[float, float]:
    px = m + (x / 0.8) * (w - 2 * m)
    py = h - (m + (y / 0.8) * (h - 2 * m))
    return px, py


def render_svg(result: DemoResult, output: str) -> None:
    width, height, margin = 920, 690, 70
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

    # axes
    x0, y0 = svg_map(0.0, 0.0, width, height, margin)
    x1, _ = svg_map(0.8, 0.0, width, height, margin)
    _, y1 = svg_map(0.0, 0.8, width, height, margin)
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    # points
    for p, lb in zip(result.z_scaled, result.labels):
        px, py = svg_map(p[0], p[1], width, height, margin)
        color = colors[lb % len(colors)]
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.2" fill="{color}" fill-opacity="0.82"/>')

    # centers as star-like polygon
    for i, c in enumerate(result.z_centers_scaled):
        px, py = svg_map(c[0], c[1], width, height, margin)
        color = "#e31a1c" if i == 0 else "#ff8c00"
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="10" fill="{color}" stroke="black" stroke-width="1.2"/>')

    xi = result.z_scaled[result.sample_idx]
    xipx, xipy = svg_map(xi[0], xi[1], width, height, margin)
    lines.append(f'<circle cx="{xipx:.2f}" cy="{xipy:.2f}" r="9" fill="purple" stroke="black" stroke-width="1.2"/>')

    for idx, text in [(0, f"S_i1={result.s_i1:.3f}"), (1, f"S_i2={result.s_i2:.3f}")]:
        c = result.z_centers_scaled[idx]
        cx, cy = svg_map(c[0], c[1], width, height, margin)
        lines.append(f'<line x1="{xipx:.2f}" y1="{xipy:.2f}" x2="{cx:.2f}" y2="{cy:.2f}" stroke="#666" stroke-width="1.5" stroke-dasharray="7,5"/>')
        tx, ty = (xipx + cx) / 2, (xipy + cy) / 2
        lines.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="16" fill="#333">{text}</text>')

    lines.append('<text x="70" y="35" font-size="24" fill="#111">Simulation + K-Means + PCA (2D)</text>')
    lines.append('<text x="660" y="40" font-size="14" fill="#333">scaled range: [0, 0.8]</text>')

    lines.append('</svg>')

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure Python K-Means + PCA demo")
    parser.add_argument("--n-samples", type=int, default=400, help="number of synthetic samples")
    parser.add_argument("--seed", type=int, default=71, help="random seed")
    parser.add_argument("--output", type=str, default="fig71_demo.svg", help="output SVG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(n_samples=args.n_samples, seed=args.seed)
    render_svg(result, output=args.output)

    print("Done.")
    print(f"samples      : {args.n_samples}")
    print(f"seed         : {args.seed}")
    print(f"output image : {args.output}")
    print(f"S_i1         : {result.s_i1:.6f}")
    print(f"S_i2         : {result.s_i2:.6f}")


if __name__ == "__main__":
    main()
