from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
SEED = 2024
MEANS = np.array([[[-.9, -1.1], [.8, .75]], [[.1, .9], [.9, -.75]]])
COV = np.diag([.75, 1.25])
TAU = np.log(.6 / .4)

def sample(n, rng):
    y = (rng.random(n) >= .6).astype(int)
    component = rng.integers(0, 2, n)
    x = MEANS[y, component] + rng.multivariate_normal([0, 0], COV, n)
    return x, y

def log_pdf(x, label):
    terms = [multivariate_normal.logpdf(x, mean=m, cov=COV) for m in MEANS[label]]
    return logsumexp(np.column_stack(terms), axis=1) - np.log(2)

def bayes_score(x):
    return log_pdf(x, 1) - log_pdf(x, 0)

def features(x, quadratic=False):
    a, b = x.T
    columns = [np.ones(len(x)), a, b]
    if quadratic:
        columns += [a*a, a*b, b*b]
    return np.column_stack(columns)

def fit(x, y, quadratic):
    phi = features(x, quadratic)
    def objective(w):
        z = phi @ w
        loss = np.mean(np.logaddexp(0, z) - y*z)
        grad = phi.T @ (expit(z) - y) / len(y)
        return loss, grad
    result = minimize(objective, np.zeros(phi.shape[1]), jac=True,
                      method='BFGS', options={'gtol': 1e-9, 'maxiter': 10000})
    grad_norm = float(np.max(np.abs(objective(result.x)[1])))
    if grad_norm > 1e-6:
        raise RuntimeError(f'Optimization did not converge: {result.message}')
    return result.x, {'success': bool(result.success), 'message': str(result.message),
                      'gradient_max_abs': grad_norm, 'mean_nll': float(result.fun)}

def evaluate(y, pred):
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, errors=fp+fn,
                error_rate=(fp+fn)/len(y), FPR=fp/(tn+fp), TPR=tp/(tp+fn))

def roc(y, scores):
    order = np.argsort(-scores, kind='stable')
    ordered_scores, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(y)-1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, np.cumsum(1-labels)[ends]]
    return fp/np.sum(y == 0), tp/np.sum(y == 1)

def plot_boundary(ax, x, y, grid_score, gx, gy, title):
    ax.contourf(gx, gy, (grid_score >= 0).reshape(gx.shape),
                levels=[-.5, .5, 1.5], colors=['#d9e9fc', '#ffe7cd'], alpha=.75)
    ax.contour(gx, gy, grid_score.reshape(gx.shape), levels=[0], colors=['#222222'], linewidths=1.4)
    for label, color in [(0, '#2563ad'), (1, '#dc7a18')]:
        ax.scatter(*x[y == label].T, s=8, alpha=.35, color=color, label=f'L={label}', rasterized=True)
    ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), xlabel='$x_1$', ylabel='$x_2$', title=title)

def main():
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
    trains = {n: sample(n, streams[i]) for i, n in enumerate([20, 200, 2000])}
    xv, yv = sample(10000, streams[3])
    arrays = {'validation_x': xv, 'validation_y': yv}
    for n, (x, y) in trains.items():
        arrays.update({f'train_{n}_x': x, f'train_{n}_y': y})
    np.savez_compressed(OUT/'datasets.npz', **arrays)
    score = bayes_score(xv)
    base = evaluate(yv, score >= TAU)
    rows = [dict(model='Bayes', n_train=0, **base)]
    fpr, tpr = roc(yv, score)
    fig, ax = plt.subplots(figsize=(6.6, 5.2), layout='constrained')
    ax.plot(fpr, tpr, label='True-pdf likelihood-ratio ROC', color='#2563ad')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.scatter(base['FPR'], base['TPR'], s=150, marker='*', c='#dc5218', zorder=5,
               label=f"Bayes threshold = ln(1.5)\nError = {base['error_rate']:.2%}")
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Part 1: Bayes ROC', xlim=(0,1), ylim=(0,1))
    ax.legend(loc='lower right'); ax.grid(alpha=.2)
    fig.savefig(OUT/'bayes_roc.png', dpi=180); plt.close(fig)
    gx, gy = np.meshgrid(np.linspace(-4.5,4.5,240), np.linspace(-4.5,4.5,240))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    fig, ax = plt.subplots(figsize=(6,5.5), layout='constrained')
    plot_boundary(ax, xv, yv, bayes_score(grid)-TAU, gx, gy, 'Bayes boundary / validation data')
    ax.legend(); fig.savefig(OUT/'bayes_boundary.png', dpi=180); plt.close(fig)
    diagnostics = {}
    for quadratic, name in [(False, 'Linear'), (True, 'Quadratic')]:
        fig, axes = plt.subplots(2,3,figsize=(14,8), layout='constrained')
        for col, (n, (x, y)) in enumerate(trains.items()):
            w, diag = fit(x,y,quadratic)
            pred = features(xv, quadratic) @ w >= 0
            result = evaluate(yv,pred)
            rows.append(dict(model=name, n_train=n, **result))
            diagnostics[f'{name}_{n}'] = dict(weights=w.tolist(), **diag)
            gs = features(grid, quadratic) @ w
            for row, (xp, yp, label) in enumerate([(x,y,'training'),(xv,yv,'validation')]):
                plot_boundary(axes[row,col], xp, yp, gs, gx, gy,
                              f'{name}, N={n}: {label}\nValidation error = {result["error_rate"]:.2%}')
        axes[0,0].legend(loc='upper left')
        fig.savefig(OUT/f'{name.lower()}_boundaries.png', dpi=160); plt.close(fig)
    output = dict(seed=SEED, threshold=TAU, validation_counts=np.bincount(yv).tolist(),
                  results=rows, optimization=diagnostics)
    (OUT/'results.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
