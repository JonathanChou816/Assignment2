"""Run python question2_solution.py. Dependencies: numpy scipy matplotlib.
Python translation of the supplied hw2q2 generator (same distribution,
not the same pseudorandom samples as MATLAB)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

def generate(n, rng):
    priors = np.array([.3,.4,.3])
    means = np.array([[-10,0,10],[0,0,0],[10,0,-10]])
    outer = np.array([[1,0,-3],[0,1,0],[-3,0,15]])
    covs = [outer, np.diag([8,.5,.5]), outer]
    labels = np.searchsorted(np.cumsum(priors), rng.random(n))
    data = np.empty((n,3))
    for k in range(3):
        mask = labels == k
        data[mask] = rng.multivariate_normal(means[k],covs[k],mask.sum())
    return data[:,:2], data[:,2]

def features(x):
    a,b = x.T
    return np.column_stack([np.ones(len(x)),a,b,a*a,a*b,b*b,
                            a**3,a*a*b,a*b*b,b**3])

def main():
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2024).spawn(2)]
    x,y = generate(100,streams[0]); xv,yv = generate(1000,streams[1])
    A,B = features(x), features(xv)
    wml = np.linalg.lstsq(A,y,rcond=None)[0]
    ml = float(np.mean((yv-B@wml)**2))
    gamma = np.logspace(-8,8,321)
    sigma2 = 1.0  # working model assumption, NOT specified by the generator
    weights=[]; errors=[]
    for g in gamma:
        lam = sigma2/g
        aug = np.vstack([A, np.sqrt(lam)*np.eye(10)])
        w = np.linalg.lstsq(aug,np.r_[y,np.zeros(10)],rcond=None)[0]
        weights.append(w); errors.append(np.mean((yv-B@w)**2))
    weights=np.array(weights); errors=np.array(errors)
    i=int(np.argmin(errors)); wbest=weights[i]
    result = dict(seed=2024,n_train=100,n_validate=1000,working_sigma2=sigma2,
                  feature_order=['1','x1','x2','x1^2','x1*x2','x2^2','x1^3','x1^2*x2','x1*x2^2','x2^3'],
                  rank=int(np.linalg.matrix_rank(A)), condition=float(np.linalg.cond(A)),
                  ml_validation_mse=ml, best_map_validation_mse=float(errors[i]),
                  best_gamma=float(gamma[i]), best_lambda=float(sigma2/gamma[i]),
                  reduction_percent=float(100*(ml-errors[i])/ml),
                  ml_train_mse=float(np.mean((y-A@wml)**2)),
                  map_train_mse=float(np.mean((y-A@wbest)**2)),
                  smallest_gamma_mse=float(errors[0]),largest_gamma_mse=float(errors[-1]),
                  ml_weights=wml.tolist(),best_map_weights=wbest.tolist())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    np.savetxt(OUT/'gamma_sweep.csv',np.c_[gamma,sigma2/gamma,errors],delimiter=',',
               header='gamma,lambda,validation_mse',comments='')
    savemat(OUT/'experiment.mat',dict(xTrain=x.T,yTrain=y[None,:],xValidate=xv.T,
                                   yValidate=yv[None,:],wML=wml,wMAP=wbest,gamma=gamma,
                                   mapMSE=errors,sigma2=sigma2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax in axes:
        ax.semilogx(gamma,errors,label='MAP',color='#2563ad')
        ax.axhline(ml,color='#cf573b',ls='--',label=f'ML: {ml:.4f}')
        ax.scatter(gamma[i],errors[i],marker='*',s=130,color='black',zorder=5,
                   label=f'Best tested MAP: {errors[i]:.4f}')
        ax.set(xlabel='Prior variance gamma (working sigma squared = 1)',ylabel='Validation MSE')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Full search'); axes[1].set_title('Detail near minimum')
    axes[1].set_ylim(max(0,errors[i]*.9),ml*1.15)
    fig.savefig(OUT/'validation_mse.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,w,name in zip(axes,[wml,wbest],['ML','Best tested MAP']):
        pred=B@w
        ax.scatter(yv,pred,s=9,alpha=.4,color='#2563ad')
        limits=[min(yv.min(),pred.min()),max(yv.max(),pred.max())]
        ax.plot(limits,limits,'--',color='#cf573b')
        ax.set(xlabel='Observed validation y',ylabel='Predicted y',title=name)
        ax.grid(alpha=.2)
    fig.savefig(OUT/'predictions.png',dpi=180);plt.close(fig)
    assert np.linalg.matrix_rank(A)==10
    assert np.all(np.isfinite(errors))
    assert abs(errors[-1]-ml)<1e-5
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
