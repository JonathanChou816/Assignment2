"""Question 3: range-based MAP localization. Run with numpy scipy matplotlib.
Uses the Gaussian MAP objective requested by the assignment; see report for
the distinction introduced by rejecting negative generated measurements.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
TRUE = np.array([0.5, 0.5])
SIGMA_NOISE = 0.3
SIGMA_PRIOR = np.array([0.25, 0.25])  # standard deviations, not variances
LEVELS = np.array([0,.5,1,2,3,4,6,8,12,18,26,38,55,80,120,180,260])

def objective(p, landmarks, ranges):
    distances = np.linalg.norm(p - landmarks, axis=1)
    return .5*np.sum(((ranges-distances)/SIGMA_NOISE)**2) + .5*np.sum((p/SIGMA_PRIOR)**2)

def solve(landmarks, ranges):
    # Dense coarse-grid seeds plus origin and anchor seeds reduce local-minimum risk.
    starts = np.array([(a,b) for a in np.linspace(-1,1,9) for b in np.linspace(-1,1,9)])
    values = np.array([objective(s,landmarks,ranges) for s in starts])
    seeds = np.vstack([starts[np.argsort(values)[:12]], [0,0], landmarks])
    results = [minimize(objective,s,args=(landmarks,ranges),method='Powell',
                        options={'xtol':1e-11,'ftol':1e-12,'maxiter':5000}) for s in seeds]
    best = min(results,key=lambda r:r.fun)
    if not best.success:
        raise RuntimeError(best.message)
    return best.x, float(best.fun)

def main():
    assert np.linalg.norm(TRUE)<1
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(2024).spawn(4)]
    axis = np.linspace(-2,2,501)
    gx,gy=np.meshgrid(axis,axis)
    records=[]; grids=[]
    for k,rng in zip(range(1,5),streams):
        theta=2*np.pi*np.arange(k)/k
        landmarks=np.column_stack([np.cos(theta),np.sin(theta)])
        true_dist=np.linalg.norm(TRUE-landmarks,axis=1)
        ranges=true_dist+rng.normal(0,SIGMA_NOISE,k)
        rejected=0
        while np.any(ranges<0):
            mask=ranges<0; rejected+=int(mask.sum())
            ranges[mask]=true_dist[mask]+rng.normal(0,SIGMA_NOISE,mask.sum())
        estimate,minimum=solve(landmarks,ranges)
        z=.5*((gx/SIGMA_PRIOR[0])**2+(gy/SIGMA_PRIOR[1])**2)
        for landmark,r in zip(landmarks,ranges):
            dist=np.hypot(gx-landmark[0],gy-landmark[1])
            z+=.5*((r-dist)/SIGMA_NOISE)**2
        # Numerical agreement with an independent dense-grid check.
        assert minimum <= float(z.min())+1e-7
        records.append(dict(K=k,landmarks=landmarks.tolist(),true_ranges=true_dist.tolist(),
                            measured_ranges=ranges.tolist(),rejected_draws=rejected,
                            map_estimate=estimate.tolist(),minimum_objective=minimum,
                            position_error=float(np.linalg.norm(estimate-TRUE)),
                            objective_at_true=float(objective(TRUE,landmarks,ranges))))
        grids.append(z)
    def draw(ax,record,z):
        contour=ax.contourf(gx,gy,z,levels=LEVELS,cmap='viridis_r',extend='max')
        lines=ax.contour(gx,gy,z,levels=[2,4,8,18,38,80],colors='white',linewidths=.5,alpha=.7)
        ax.clabel(lines,fontsize=7,inline=True,fmt='%g')
        circle=np.linspace(0,2*np.pi,361)
        ax.plot(np.cos(circle),np.sin(circle),'--',color='gray',linewidth=.9,alpha=.7)
        lm=np.array(record['landmarks']); est=np.array(record['map_estimate'])
        ax.scatter(lm[:,0],lm[:,1],marker='o',s=55,color='#f4b942',edgecolors='black',label='Landmarks',zorder=5)
        ax.scatter(*TRUE,marker='+',s=190,linewidths=2.6,color='#e83d66',label='True location',zorder=7)
        ax.scatter(*est,marker='*',s=160,color='white',edgecolors='black',label='MAP estimate',zorder=8)
        ax.scatter(0,0,marker='x',s=35,color='black',label='Prior mean',zorder=6)
        ax.set(xlim=(-2,2),ylim=(-2,2),xlabel='x',ylabel='y',
               title=f"K={record['K']} | MAP=({est[0]:.3f}, {est[1]:.3f})\nPosition error={record['position_error']:.3f}")
        ax.set_aspect('equal'); ax.legend(fontsize=7,loc='lower left',framealpha=.9)
        return contour
    fig,axes=plt.subplots(2,2,figsize=(10.5,9.5),layout='constrained')
    for ax,record,z in zip(axes.ravel(),records,grids):
        contour=draw(ax,record,z)
    fig.colorbar(contour,ax=axes.ravel().tolist(),label='Gaussian MAP negative log objective J(x,y)',shrink=.85)
    fig.suptitle('Range-based MAP localization | identical contour levels in every panel',fontsize=13)
    fig.savefig(OUT/'map_contours_all.png',dpi=190);plt.close(fig)
    for record,z in zip(records,grids):
        fig,ax=plt.subplots(figsize=(6.7,5.5),layout='constrained')
        contour=draw(ax,record,z)
        fig.colorbar(contour,ax=ax,label='J(x,y)')
        fig.savefig(OUT/f"map_contours_K{record['K']}.png",dpi=180);plt.close(fig)
    output=dict(seed=2024,true_position=TRUE.tolist(),measurement_std=SIGMA_NOISE,
                prior_std=SIGMA_PRIOR.tolist(),contour_levels=LEVELS.tolist(),cases=records)
    (OUT/'results.json').write_text(json.dumps(output,indent=2))
    np.savetxt(OUT/'summary.csv',np.array([[r['K'],*r['map_estimate'],r['position_error'],r['minimum_objective']] for r in records]),
               delimiter=',',header='K,x_MAP,y_MAP,position_error,minimum_objective',comments='')
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()
