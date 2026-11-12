"""Threshold-free image detection metrics, preserving the predeclared score direction."""
import numpy as np

def auroc(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=float)
    assert labels.shape==scores.shape and np.isfinite(scores).all()
    positive=scores[labels==1];negative=scores[labels==0]
    if not len(positive) or not len(negative):raise ValueError('Both classes required')
    differences=positive[:,None]-negative[None,:]
    return float(np.mean((differences>0)+.5*(differences==0)))

def bootstrap_auc(labels,scores,replicates=2000,seed=180):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=float);positive=scores[labels==1];negative=scores[labels==0];rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(replicates):
        a=rng.choice(positive,len(positive),replace=True);b=rng.choice(negative,len(negative),replace=True)
        estimates.append(auroc(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]))
    return np.percentile(estimates,[2.5,97.5]).tolist()
