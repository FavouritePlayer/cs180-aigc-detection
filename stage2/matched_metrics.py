"""Caption-cluster uncertainty for one real and G generated images per group."""
import numpy as np
from detection_metrics import auroc

def matched_summary(real,fake,replicates=2000,seed=180):
    real=np.asarray(real,dtype=float);fake=np.asarray(fake,dtype=float)
    if real.ndim!=1 or fake.ndim!=2 or fake.shape[0]!=len(real):raise ValueError('Expected N real scores and an N x G generated matrix')
    if not len(real) or not fake.shape[1] or not (np.isfinite(real).all() and np.isfinite(fake).all()):raise ValueError('Finite scores and nonempty matched groups required')
    n,g=fake.shape
    def area(r,f):return auroc(np.r_[np.zeros(len(r)),np.ones(f.size)],np.r_[r,f.ravel()])
    pooled=area(real,fake);per=[area(real,fake[:,j:j+1]) for j in range(g)]
    assert np.isclose(pooled,np.mean(per)), 'Equal-sized generator AUROCs must average to the pooled AUROC'
    rng=np.random.default_rng(seed);boot=np.empty(replicates);diff=np.empty(replicates);per_boot=np.empty((replicates,g))
    for b in range(replicates):
        idx=rng.integers(0,n,size=n);r=real[idx];f=fake[idx];boot[b]=area(r,f);diff[b]=float(np.mean(f.mean(axis=1)-r))
        for j in range(g):per_boot[b,j]=area(r,f[:,j:j+1])
    return {'auroc':pooled,'auroc_95pct_caption_cluster_bootstrap':np.percentile(boot,[2.5,97.5]).tolist(),'paired_mean_difference':float(np.mean(fake.mean(axis=1)-real)),'paired_difference_95pct_caption_cluster_bootstrap':np.percentile(diff,[2.5,97.5]).tolist(),'per_generator_auroc':per,'per_generator_auroc_ci95':np.percentile(per_boot,[2.5,97.5],axis=0).T.tolist(),'per_generator_paired_mean_difference':np.mean(fake-real[:,None],axis=0).tolist(),'per_generator_fraction_above_matched_real':np.mean(fake>real[:,None],axis=0).tolist(),'bootstrap_replicates':replicates,'bootstrap_seed':seed,'bootstrap_unit':'complete caption group, preserving real and all generated counterparts','groups':n,'generators':g}
