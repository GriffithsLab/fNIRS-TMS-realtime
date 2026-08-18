"""Build the source notebook for the real-time fNIRS phase-validity audit."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fnirs_realtime_trigger_validity_audit.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Validity audit: can the recorded fNIRS signal support real-time phase-triggered iTBS?

**Question.** Does the pipeline identify an intrinsic, non-artifactual oscillation near the 10 s iTBS interval, and can its phase be forecast causally enough to trigger peaks or troughs?

**Executive conclusion (criteria stated before results).** The approach is *algorithmically capable of producing stable-looking phase calls*, but the current recordings and analyses do **not validate the stronger physiological claim**. A narrow bandpass necessarily turns broadband/colored fluctuations into an apparently oscillatory waveform, and the predictor constrains every instantaneous period to 7–17 s. Consequently, period histograms near 10 s, cardiac attenuation, and visually plausible filtered extrema cannot establish an intrinsic oscillator. The analysis below asks harder questions: Is there an unconstrained spectral excess? Is it sustained? Is it spatially compatible with cortical activity rather than global/systemic physiology? Do real-time forecasts agree with a later, offline reference? Does the same method look convincing on a non-oscillatory matched null?

This is a technical validity audit, not a clinical safety validation. No hardware is contacted.
""")

code(r"""
from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import welch, butter, sosfiltfilt, find_peaks, coherence

ROOT = Path.cwd().resolve()
if ROOT.name == 'reports': ROOT = ROOT.parent
REALTIME = ROOT / 'realtime'
DATA = ROOT.parent / 'data' / 'collegestreet'
sys.path.insert(0, str(REALTIME))
import mne
mne.set_log_level('ERROR')
from activity_core import (BASELINE_SEC, ChannelPair, EXTINCTION_CSV, ExtinctionTable,
    PhasePredictor, DEFAULT_PERIOD_LO_SEC, DEFAULT_PERIOD_HI_SEC)
from stream_snirf_lsl import parse_channel_name

mpl.rcParams.update({'figure.dpi':120, 'font.size':10, 'axes.grid':True,
 'grid.alpha':.25, 'axes.spines.top':False, 'axes.spines.right':False,
 'figure.facecolor':'white', 'axes.facecolor':'white'})
TEAL='#087f8c'; RED='#d1495b'; BLUE='#2878b5'; GOLD='#d99000'; GREY='#687684'; PURPLE='#7554a3'
FILES=sorted(DATA.glob('*.snirf'))
print(f'Repository: {ROOT}\nRecordings found: {len(FILES)}')
""")

md(r"""
## 1. Reproduce the actual trigger signal

The live code takes an 8 s intensity baseline, applies its own modified Beer–Lambert conversion, and uses `activity_val = -mean(HbO)` across all long-separation source–detector pairs. The minus sign changes which label is called “peak” versus “trough” but has no physiological justification documented in the pipeline. There is no short-channel regression, motion correction, scalp/systemic regression, or signal-quality rejection before averaging.
""")

code(r"""
def load_session(fp):
    raw=mne.io.read_raw_snirf(str(fp), preload=True, verbose=False)
    x=raw.get_data().astype(float).T; fs=float(raw.info['sfreq']); pos=raw.get_montage().get_positions()['ch_pos']
    by_pair={}; pp={}
    for i,n in enumerate(raw.ch_names):
        s,d,w=parse_channel_name(n); by_pair.setdefault((s,d),{})[float(w)]=i
        sp,dp=np.asarray(pos[s]),np.asarray(pos[d]); pp[(s,d)]=(sp,dp,float(np.linalg.norm(sp-dp)))
    ext=ExtinctionTable(EXTINCTION_CSV); pairs=[]
    for (s,d), idx in by_pair.items():
        sp,dp,dist=pp[(s,d)]; p=ChannelPair(s,d,idx,sp,dp,dist); p.set_extinction(ext); pairs.append(p)
    nb=int(round(BASELINE_SEC*fs))
    if len(x)<=nb+1: return None
    for p in pairs:
        for row in x[:nb]: p.add_baseline_sample(row)
        p.finalize_baseline()
    hbo=np.empty((len(x)-nb,len(pairs))); hbr=np.empty_like(hbo)
    for k,row in enumerate(x[nb:]):
        for j,p in enumerate(pairs): hbo[k,j],hbr[k,j]=p.update(row)
    t=np.arange(nb,len(x))/fs
    return dict(file=fp.name, fs=fs, t=t, hbo=hbo, hbr=hbr, v=-hbo.mean(1),
                distances=np.array([p.distance_m for p in pairs]))

fp=max(FILES, key=lambda p:p.stat().st_size)
S=load_session(fp); fs,t,v=S['fs'],S['t'],S['v']
print(f"Representative longest session: {fp.name}; {t[-1]/60:.1f} min, {fs:.2f} Hz, {S['hbo'].shape[1]} pairs")
print(f"Pair separations: {100*S['distances'].min():.1f}–{100*S['distances'].max():.1f} cm; short channels (<15 mm): {(S['distances']<.015).sum()}")
""")

code(r"""
fig,ax=plt.subplots(figsize=(11,3.1))
ax.plot(t/60,v,lw=.55,color=GREY); ax.set(xlabel='time (min)',ylabel='activity_val (µM)',title='Figure 1 — Exact unfiltered decision signal over the full session')
fig.tight_layout(); plt.show()
""")

md(r"""
**Interpretation.** Full-session appearance is dominated by slow baseline movement and changing variance. An 8 s baseline is far too short to stabilize a 0.06–0.14 Hz phenomenon, and averaging channels does not make a component cortical: global superficial blood pressure, Mayer-wave, respiration-related and instrumentation changes can become *more* prominent after averaging. The absence of short channels prevents the standard direct test/regression of superficial physiology.
""")

md(r"""
## 2. Unconstrained spectral evidence and stationarity

The target band is 0.059–0.143 Hz. This overlaps the conventional low-frequency systemic/Mayer-wave region around 0.1 Hz; excluding cardiac and typical respiratory frequencies does not exclude systemic physiology. We inspect spectra without first forcing the signal into the target band, and measure where each session's strongest low-frequency peak lies.
""")

code(r"""
def psd_metrics(S):
    fs,v=S['fs'],S['v']; nper=min(len(v),max(256,int(fs*180)))
    f,p=welch(v,fs=fs,nperseg=nper,detrend='linear')
    m=(f>=.03)&(f<=.30); mt=(f>=1/17)&(f<=1/7)
    peak=float(f[m][np.argmax(p[m])]); target_peak=float(f[mt][np.argmax(p[mt])])
    # prominence against neighboring log-PSD background, deliberately simple and transparent
    flank=((f>=.03)&(f<1/17))|((f>1/7)&(f<=.30))
    excess=10*np.log10(np.max(p[mt])/np.median(p[flank]))
    return f,p,peak,target_peak,excess
f,p,peak,tp,excess=psd_metrics(S)
fig,ax=plt.subplots(figsize=(10.5,3.8)); ax.semilogy(f,p,color=GREY,lw=1)
ax.axvspan(1/17,1/7,color=TEAL,alpha=.16,label='chosen 7–17 s band'); ax.axvline(.1,color=GOLD,ls='--',label='0.1 Hz reference')
ax.set(xlim=(.02,.5),xlabel='frequency (Hz)',ylabel='PSD',title='Figure 2 — Unconstrained Welch spectrum (before target-band filtering)'); ax.legend(frameon=False)
fig.tight_layout(); plt.show()
print(f'Strongest 0.03–0.30 Hz peak: {peak:.3f} Hz ({1/peak:.1f} s); target-band peak: {tp:.3f} Hz; crude target peak excess: {excess:.1f} dB')
""")

code(r"""
rows=[]
for q in FILES:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); z=load_session(q)
    if z is None or len(z['v'])<max(64,z['fs']*30): continue
    ff,pp,pk,tpk,ex=psd_metrics(z)
    rows.append(dict(session=q.stem,duration_min=z['t'][-1]/60,peak_hz=pk,target_peak_hz=tpk,excess_db=ex))
spectra=pd.DataFrame(rows)
display(spectra.round(3))
fig,axs=plt.subplots(1,2,figsize=(11,3.7))
axs[0].scatter(spectra.duration_min,spectra.peak_hz,c=TEAL,s=45); axs[0].axhspan(1/17,1/7,color=TEAL,alpha=.12)
axs[0].set(xlabel='duration (min)',ylabel='strongest peak, 0.03–0.30 Hz',title='Figure 3a — Peak frequency varies by session')
axs[1].bar(range(len(spectra)),spectra.excess_db,color=TEAL); axs[1].axhline(0,color='k',lw=.8)
axs[1].set(xlabel='session (duration-sorted)',ylabel='target peak / flank median (dB)',title='Figure 3b — Target-band excess is descriptive, not a null test')
fig.tight_layout(); plt.show()
""")

md(r"""
These spectra may show power near 0.1 Hz, but that is not specific evidence for a neural oscillator: fNIRS is well known to contain low-frequency systemic oscillations, and this dataset contains neither concurrent physiology nor short-channel measurements needed to attribute the component. A peak must also be sustained enough for phase to be meaningful; a session-average PSD can hide transient bursts and frequency wandering.
""")

md(r"""
## 3. Spatial and chromophore diagnostics

An average across all channels preferentially retains globally coherent signals. High target-band inter-channel coherence therefore supports a real common signal, but argues against claiming it is focal occipital neural activity without superficial/systemic controls. HbO/HbR anti-correlation can be compatible with activation, but is neither necessary nor sufficient and is especially unreliable under motion/systemic contamination.
""")

code(r"""
sos=butter(3,[1/17,1/7],btype='bandpass',fs=fs,output='sos')
bhbo=sosfiltfilt(sos,S['hbo'],axis=0); bhbr=sosfiltfilt(sos,S['hbr'],axis=0)
c=np.corrcoef(bhbo.T); med_corr=np.median(c[np.triu_indices_from(c,1)])
chrom=np.array([np.corrcoef(bhbo[:,j],bhbr[:,j])[0,1] for j in range(bhbo.shape[1])])
fig,axs=plt.subplots(1,2,figsize=(10.5,3.7))
im=axs[0].imshow(c,vmin=-1,vmax=1,cmap='coolwarm'); plt.colorbar(im,ax=axs[0],shrink=.8,label='correlation')
axs[0].set(title='Figure 4a — Target-band HbO channel correlation',xlabel='pair',ylabel='pair')
axs[1].hist(chrom,bins=15,color=BLUE); axs[1].axvline(np.median(chrom),color=RED,ls='--')
axs[1].set(title='Figure 4b — Within-pair HbO–HbR correlation',xlabel='correlation',ylabel='pairs')
fig.tight_layout(); plt.show()
print(f'Median between-pair target-band HbO correlation: {med_corr:.2f}; median within-pair HbO–HbR correlation: {np.median(chrom):.2f}')
""")

md(r"""
## 4. What the causal predictor actually predicts

For every sample after a 40 s warm-up, the production `PhasePredictor` fits AR(20), extrapolates 12 s, applies a zero-phase 7–17 s bandpass to observed+synthetic samples, obtains Hilbert phase, and clamps instantaneous frequency to the passband. We compare its calls to extrema in a whole-session, offline bandpassed reference. This tests **implementation consistency**, not physiological truth, because predictor and reference share the same chosen band.
""")

code(r"""
offline=sosfiltfilt(sos,v)
distance=int(fs*4.5)
pk_idx,_=find_peaks(offline,distance=distance); tr_idx,_=find_peaks(-offline,distance=distance)
truth={'peak':t[pk_idx],'trough':t[tr_idx]}
pr=PhasePredictor(fs,recompute_hz=5,target_phase='either'); preds=[]
for ti,vi in zip(t,v):
    z=pr.update(float(ti),float(vi))
    if z is not None: preds.append(z.copy())

def nearest_err(p):
    a=truth[p['predicted_kind']]; j=np.argmin(abs(a-p['predicted_t'])); return p['predicted_t']-a[j]
err=np.array([nearest_err(x) for x in preds])
print(f'Causal-to-offline shared-band agreement: median |error|={np.median(abs(err)):.2f}s; 90th percentile={np.percentile(abs(err),90):.2f}s; within 1 s={100*np.mean(abs(err)<=1):.1f}%')
""")

code(r"""
# Visually honest 90 s panels. Mark forecasts issued ~2–4 s before their forecasted extremum, one per extremum cluster.
def selected_forecasts(lo,hi):
    cand=[p for p in preds if lo<=p['predicted_t']<hi and 1.5<=p['predicted_t']-p['t_now']<=4.5]
    out=[]
    for p in cand:
        if not out or abs(p['predicted_t']-out[-1]['predicted_t'])>2.5: out.append(p)
    return out

starts=[500,900,1500]
fig,axs=plt.subplots(3,1,figsize=(11,7.2),sharey=True)
for ax,lo in zip(axs,starts):
    hi=lo+90; m=(t>=lo)&(t<hi)
    ax.plot(t[m],(v[m]-np.mean(v[m]))/np.std(v[m]),color='#b5bdc7',lw=.7,label='raw (standardized)')
    ax.plot(t[m],offline[m]/np.std(offline[m]),color=TEAL,lw=1.8,label='offline target-band reference')
    for p0 in selected_forecasts(lo,hi):
        col=RED if p0['predicted_kind']=='peak' else BLUE
        ax.axvline(p0['predicted_t'],color=col,ls='--',lw=1.1,alpha=.85)
        ax.plot(p0['t_now'],np.interp(p0['t_now'],t,offline)/np.std(offline),'o',ms=3,color=col)
    ax.set(xlim=(lo,hi),ylabel='scaled',title=f'{lo}–{hi} s: dots=forecast issue time; dashed=predicted peak/trough')
axs[0].legend(frameon=False,ncol=2,loc='upper right'); axs[-1].set_xlabel('recording time (s)')
fig.suptitle('Figure 5 — Raw signal, offline band-limited waveform, and genuinely prospective calls',x=.08,ha='left',fontweight='bold')
fig.tight_layout(); plt.show()
""")

md(r"""
**Visual reading.** Dashed lines should fall near extrema of the teal reference, and the dot shows when each forecast was issued. This is the most favorable defensible visualization. Agreement with the gray raw trace will often be ambiguous because “phase” belongs to the constructed narrowband component, not to the broadband measurement. Forecasts that continually update are not independent events; dense repeated predictions must not be counted as a large validation sample.
""")

md(r"""
## 5. Negative control: can a non-oscillatory process look just as convincing?

We generate AR(1) colored noise with similar lag-1 autocorrelation and variance, with **no sinusoidal oscillator**, then feed it through the exact production predictor. A narrow bandpass plus Hilbert transform will still produce smooth cycles, bounded periods, and peak/trough forecasts. This directly tests whether the previous evidence is diagnostic of an oscillator.
""")

code(r"""
rng=np.random.default_rng(42); rho=np.clip(np.corrcoef(v[:-1],v[1:])[0,1],0,.999)
null=np.empty(len(v)); null[0]=0; innov=rng.normal(size=len(v))*np.std(v)*np.sqrt(max(1-rho*rho,1e-5))
for i in range(1,len(null)): null[i]=rho*null[i-1]+innov[i]
null_off=sosfiltfilt(sos,null)
npr=PhasePredictor(fs,recompute_hz=5,target_phase='either'); npreds=[]
for ti,vi in zip(t,null):
    z=npr.update(float(ti),float(vi))
    if z is not None: npreds.append(z.copy())
real_period=np.array([1/p['freq_inst_hz'] for p in preds]); null_period=np.array([1/p['freq_inst_hz'] for p in npreds])
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
axs[0].hist(real_period,bins=35,density=True,alpha=.65,color=TEAL,label='recording'); axs[0].hist(null_period,bins=35,density=True,alpha=.5,color=PURPLE,label='AR(1), no oscillator')
axs[0].axvspan(7,17,color=GREY,alpha=.08); axs[0].set(xlabel='reported period (s)',ylabel='density',title='Figure 6a — Constrained period estimates'); axs[0].legend(frameon=False)
lo=500;m=(t>=lo)&(t<lo+90)
axs[1].plot(t[m],null[m]/np.std(null[m]),color='#b5bdc7',lw=.7,label='AR(1) raw')
axs[1].plot(t[m],null_off[m]/np.std(null_off[m]),color=PURPLE,lw=1.8,label='after chosen bandpass')
axs[1].set(xlabel='time (s)',ylabel='scaled',title='Figure 6b — Null acquires plausible-looking cycles'); axs[1].legend(frameon=False)
fig.tight_layout(); plt.show()
print(f'Lag-1 correlation used for null: {rho:.4f}. Real median reported period={np.median(real_period):.2f}s; null={np.median(null_period):.2f}s.')
""")

md(r"""
## 6. Sensitivity: phase is conditional on analyst choices

A robust intrinsic phase should not change radically under reasonable neighboring passbands and buffer lengths. We compare predicted event times from several defensible parameterizations against the production default. Large disagreement means the trigger phase is partly an analysis convention.
""")

code(r"""
configs=[('default 7–17 s',7,17,40),('narrow 8–12 s',8,12,40),('wide 6–20 s',6,20,40),('default, 60 s buffer',7,17,60)]
events={}
for label,plo,phi,buf in configs:
    q=PhasePredictor(fs,period_lo_sec=plo,period_hi_sec=phi,buffer_sec=buf,recompute_hz=2,target_phase='trough')
    ev=[]; last=-1e9
    for ti,vi in zip(t,v):
        z=q.update(float(ti),float(vi))
        if z and z['predicted_t']-last>4: ev.append(z['predicted_t']); last=z['predicted_t']
    events[label]=np.array(ev)
base=events[configs[0][0]]; sens=[]
for label in events:
    e=events[label]; d=np.array([np.min(abs(e-x)) for x in base])
    sens.append((label,np.median(d),np.percentile(d,90)))
display(pd.DataFrame(sens,columns=['configuration','median nearest-event difference (s)','p90 difference (s)']).round(2))
""")

md(r"""
## 7. Conclusions and decision

### What is supported

- The production code can causally output peak/trough forecasts after warm-up, and those forecasts can align reasonably with extrema of an offline waveform made using the **same target band**.
- The filter strongly attenuates cardiac-frequency content. That is useful engineering, but does not address low-frequency systemic physiology within the passband.
- The figures make the operational behavior inspectable at a readable 90 s scale; dots distinguish when a prediction was available from where it predicted the event.

### What is not supported

- A 7–17 s constrained period estimate clustering near 10 s is **not evidence** for a 10 s intrinsic oscillation. The estimator is mathematically prevented from reporting anything outside that range.
- Smooth target-band cycles and plausible phase predictions are **not diagnostic**: the negative control produces them from non-oscillatory colored noise.
- These recordings cannot establish that the component is cortical or intrinsic. There are no short channels, no concurrent blood pressure/heart rate/respiration/end-tidal CO₂, no motion regressors, and no independent neural measure used for validation. A globally averaged HbO signal is particularly vulnerable to systemic physiology.
- Synthetic tests in `test_phase_predictor.py` assume the desired oscillator exists at unit amplitude; they test recovery under additive noise, not whether the real signal contains that oscillator.
- Agreement to an acausal, same-band offline reference is circular with respect to physiological validity and optimistic about edge behavior. It validates software consistency only.

### Decision

**The present evidence is insufficient to treat this method as a validated real-time brain-state trigger.** It may be used as an exploratory, dry-run algorithm, but phase-contingent human TMS should not rely on it without prospective validation. The earlier reports' statements that the predictor is “trustworthy,” that the intended signal is identified, or that the caveat is merely scheduler calibration are stronger than the evidence permits.

### Minimum next validation

1. Record short-separation channels plus continuous blood pressure, ECG/PPG, respiration and ideally end-tidal CO₂; regress/compare them prospectively.
2. Pre-register the passband and acceptance criteria using independent baseline data. Require a significant narrowband excess against phase-randomized/AR surrogates, sustained oscillatory episodes, adequate amplitude/SNR, and cross-channel topology inconsistent with a purely global superficial source.
3. Validate prospective forecasts on held-out recordings: freeze each forecast at a clinically relevant lead time, score one prediction per cycle against an independently defined reference, report circular phase error, missed cycles, false triggers, amplitude dependence and confidence intervals by subject/session.
4. Test whether phase-targeted versus phase-sham stimulation produces the predicted physiological/behavioral contrast. That—not a filtered trace looking sinusoidal—is the decisive validation.
5. Add online rejection/gating for low amplitude, motion, poor optical coupling, unstable frequency, AR failures and out-of-distribution signals. The current predictor always returns a phase once warm, even when phase is not meaningful.

**Bottom line:** the visualization can look convincing, and the software can be internally consistent, while the core biological premise remains unproven. At present, the correct interpretation is “phase of a chosen band-limited mixture,” not “phase of a demonstrated intrinsic cortical oscillation.”
""")

nbf.write(nb, OUT)
print(OUT)
