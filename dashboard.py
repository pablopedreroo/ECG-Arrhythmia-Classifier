import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt
import wfdb
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CardioSense",
    layout="wide",
    initial_sidebar_state="expanded"
)

components.html("""
<script>
const hide = () => {
    const buttons = window.parent.document.querySelectorAll(
        '[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], button[kind="header"]'
    );
    buttons.forEach(b => b.remove());
};
hide();
setTimeout(hide, 500);
setTimeout(hide, 1500);
</script>
""", height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.stApp { background: #1c1c1e; color: #f5f5f7; }

section[data-testid="stSidebar"] {
    background: #2c2c2e;
    border-right: 1px solid #3a3a3c;
}

[data-testid="collapsedControl"]        { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.st-emotion-cache-zq5wmm                { display: none !important; }
.st-emotion-cache-1rtdyuf               { display: none !important; }

.cs-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0 16px 0; border-bottom: 1px solid #3a3a3c; margin-bottom: 24px;
}
.cs-title    { font-size: 20px; font-weight: 700; color: #f5f5f7; letter-spacing: -0.5px; }
.cs-subtitle { font-size: 12px; color: #8e8e93; margin-top: 2px; font-weight: 400; }

.cs-live {
    background: #30d158; color: #fff; font-size: 10px; font-weight: 700;
    padding: 4px 10px; border-radius: 20px; letter-spacing: 0.5px;
}
.cs-live-alert {
    background: #ff453a; color: #fff; font-size: 10px; font-weight: 700;
    padding: 4px 10px; border-radius: 20px; letter-spacing: 0.5px;
    animation: pulse 1s infinite;
}
.cs-live-warn {
    background: #ffd60a; color: #1c1c1e; font-size: 10px; font-weight: 700;
    padding: 4px 10px; border-radius: 20px; letter-spacing: 0.5px;
}
@keyframes pulse { 0%{opacity:1} 50%{opacity:0.5} 100%{opacity:1} }

/* Confidence badges */
.conf-high   { display:inline-block; background:#1a3a22; color:#30d158;
               border:1px solid #30d15844; border-radius:8px;
               padding:2px 10px; font-size:11px; font-weight:600; }
.conf-medium { display:inline-block; background:#3a3010; color:#ffd60a;
               border:1px solid #ffd60a44; border-radius:8px;
               padding:2px 10px; font-size:11px; font-weight:600; }
.conf-low    { display:inline-block; background:#2c1a1a; color:#ff453a;
               border:1px solid #ff453a44; border-radius:8px;
               padding:2px 10px; font-size:11px; font-weight:600; }

.cs-metric {
    background: #2c2c2e; border-radius: 16px; padding: 18px 20px;
    margin-bottom: 10px; border: 1px solid #3a3a3c;
}
.cs-metric-label  { font-size:11px; font-weight:500; color:#8e8e93;
                    letter-spacing:0.3px; text-transform:uppercase; margin-bottom:8px; }
.cs-metric-value        { font-size:32px; font-weight:700; letter-spacing:-1px; color:#f5f5f7; }
.cs-metric-value-green  { font-size:32px; font-weight:700; letter-spacing:-1px; color:#30d158; }
.cs-metric-value-red    { font-size:32px; font-weight:700; letter-spacing:-1px; color:#ff453a; }
.cs-metric-value-yellow { font-size:32px; font-weight:700; letter-spacing:-1px; color:#ffd60a; }

.cs-alert {
    background:#2c1a1a; border:1px solid #ff453a44;
    border-radius:16px; padding:16px 20px; margin:12px 0;
}
.cs-alert-title {
    font-size:14px; font-weight:600; color:#ff453a; margin-bottom:6px;
    display:flex; align-items:center; gap:8px;
}
.cs-alert-body { font-size:12px; color:#8e8e93; line-height:1.7; font-weight:400; }

.cs-warn {
    background:#2c2a10; border:1px solid #ffd60a44;
    border-radius:16px; padding:16px 20px; margin:12px 0;
}
.cs-warn-title {
    font-size:14px; font-weight:600; color:#ffd60a; margin-bottom:6px;
    display:flex; align-items:center; gap:8px;
}
.cs-warn-body { font-size:12px; color:#8e8e93; line-height:1.7; }

.cs-normal {
    background:#1a2c1e; border:1px solid #30d15844;
    border-radius:16px; padding:14px 20px; margin:12px 0;
}
.cs-normal-title { font-size:13px; font-weight:500; color:#30d158; }

.cs-beat-label {
    font-size:11px; font-weight:600; color:#8e8e93;
    text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; margin-top:14px;
}

.cs-log-title {
    font-size:11px; font-weight:600; color:#8e8e93;
    text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;
}
.cs-log-v {
    font-size:12px; font-weight:500; color:#ff453a; padding:5px 0;
    border-bottom:1px solid #3a3a3c; display:flex; justify-content:space-between;
}
.cs-log-r {
    font-size:12px; font-weight:500; color:#ffd60a; padding:5px 0;
    border-bottom:1px solid #3a3a3c; display:flex; justify-content:space-between;
}
.cs-log-n {
    font-size:12px; color:#3a3a3c; padding:5px 0;
    border-bottom:1px solid #2c2c2e; display:flex; justify-content:space-between;
}

.cs-sidebar-title { font-size:18px; font-weight:700; color:#f5f5f7; letter-spacing:-0.5px; margin-bottom:2px; }
.cs-sidebar-sub   { font-size:10px; color:#8e8e93; text-transform:uppercase; letter-spacing:1px; margin-bottom:20px; }
.cs-info { background:#3a3a3c; border-radius:12px; padding:14px 16px; font-size:12px; color:#8e8e93; line-height:2; }
.cs-info-row { display:flex; justify-content:space-between; }
.cs-info-key { color:#636366; }
.cs-info-val { color:#aeaeb2; font-weight:500; }

.stButton > button {
    background:#0a84ff; color:#fff; border:none; border-radius:12px;
    font-size:14px; font-weight:600; padding:12px 32px; width:100%;
    letter-spacing:0.2px; transition:all 0.15s;
}
.stButton > button:hover { background:#409cff; color:#fff; }

div[data-baseweb="select"] > div {
    background:#3a3a3c !important; border:1px solid #48484a !important;
    border-radius:10px !important; color:#f5f5f7 !important;
}
.stSelectbox label, .stSlider label {
    font-size:11px !important; font-weight:500 !important; color:#8e8e93 !important;
    text-transform:uppercase !important; letter-spacing:0.5px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model — 3-class (N=0, R=1, V=2) ───────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.conv(x))

class ECGResNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.blocks = nn.Sequential(
            ResBlock(64), nn.MaxPool1d(2),
            ResBlock(64), nn.MaxPool1d(2),
            ResBlock(64)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.pool(self.blocks(self.stem(x))))
    def forward_features(self, x):
        return self.blocks(self.stem(x))

@st.cache_resource
def load_model():
    m = ECGResNet(num_classes=3, in_channels=2)
    m.load_state_dict(torch.load('ecg_resnet_best.pth', map_location='cpu', weights_only=True))
    m.eval()
    return m

def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40.0):
    nyq = fs / 2
    b, a = butter(3, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def normalize_beat(x):
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mu) / std

def conf_badge(conf):
    pct = conf * 100
    if conf >= 0.85:
        return f'<span class="conf-high">● {pct:.0f}%</span>'
    elif conf >= 0.60:
        return f'<span class="conf-medium">● {pct:.0f}%</span>'
    else:
        return f'<span class="conf-low">● {pct:.0f}%</span>'

def gradcam(model, x_norm, target_class):
    x = torch.tensor(x_norm[np.newaxis])
    grads, acts = [], []
    fh = model.blocks[-1].register_forward_hook(lambda m,i,o: acts.append(o))
    bh = model.blocks[-1].register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    feats  = model.forward_features(x)
    logits = model.classifier(model.pool(feats))
    model.zero_grad()
    logits[0, target_class].backward()
    fh.remove(); bh.remove()
    g   = grads[0].squeeze(0); a = acts[0].squeeze(0)
    cam = torch.relu((g.mean(dim=-1)[:,None] * a).sum(dim=0)).cpu().detach().numpy()
    cam = np.interp(np.linspace(0, len(cam), x_norm.shape[-1]), np.arange(len(cam)), cam)
    return cam / cam.max() if cam.max() > 0 else cam

@st.cache_data
def load_record(record_id):
    rec  = wfdb.rdrecord(str(record_id), pn_dir='mitdb')
    ann  = wfdb.rdann(str(record_id), 'atr', pn_dir='mitdb')
    sig0 = bandpass_filter(rec.p_signal[:, 0])
    sig1 = bandpass_filter(rec.p_signal[:, 1])
    beats_raw, beats_norm, peaks, labels = [], [], [], []
    LABEL_MAP = {'N': 'N', 'L': 'R', 'R': 'R', 'e': 'N', 'j': 'N', 'V': 'V', 'E': 'V'}
    for s, sym in zip(ann.sample, ann.symbol):
        mapped = LABEL_MAP.get(sym)
        if mapped is None: continue
        if s-250 < 0 or s+250 > len(sig0): continue
        raw = np.stack([sig0[s-250:s+250], sig1[s-250:s+250]]).astype(np.float32)
        beats_raw.append(raw)
        beats_norm.append(normalize_beat(raw.copy()))
        peaks.append(s)
        labels.append(mapped)
    return sig0, beats_raw, beats_norm, peaks, labels, rec.fs

# ── Sidebar ───────────────────────────────────────────────────────────────
model = load_model()
le    = LabelEncoder(); le.fit(['N', 'R', 'V'])

st.sidebar.markdown("""
<div style="padding:20px 0 16px 0;border-bottom:1px solid #3a3a3c;margin-bottom:20px">
    <div class="cs-sidebar-title">CardioSense</div>
    <div class="cs-sidebar-sub">Arrhythmia Detection</div>
</div>
""", unsafe_allow_html=True)

TEST_RECORDS = [208, 210, 213, 214, 219, 221, 228, 231, 105, 106, 119, 200, 207, 217]
record_id    = st.sidebar.selectbox("Patient", TEST_RECORDS)
speed        = st.sidebar.select_slider("Speed", ["Slow","Normal","Fast"], value="Normal")
delay        = {"Slow": 0.5, "Normal": 0.18, "Fast": 0.06}[speed]

st.sidebar.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="cs-info">
    <div class="cs-info-row"><span class="cs-info-key">Model</span><span class="cs-info-val">ECGResNet</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Classes</span><span class="cs-info-val">N · R · V</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Leads</span><span class="cs-info-val">MLII · V5</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Sample rate</span><span class="cs-info-val">360 Hz</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Training</span><span class="cs-info-val">34 patients</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Validation</span><span class="cs-info-val">14 patients</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Macro F1</span><span class="cs-info-val">0.860</span></div>
    <div class="cs-info-row"><span class="cs-info-key">Dataset</span><span class="cs-info-val">MIT-BIH + INCART</span></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="margin-top:14px;background:#3a3a3c;border-radius:12px;padding:14px 16px;
            font-size:11px;color:#636366;line-height:2.2">
    <div style="color:#8e8e93;font-weight:600;margin-bottom:4px;
                text-transform:uppercase;letter-spacing:0.5px">Confidence</div>
    <div><span style="color:#30d158">●</span> &nbsp; High &nbsp;≥ 85%</div>
    <div><span style="color:#ffd60a">●</span> &nbsp; Medium &nbsp;60–85%</div>
    <div><span style="color:#ff453a">●</span> &nbsp; Low &nbsp;&lt; 60%</div>
</div>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────
with st.spinner(""):
    sig0, beats_raw, beats_norm, peaks, true_labels, fs = load_record(record_id)

# ── Header ────────────────────────────────────────────────────────────────
header_slot = st.empty()
header_slot.markdown(f"""
<div class="cs-header">
    <div>
        <div class="cs-title">Arrhythmia Detection System</div>
        <div class="cs-subtitle">Patient {record_id} &nbsp;·&nbsp; ECGResNet 3-class &nbsp;·&nbsp; MIT-BIH Arrhythmia Database</div>
    </div>
    <div class="cs-live">● READY</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────
col_main, col_panel = st.columns([3, 1])

with col_main:
    slot_ecg       = st.empty()
    slot_status    = st.empty()
    slot_cam_label = st.empty()
    slot_cam       = st.empty()

with col_panel:
    # ── Current beat waveform (top, merged with beat count) ───────────
    slot_beat = st.empty()
    # ──────────────────────────────────────────────────────────────────
    slot_m2 = st.empty()
    slot_m3 = st.empty()
    slot_m4 = st.empty()
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="cs-log-title">Event Log</div>', unsafe_allow_html=True)
    slot_log = st.empty()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
start = st.button("Start Monitoring")

CLASS_COLORS = {'N': '#30d158', 'R': '#ffd60a', 'V': '#ff453a'}

# ── Loop ──────────────────────────────────────────────────────────────────
if start:
    n_N, n_V, n_R = 0, 0, 0
    log_entries = []
    buf_peaks, buf_preds, buf_confs = [], [], []
    visible = 14

    for i, (b_raw, b_norm, peak, true_cls) in enumerate(
            zip(beats_raw, beats_norm, peaks, true_labels)):

        with torch.no_grad():
            logits = model(torch.tensor(b_norm[np.newaxis]))
            probs  = torch.softmax(logits, dim=1).numpy()[0]
            pred   = int(probs.argmax())
            conf   = float(probs[pred])
        pred_cls = le.inverse_transform([pred])[0]

        if pred_cls == 'N':   n_N += 1
        elif pred_cls == 'V': n_V += 1
        else:                 n_R += 1

        buf_peaks.append(peak)
        buf_preds.append(pred_cls)
        buf_confs.append(conf)

        # Header badge
        if pred_cls == 'V':
            badge = '<div class="cs-live-alert">● ALERT V</div>'
        elif pred_cls == 'R':
            badge = '<div class="cs-live-warn">● ALERT R</div>'
        else:
            badge = '<div class="cs-live">● LIVE</div>'

        header_slot.markdown(f"""
        <div class="cs-header">
            <div>
                <div class="cs-title">Arrhythmia Detection System</div>
                <div class="cs-subtitle">Patient {record_id} &nbsp;·&nbsp; ECGResNet 3-class &nbsp;·&nbsp; MIT-BIH Arrhythmia Database</div>
            </div>
            {badge}
        </div>
        """, unsafe_allow_html=True)

        # ECG plot
        sb  = max(0, len(buf_peaks) - visible)
        p_s = buf_peaks[sb] - 250
        p_e = min(len(sig0), peak + 300)
        t   = np.arange(p_s, p_e) / fs
        sig = sig0[p_s:p_e]

        fig, ax = plt.subplots(figsize=(11, 3))
        fig.patch.set_facecolor('#1c1c1e')
        ax.set_facecolor('#1c1c1e')
        ax.grid(which='major', color='#2c2c2e', linewidth=0.6)
        ax.grid(which='minor', color='#242426', linewidth=0.3)
        ax.minorticks_on()
        ax.plot(t, sig, color='#30d158', linewidth=1.0, zorder=3)

        for j in range(sb, len(buf_peaks)):
            pk  = buf_peaks[j]
            lbl = buf_preds[j]
            cf  = buf_confs[j]
            if p_s <= pk <= p_e:
                c = CLASS_COLORS.get(lbl, '#30d158')
                ax.axvline(pk/fs, color=c, alpha=0.3, linewidth=1.2, zorder=2)
                ax.scatter(pk/fs, sig0[pk], color=c, s=28, zorder=5, edgecolors='none',
                           alpha=1.0 if cf >= 0.60 else 0.35)

        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time (s)", color='#636366', fontsize=9)
        ax.set_ylabel("mV", color='#636366', fontsize=9)
        ax.tick_params(colors='#636366', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#2c2c2e')
        plt.tight_layout(pad=0.8)
        slot_ecg.pyplot(fig)
        plt.close()

        badge_conf = conf_badge(conf)

        # ── Current beat waveform — top of right panel ────────────────
        beat_color = CLASS_COLORS[pred_cls]
        beat_sig   = b_raw[0]   # lead MLII
        t_beat     = np.arange(len(beat_sig)) / fs

        fig_beat, ax_beat = plt.subplots(figsize=(3, 1.6))
        fig_beat.patch.set_facecolor('#2c2c2e')
        ax_beat.set_facecolor('#2c2c2e')

        ax_beat.fill_between(t_beat, beat_sig, alpha=0.15, color=beat_color)
        ax_beat.plot(t_beat, beat_sig, color=beat_color, linewidth=1.5)

        # Class — top right, inside plot
        ax_beat.text(0.96, 0.92, pred_cls,
                     transform=ax_beat.transAxes,
                     color=beat_color, fontsize=16, fontweight='bold',
                     ha='right', va='top')

        ax_beat.set_xticks([]); ax_beat.set_yticks([])
        ax_beat.set_title(f'Beat {i+1}',
                          color='#8e8e93', fontsize=9, fontweight='500',
                          loc='left', pad=4)
        for sp in ax_beat.spines.values():
            sp.set_edgecolor(beat_color)
            sp.set_linewidth(1.5)
        plt.tight_layout(pad=0.3)
        slot_beat.pyplot(fig_beat)
        plt.close()
        # ──────────────────────────────────────────────────────────────

        # Status panel
        if pred_cls == 'V':
            slot_status.markdown(f"""
            <div class="cs-alert">
                <div class="cs-alert-title">
                    ⚠ &nbsp; Ventricular Arrhythmia (PVC) Detected
                    <span style="font-weight:400;color:#636366;font-size:12px">Beat #{i+1}</span>
                    {badge_conf}
                </div>
                <div class="cs-alert-body">
                    Wide QRS complex with abnormal morphology consistent with a ventricular
                    ectopic beat originating outside the normal conduction system.
                    The Grad-CAM map below shows the regions driving this prediction.
                </div>
            </div>""", unsafe_allow_html=True)

        elif pred_cls == 'R':
            slot_status.markdown(f"""
            <div class="cs-warn">
                <div class="cs-warn-title">
                    ◈ &nbsp; Right Bundle Branch Block (RBBB) Detected
                    <span style="font-weight:400;color:#636366;font-size:12px">Beat #{i+1}</span>
                    {badge_conf}
                </div>
                <div class="cs-warn-body">
                    Wide QRS pattern consistent with delayed right ventricular activation.
                    RSR' morphology indicates conduction through myocardium rather than
                    the specialised conduction system.
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            slot_status.markdown(f"""
            <div class="cs-normal">
                <div class="cs-normal-title">
                    ✓ &nbsp; Normal sinus rhythm
                    <span style="color:#636366;font-weight:400;font-size:12px">
                        &nbsp;·&nbsp; Beat #{i+1}
                    </span>
                    &nbsp;{badge_conf}
                </div>
            </div>""", unsafe_allow_html=True)

        # Grad-CAM for V and R
        if pred_cls in ('V', 'R'):
            cam_cmap  = 'Reds' if pred_cls == 'V' else 'YlOrBr'
            alert_col = '#ff453a' if pred_cls == 'V' else '#ffd60a'
            cam       = gradcam(model, b_norm, pred)
            raw_sig   = b_raw[0]
            t_beat_gc = np.arange(len(raw_sig)) / fs
            ctx_start = max(0, peak - 500)
            ctx_end   = min(len(sig0), peak + 750)
            t_ctx     = np.arange(ctx_start, ctx_end) / fs
            sig_ctx   = sig0[ctx_start:ctx_end]
            lbl_ctx   = "PVC" if pred_cls == 'V' else "RBBB"

            fig2, (ax_ctx, ax_cam) = plt.subplots(2, 1, figsize=(11, 4),
                                                    gridspec_kw={'height_ratios':[1, 1.4]})
            fig2.patch.set_facecolor('#1c1c1e')

            ax_ctx.set_facecolor('#1c1c1e')
            ax_ctx.plot(t_ctx, sig_ctx, color='#aeaeb2', linewidth=0.9)
            ax_ctx.axvspan((peak-250)/fs, (peak+250)/fs, color=alert_col, alpha=0.12)
            ax_ctx.axvline((peak-250)/fs, color=alert_col, alpha=0.4, linewidth=0.8)
            ax_ctx.axvline((peak+250)/fs, color=alert_col, alpha=0.4, linewidth=0.8)
            ax_ctx.set_ylabel("mV", color='#636366', fontsize=8)
            ax_ctx.tick_params(colors='#636366', labelsize=7)
            ax_ctx.set_title(f"Context window · {lbl_ctx} beat highlighted",
                             color='#636366', fontsize=9, loc='left', pad=6)
            for sp in ax_ctx.spines.values(): sp.set_edgecolor('#2c2c2e')
            ax_ctx.grid(alpha=0.06, color='#2c2c2e')

            ax_cam.set_facecolor('#1c1c1e')
            cmap = plt.cm.get_cmap(cam_cmap)
            for k in range(len(t_beat_gc)-1):
                ax_cam.axvspan(t_beat_gc[k], t_beat_gc[k+1],
                               alpha=0.7, color=cmap(cam[k]), linewidth=0)
            ax_cam.plot(t_beat_gc, raw_sig, color='#ffffff', linewidth=1.1, zorder=3)
            ax_cam.set_xlabel("Time (s)", color='#636366', fontsize=8)
            ax_cam.set_ylabel("mV", color='#636366', fontsize=8)
            ax_cam.set_title("Grad-CAM · Model activation · Darker = stronger influence",
                             color='#636366', fontsize=9, loc='left', pad=6)
            ax_cam.tick_params(colors='#636366', labelsize=7)
            for sp in ax_cam.spines.values(): sp.set_edgecolor('#2c2c2e')
            ax_cam.grid(alpha=0.06, color='#2c2c2e')

            sm = plt.cm.ScalarMappable(cmap=cam_cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax_cam, fraction=0.018, pad=0.01)
            cb.ax.yaxis.set_tick_params(color='#636366', labelsize=7)
            cb.set_label('Activation', color='#636366', fontsize=8)

            plt.tight_layout(pad=0.8)
            slot_cam.pyplot(fig2)
            plt.close()
            time.sleep(5)

        else:
            slot_cam.empty()
            slot_cam_label.empty()
            time.sleep(delay)

        # Metrics
        total   = n_N + n_V + n_R
        pct_arr = 100 * (n_V + n_R) / total

        slot_m2.markdown(f"""
        <div class="cs-metric">
            <div class="cs-metric-label">PVC (V)</div>
            <div class="{'cs-metric-value-red' if n_V > 0 else 'cs-metric-value-green'}">{n_V}</div>
        </div>""", unsafe_allow_html=True)

        slot_m3.markdown(f"""
        <div class="cs-metric">
            <div class="cs-metric-label">RBBB (R)</div>
            <div class="{'cs-metric-value-yellow' if n_R > 0 else 'cs-metric-value-green'}">{n_R}</div>
        </div>""", unsafe_allow_html=True)

        slot_m4.markdown(f"""
        <div class="cs-metric">
            <div class="cs-metric-label">Arrhythmia rate</div>
            <div class="{'cs-metric-value-red' if pct_arr > 10 else 'cs-metric-value-green'}">{pct_arr:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Event log
        badge_log = conf_badge(conf)
        if pred_cls == 'V':
            entry = f'<div class="cs-log-v"><span>⚠ PVC #{i+1:04d}</span>{badge_log}</div>'
        elif pred_cls == 'R':
            entry = f'<div class="cs-log-r"><span>◈ RBBB #{i+1:04d}</span>{badge_log}</div>'
        else:
            entry = f'<div class="cs-log-n"><span>· N #{i+1:04d}</span>{badge_log}</div>'
        log_entries = [entry] + log_entries if 'log_entries' in dir() else [entry]
        log_entries = log_entries[:15]  # keep last 15 only
        slot_log.markdown("".join(log_entries), unsafe_allow_html=True)