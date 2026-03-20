from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from mne import Epochs, create_info, events_from_annotations, pick_types
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import RawArray, concatenate_raws, read_raw_edf
from mne.time_frequency import psd_array_welch
from mne.channels import make_standard_montage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC


app = Flask(__name__)


CHANNELS = ["FCz", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]


@dataclass
class EEGState:
    raw: Any | None = None
    epochs: Any | None = None
    labels: Any | None = None
    status: str = "No data loaded"
    source: str = "None"


STATE = EEGState()


def _make_json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _extract_mi_epochs(raw):
    raw.load_data()
    raw.filter(8.0, 30.0, fir_design="firwin", verbose=False)
    events, event_id = events_from_annotations(raw, verbose=False)
    wanted = {}
    if "T1" in event_id:
        wanted["left"] = event_id["T1"]
    if "T2" in event_id:
        wanted["right"] = event_id["T2"]
    if not wanted:
        raise ValueError("Could not find motor imagery annotations (T1/T2).")
    picks = pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    epochs = Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=0.0,
        tmax=3.0,
        picks=picks,
        baseline=None,
        preload=True,
        proj=False,
        verbose=False,
    )
    labels = epochs.events[:, -1]
    return epochs, labels


def _get_signal_payload(epochs):
    data = epochs.get_data()  # n_epochs, n_channels, n_times
    sfreq = float(epochs.info["sfreq"])
    avg = data.mean(axis=0)
    n_times = avg.shape[1]
    max_points = min(1200, n_times)
    if n_times > max_points:
        idx = np.linspace(0, n_times - 1, max_points).astype(int)
    else:
        idx = np.arange(n_times)
    times = epochs.times[idx]
    selected_indices = list(range(min(len(CHANNELS), avg.shape[0])))
    channels = [epochs.ch_names[i] for i in selected_indices]
    signal = avg[selected_indices][:, idx]
    return {
        "times": times.tolist(),
        "channels": channels,
        "signal": signal.tolist(),
        "sfreq": sfreq,
    }


def _band_power_features(epochs):
    data = epochs.get_data().mean(axis=0)
    sfreq = float(epochs.info["sfreq"])
    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=2,
        fmax=45,
        n_fft=min(512, data.shape[-1]),
        verbose=False,
    )
    mean_psd = psd.mean(axis=0)
    bands = {"alpha": (8, 12), "beta": (13, 30), "gamma": (31, 45)}
    powers = {}
    for key, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        powers[key] = float(mean_psd[mask].mean()) if mask.any() else 0.0
    spec = np.log10(psd + 1e-12)
    return {
        "freqs": freqs.tolist(),
        "spectrogram": spec.tolist(),
        "band_power": powers,
    }


def _build_status(labels):
    unique, counts = np.unique(labels, return_counts=True)
    mapping = dict(zip(unique.tolist(), counts.tolist()))
    left = mapping.get(2, mapping.get(1, 0))
    right = mapping.get(3, mapping.get(2, 0)) if left == 0 else mapping.get(3, 0)
    return left, right


def _run_classification(epochs, labels):
    x = epochs.get_data()
    y = labels.copy()
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Classification requires exactly two classes.")
    label_map = {classes[0]: 0, classes[1]: 1}
    inv_label_map = {0: classes[0], 1: classes[1]}
    y_bin = np.array([label_map[v] for v in y], dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    x_train_csp = csp.fit_transform(x_train, y_train)
    x_test_csp = csp.transform(x_test)
    base = SVC(kernel="rbf", C=2.0, gamma="scale")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(x_train_csp, y_train)
    y_pred = model.predict(x_test_csp)
    probs = model.predict_proba(x_test_csp)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    top_idx = 0
    top_pred = int(y_pred[top_idx])
    conf = float(np.max(probs[top_idx]))
    pred_label = "Left Hand Movement" if top_pred == 0 else "Right Hand Movement"
    csp_cv = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    x_csp = csp_cv.fit_transform(x, y_bin)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        CalibratedClassifierCV(SVC(kernel="rbf", C=2.0, gamma="scale"), method="sigmoid", cv=3),
        x_csp,
        y_bin,
        cv=cv,
        scoring="accuracy",
    )
    return {
        "predicted_class": pred_label,
        "confidence": conf,
        "accuracy": acc,
        "cv_score": float(np.mean(cv_scores)),
        "confusion_matrix": cm,
        "class_labels": ["Left", "Right"],
        "sample_truth": "Left" if inv_label_map[y_test[top_idx]] == classes[0] else "Right",
    }


def _load_physionet_subject(subject: int = 1):
    runs = [4, 8, 12]
    files = eegbci.load_data(subject, runs, verbose=False)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, verbose=False)
    picks = [c for c in CHANNELS if c in raw.ch_names]
    if len(picks) >= 6:
        raw.pick(picks)
    else:
        raw.pick(raw.ch_names[: min(8, len(raw.ch_names))])
    epochs, labels = _extract_mi_epochs(raw)
    return raw, epochs, labels


def _load_synthetic():
    sfreq = 160.0
    n_epochs = 48
    epoch_len = int(3.0 * sfreq)
    total_samples = n_epochs * epoch_len
    t = np.arange(total_samples) / sfreq
    rng = np.random.default_rng(42)
    signals = []
    for i in range(len(CHANNELS)):
        alpha = np.sin(2 * np.pi * 10 * t + i * 0.2)
        beta = 0.6 * np.sin(2 * np.pi * 20 * t + i * 0.4)
        drift = 0.15 * np.sin(2 * np.pi * 1.2 * t + i * 0.1)
        noise = 0.5 * rng.standard_normal(len(t))
        signals.append(alpha + beta + drift + noise)
    data = np.array(signals) * 1e-6
    info = create_info(ch_names=CHANNELS, sfreq=sfreq, ch_types=["eeg"] * len(CHANNELS))
    raw = RawArray(data, info, verbose=False)
    raw.set_montage(make_standard_montage("standard_1005"), verbose=False)
    events = []
    for i in range(n_epochs):
        onset = i * epoch_len
        cls = 2 if i % 2 == 0 else 3
        events.append([onset, 0, cls])
        if cls == 2:
            idx = slice(onset, onset + epoch_len)
            raw._data[1:3, idx] += 0.8e-6 * np.sin(2 * np.pi * 11 * np.arange(epoch_len) / sfreq)
        else:
            idx = slice(onset, onset + epoch_len)
            raw._data[3:5, idx] += 0.8e-6 * np.sin(2 * np.pi * 18 * np.arange(epoch_len) / sfreq)
    events = np.array(events, dtype=int)
    epochs = Epochs(
        raw,
        events,
        event_id={"left": 2, "right": 3},
        tmin=0.0,
        tmax=3.0 - 1 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )
    labels = epochs.events[:, -1]
    return raw, epochs, labels


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/load_physionet")
def api_load_physionet():
    payload = request.get_json(silent=True) or {}
    subject = int(payload.get("subject", 1))
    try:
        raw, epochs, labels = _load_physionet_subject(subject)
        STATE.raw = raw
        STATE.epochs = epochs
        STATE.labels = labels
        left, right = _build_status(labels)
        STATE.status = f"Subject {subject:03d} | {left} left + {right} right epochs"
        STATE.source = "PhysioNet EEGBCI"
        signal = _get_signal_payload(epochs)
        analytics = _band_power_features(epochs)
        return jsonify(
            {
                "ok": True,
                "status": STATE.status,
                "source": STATE.source,
                "signal": signal,
                "analytics": analytics,
            }
        )
    except Exception as exc:
        return _make_json_error(f"PhysioNet load failed: {exc}", status=500)


@app.post("/api/upload_edf")
def api_upload_edf():
    if "file" not in request.files:
        return _make_json_error("No file supplied.")
    uploaded = request.files["file"]
    if uploaded.filename == "":
        return _make_json_error("Empty filename.")
    try:
        with tempfile.NamedTemporaryFile(suffix=".edf") as tmp:
            uploaded.save(tmp.name)
            raw = read_raw_edf(tmp.name, preload=True, verbose=False)
        if raw.info["nchan"] > 8:
            raw.pick(raw.ch_names[:8])
        epochs, labels = _extract_mi_epochs(raw)
        STATE.raw = raw
        STATE.epochs = epochs
        STATE.labels = labels
        left, right = _build_status(labels)
        STATE.status = f"Uploaded EDF | {left} left + {right} right epochs"
        STATE.source = "Uploaded EDF"
        signal = _get_signal_payload(epochs)
        analytics = _band_power_features(epochs)
        return jsonify(
            {
                "ok": True,
                "status": STATE.status,
                "source": STATE.source,
                "signal": signal,
                "analytics": analytics,
            }
        )
    except Exception as exc:
        return _make_json_error(f"EDF processing failed: {exc}", status=500)


@app.post("/api/generate_synthetic")
def api_generate_synthetic():
    raw, epochs, labels = _load_synthetic()
    STATE.raw = raw
    STATE.epochs = epochs
    STATE.labels = labels
    left, right = _build_status(labels)
    STATE.status = f"Subject 001 | {left} left + {right} right epochs"
    STATE.source = "Synthetic EEG"
    signal = _get_signal_payload(epochs)
    analytics = _band_power_features(epochs)
    return jsonify(
        {
            "ok": True,
            "status": STATE.status,
            "source": STATE.source,
            "signal": signal,
            "analytics": analytics,
        }
    )


@app.post("/api/run_classification")
def api_run_classification():
    if STATE.epochs is None or STATE.labels is None:
        return _make_json_error("Load PhysioNet data or generate synthetic data first.")
    try:
        result = _run_classification(STATE.epochs, STATE.labels)
        return jsonify({"ok": True, "result": result, "status": STATE.status, "source": STATE.source})
    except Exception as exc:
        return _make_json_error(f"Classification failed: {exc}", status=500)


@app.get("/api/status")
def api_status():
    return jsonify({"ok": True, "status": STATE.status, "source": STATE.source})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
