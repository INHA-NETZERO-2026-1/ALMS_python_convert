"""순수 matplotlib 플로팅 함수. fig를 받아 그림만 그린다.
canvas.draw()는 호출자(UI 슬롯) 책임."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm


def set_korean_font():
    """OS별 한글 폰트를 matplotlib에 자동 적용. 모듈 import 시 1회 호출."""
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)
    if chosen:
        matplotlib.rc('font', family=chosen)
    matplotlib.rcParams['axes.unicode_minus'] = False


set_korean_font()


def _ch_display_name(ch, idx: int) -> str:
    name = (getattr(ch, "name", None) or getattr(ch, "ch_name", None) or f"CH{idx}")
    return name.strip() or f"CH{idx}"


def plot_raw(fig, raw, fs: int, dur_ms: int, ch_name: str):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor('#181825')
    fig.patch.set_facecolor('#1e1e2e')

    raw = np.asarray(raw, dtype=np.float64)
    t = np.linspace(0, dur_ms / 1000, len(raw))
    rms_val = float(np.sqrt(np.mean(raw ** 2))) if len(raw) > 0 else 0.0

    ax.plot(t, raw, color='#89b4fa', linewidth=0.6, alpha=0.9)
    ax.axhline(0, color='#45475a', linewidth=0.5)
    ax.set_title(
        f"Raw Signal  |  {ch_name}  |  RMS = {rms_val:.5f} V  |  {len(raw):,} samples",
        color='#cdd6f4', fontsize=12,
    )
    ax.set_xlabel("Time (sec)", color='#a6adc8')
    ax.set_ylabel("Amplitude (V)", color='#a6adc8')
    ax.tick_params(colors='#a6adc8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color('#45475a')

    fig.tight_layout()


def plot_spectrogram(fig, freqs, times, power_db, raw, dur_ms: int,
                     ch_name: str, fmax_khz: int, nperseg: int,
                     overlap_pct: int, win_func: str):
    fig.clear()
    fig.patch.set_facecolor('#1e1e2e')
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    freq_mask = freqs <= fmax_khz * 1000
    f_plot = freqs[freq_mask] / 1000
    p_plot = power_db[freq_mask, :]

    ax_spec = fig.add_subplot(gs[0])
    ax_spec.set_facecolor('#181825')
    im = ax_spec.pcolormesh(times, f_plot, p_plot, shading='gouraud', cmap='inferno')

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.01)
    cbar.set_label('Power (dB)', color='#a6adc8')
    cbar.ax.yaxis.set_tick_params(color='#a6adc8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a6adc8')

    ax_spec.set_title(
        f"STFT Spectrogram  |  {ch_name}  |  "
        f"Window={nperseg}  Overlap={overlap_pct}%  [{win_func}]",
        color='#cdd6f4', fontsize=11,
    )
    ax_spec.set_ylabel("Frequency (kHz)", color='#a6adc8')
    ax_spec.tick_params(colors='#a6adc8')
    for sp in ax_spec.spines.values():
        sp.set_color('#45475a')

    for f_khz, label, clr in [(60, '60 kHz (AE)', '#a6e3a1'), (100, '100 kHz (AE)', '#f38ba8')]:
        if f_khz <= fmax_khz:
            ax_spec.axhline(f_khz, color=clr, linewidth=0.8, linestyle='--', alpha=0.7, label=label)
    ax_spec.legend(loc='upper right', fontsize=8, facecolor='#313244', labelcolor='#cdd6f4')

    raw = np.asarray(raw, dtype=np.float64)
    t = np.linspace(0, dur_ms / 1000, len(raw))
    ax_raw = fig.add_subplot(gs[1])
    ax_raw.set_facecolor('#181825')
    ax_raw.plot(t, raw, color='#89b4fa', linewidth=0.5)
    ax_raw.set_xlabel("Time (sec)", color='#a6adc8')
    ax_raw.set_ylabel("Amplitude (V)", color='#a6adc8')
    ax_raw.tick_params(colors='#a6adc8')
    for sp in ax_raw.spines.values():
        sp.set_color('#45475a')
    if len(times) > 0:
        ax_raw.set_xlim([times[0], times[-1]])


def plot_overview(fig, channels, dur_ms: int, event_ch: int):
    fig.clear()
    fig.patch.set_facecolor('#1e1e2e')

    n_ch = len(channels)
    if n_ch == 0:
        return

    cols = 4
    rows = (n_ch + cols - 1) // cols

    for i, ch in enumerate(channels):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor('#181825')

        raw = np.asarray(ch.raw_data, dtype=np.float64)
        t = np.linspace(0, dur_ms / 1000, len(raw))
        clr = '#f9e2af' if i == event_ch else '#89b4fa'
        ax.plot(t, raw, color=clr, linewidth=0.4)

        ch_name = _ch_display_name(ch, i)
        rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) > 0 else 0.0
        marker = "★ " if i == event_ch else ""
        ax.set_title(f"{marker}{ch_name}\nRMS={rms:.4f}", fontsize=7, color='#cdd6f4')
        ax.tick_params(labelsize=6, colors='#6c7086')
        for sp in ax.spines.values():
            sp.set_color('#313244')

    fig.suptitle("전채널 Raw Signal 개요  (★ 노랑 = 이벤트 채널)", color='#cdd6f4', fontsize=12)
    fig.tight_layout()


def plot_rms_bar(fig, results: list, event_ch: int):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor('#181825')
    fig.patch.set_facecolor('#1e1e2e')

    names = [r.get("ch_name", f"CH{r['ch_index']}") for r in results]
    rms_v = [r.get("rms", 0) for r in results]
    colors = []
    for r in results:
        if r["ch_index"] == event_ch:
            colors.append('#f9e2af')
        elif r.get("rms", 0) > 0.01:
            colors.append('#f38ba8')
        elif r.get("rms", 0) > 0.001:
            colors.append('#fab387')
        else:
            colors.append('#89b4fa')

    ax.bar(range(len(names)), rms_v, color=colors, edgecolor='#45475a', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='#a6adc8')
    ax.set_ylabel("RMS (V)", color='#a6adc8')
    ax.set_title("채널별 RMS 레벨  (★ 노랑 = 이벤트 채널)", color='#cdd6f4')
    ax.tick_params(colors='#a6adc8')
    for sp in ax.spines.values():
        sp.set_color('#45475a')

    fig.tight_layout()
