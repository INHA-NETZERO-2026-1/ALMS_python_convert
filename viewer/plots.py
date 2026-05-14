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
                     overlap_pct: int, win_func: str, fs: int = None):
    # =====================================================================
    # 남정우 수정 주석: 이 함수는 services.compute_interactive_stft 의
    # 반환값을 받아 그립니다. services.py 에서 이미 선형 magnitude 로
    # 변환되어 오므로, 여기서는 power_db 라는 파라미터명을 유지하되
    # 실제 값은 선형 magnitude 입니다 (하위 호환성 유지).
    # =====================================================================
    fig.clear()
    fig.patch.set_facecolor('#1e1e2e')
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1, 5], height_ratios=[3, 1],
        hspace=0.35, wspace=0.08,
    )

    freq_mask = freqs <= fmax_khz * 1000
    f_plot = freqs[freq_mask] / 1000
    p_plot = power_db[freq_mask, :]   # 실제로는 linear magnitude

    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    sr = fs if fs is not None else (int(n / (dur_ms / 1000)) if dur_ms > 0 else 0)

    # 좌측 FFT 파형 (Frequency를 y축으로 공유)
    ax_fft = fig.add_subplot(gs[0, 0])
    ax_fft.set_facecolor('#181825')
    if n > 0 and sr > 0:
        yf = np.abs(np.fft.rfft(raw)) / n
        xf = np.fft.rfftfreq(n, d=1 / sr)
        fft_mask = xf <= fmax_khz * 1000
        ax_fft.plot(yf[fft_mask], xf[fft_mask] / 1000, color='#a6e3a1', linewidth=1.0)
        actual_max = xf[fft_mask].max() / 1000 if fft_mask.any() else fmax_khz
        ax_fft.set_ylim([0, actual_max])
    ax_fft.set_ylabel("Frequency (kHz)", color='#a6adc8')
    ax_fft.set_xlabel("Amplitude", color='#a6adc8')
    ax_fft.tick_params(colors='#a6adc8')
    for sp in ax_fft.spines.values():
        sp.set_color('#45475a')

    # =====================================================================
    # 남정우 수정 (1/3) : Colormap 범위 — 퍼센타일 클리핑 적용
    # ---------------------------------------------------------------------
    # [수정 전] pcolormesh(...) vmin/vmax 없음 → 자동 범위
    #           → 극단적인 피크 1개가 전체 색상 범위를 결정
    #           → 나머지 신호가 모두 어둡게 뭉개짐
    #
    # [수정 후] vmax = 99.5 퍼센타일
    #           → 상위 0.5%는 클리핑, 핵심 주파수 대역이 선명하게 강조
    # =====================================================================
    vmax = np.percentile(p_plot, 99.5)
    vmax = vmax if vmax > 0 else 1.0

    # Spectrogram (y축은 좌측 FFT와 공유)
    ax_spec = fig.add_subplot(gs[0, 1], sharey=ax_fft)
    ax_spec.set_facecolor('#181825')

    # =====================================================================
    # 남정우 수정 (2/3) : pcolormesh 파라미터 변경
    # ---------------------------------------------------------------------
    # [수정 전] pcolormesh(times, f_plot, p_plot, shading='gouraud', cmap='inferno')
    #           → times 단위: 초(sec), shading='gouraud', vmin/vmax 없음
    #
    # [수정 후] times * 1000 → ms 단위로 변환
    #           shading='auto' (gouraud → auto, 데이터 크기 유연하게 처리)
    #           vmin=0, vmax=퍼센타일 클리핑 적용
    # =====================================================================
    im = ax_spec.pcolormesh(
        times * 1000, f_plot, p_plot,   # 남정우 수정: times → ms 단위
        shading='auto', cmap='inferno',
        vmin=0, vmax=vmax,              # 남정우 수정: 퍼센타일 클리핑
    )

    # =====================================================================
    # 남정우 수정 (3/3) : colorbar 라벨 변경
    # ---------------------------------------------------------------------
    # [수정 전] cbar.set_label('Power (dB)', ...)
    #           → dB 스케일 시절 라벨, 현재는 선형 magnitude 이므로 오해 소지
    #
    # [수정 후] cbar.set_label('Magnitude', ...)
    # =====================================================================
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.01)
    cbar.set_label('Magnitude', color='#a6adc8')   # 남정우 수정: 'Power (dB)' → 'Magnitude'
    cbar.ax.yaxis.set_tick_params(color='#a6adc8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a6adc8')

    ax_spec.set_title(
        f"STFT Spectrogram  |  {ch_name}  |  "
        f"Window={nperseg}  Overlap={overlap_pct}%  [{win_func}]",
        color='#cdd6f4', fontsize=11,
    )
    plt.setp(ax_spec.get_yticklabels(), visible=False)
    ax_spec.tick_params(colors='#a6adc8', left=False)
    for sp in ax_spec.spines.values():
        sp.set_color('#45475a')

    for f_khz, label, clr in [(60, '60 kHz (AE)', '#a6e3a1'), (100, '100 kHz (AE)', '#f38ba8')]:
        if f_khz <= fmax_khz:
            ax_spec.axhline(f_khz, color=clr, linewidth=0.8, linestyle='--', alpha=0.7, label=label)
    ax_spec.legend(loc='upper right', fontsize=8, facecolor='#313244', labelcolor='#cdd6f4')

    # 하단: Raw signal — 남정우 수정: 시간축 sec → ms
    t_ms = np.linspace(0, dur_ms, n)   # 남정우 수정: dur_ms 가 이미 ms 단위
    ax_raw = fig.add_subplot(gs[1, 1])
    ax_raw.set_facecolor('#181825')
    ax_raw.plot(t_ms, raw, color='#89b4fa', linewidth=0.5)
    ax_raw.set_xlabel("Time (ms)", color='#a6adc8')   # 남정우 수정: "Time (sec)" → "Time (ms)"
    ax_raw.set_ylabel("Amplitude (V)", color='#a6adc8')
    ax_raw.tick_params(colors='#a6adc8')
    for sp in ax_raw.spines.values():
        sp.set_color('#45475a')
    ax_raw.set_xlim([0, dur_ms])   # 남정우 수정: [times[0], times[-1]] → [0, dur_ms] (ms 기준)


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
