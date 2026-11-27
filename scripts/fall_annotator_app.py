"""
fall_annotator_app.py — Interactive Streamlit Fall Timestamp Annotator (10 FPS).

Lets the user watch every Fall video (`.avi` / `.mp4`) directly with burned-in tick
numbers and colored zone overlays, inspect any individual tick, and adjust the
`[fall_start_tick, fall_end_tick]` range saved in `data/skeleton_dataset/fall_timestamps.csv`:
  • [0, start - 1]  -> Idle  (standing/sitting before collapse)
  • [start, end]    -> Fall  (active collapse & impact window)
  • [end + 1, T-1]  -> Lying (motionless horizontal post-fall phase)

Run from workspace root:
    .venv\\Scripts\\streamlit.exe run scripts\\fall_annotator_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from annotate_fall_timestamps import (
    DEFAULT_OUTPUT_CSV,
    PER_FRAME_FALL_DIR,
    estimate_fall_interval_from_keypoints,
    generate_fall_timestamps_csv,
)
from utils.config import TARGET_FPS
from utils.skeleton_utils import compute_resample_indices


@st.cache_data(show_spinner=False, max_entries=16)
def _load_all_sampled_frames_rgb(video_path_str: str, target_width: int = 400) -> list[np.ndarray]:
    """Decode and return all 10-FPS resampled frames (RGB) for the given video."""
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return []
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if native_fps <= 0.0 or native_fps > 240.0:
        native_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices, _ = compute_resample_indices(total_frames, native_fps, float(TARGET_FPS))
    if len(indices) == 0:
        cap.release()
        return []

    frames_rgb: list[np.ndarray] = []
    cur_pos = -1
    for target_idx in indices:
        ti = int(target_idx)
        bgr: Optional[np.ndarray] = None
        while cur_pos < ti:
            ret, frm = cap.read()
            cur_pos += 1
            if not ret:
                frm = None
                break
            if cur_pos == ti:
                bgr = frm
        if bgr is None:
            break
        h, w = bgr.shape[:2]
        if w > target_width and w > 0:
            scale = float(target_width) / float(w)
            bgr = cv2.resize(bgr, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames_rgb


def _annotate_tick_frame_rgb(
    frame_rgb: np.ndarray,
    tick_idx: int,
    start_tick: int,
    end_tick: int,
) -> np.ndarray:
    """Burn `Tick X (X.Xs)` and colored zone tag (`IDLE` / `FALL` / `LYING`) onto an RGB frame."""
    out = frame_rgb.copy()
    h, w = out.shape[:2]
    if tick_idx < start_tick:
        color, tag = (22, 163, 74), "IDLE"
    elif tick_idx <= end_tick:
        color, tag = (220, 38, 38), "FALL"
    else:
        color, tag = (217, 119, 6), "LYING"

    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 4)

    tick_txt = f"Tick {tick_idx} ({tick_idx / float(TARGET_FPS):.1f}s)"
    (tw, th), _ = cv2.getTextSize(tick_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(out, (4, 4), (tw + 14, th + 14), (15, 23, 42), -1)
    cv2.putText(out, tick_txt, (9, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    (sw, sth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x0 = max(4, w - sw - 16)
    cv2.rectangle(out, (x0, 4), (w - 4, sth + 14), color, -1)
    cv2.putText(out, tag, (x0 + 6, sth + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _build_annotated_gif_bytes(
    frames_rgb: list[np.ndarray],
    start_tick: int,
    end_tick: int,
) -> bytes:
    """Render an in-memory 10-FPS looping GIF with burned-in tick numbers and zone colors."""
    if not frames_rgb:
        return b""
    pil_frames = [
        Image.fromarray(_annotate_tick_frame_rgb(frm, t_i, start_tick, end_tick))
        for t_i, frm in enumerate(frames_rgb)
    ]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(round(1000.0 / float(TARGET_FPS))),
        loop=0,
    )
    return buf.getvalue()


def _save_df(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main() -> None:
    st.set_page_config(
        page_title="ActionGuard AI — Fall Timestamp Annotator",
        page_icon="🎬",
        layout="wide",
    )
    st.title("🎬 Công cụ gán nhãn thời gian pha ngã (`Fall` Temporal Annotator)")
    st.info(
        "**Cách dùng nhanh trong 3 bước:**\n"
        "1. **Xem Video bên trái** (tự phát lặp với số `Tick` ở góc trên trái) hoặc nhìn **Dải ảnh toàn cảnh** bên dưới để biết chính xác Tick nào người bắt đầu ngã và Tick nào chạm đất xong.\n"
        "2. **Kéo thanh trượt 2 đầu (`[Bắt đầu ngã, Kết thúc ngã]`):** "
        "**Nút đầu trái** = Tick vừa mất thăng bằng đổ xuống (kết thúc `🟢 Idle`); "
        "**Nút đầu phải** = Tick vừa chạm sàn xong (sau đó chuyển sang `🟡 Lying`).\n"
        "3. Bấm **✅ Lưu & Sang clip tiếp theo** (hoặc chỉnh từng frame bằng thanh **Soi chi tiết 1 Tick** bên phải rồi lưu)."
    )

    csv_path = DEFAULT_OUTPUT_CSV
    if not csv_path.exists():
        with st.spinner("Đang tự động ước lượng mốc thời gian cho toàn bộ clip Fall..."):
            generate_fall_timestamps_csv(csv_path)

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        st.error(f"Không tìm thấy dữ liệu trong `{csv_path}`.")
        return

    # ── Sidebar navigation & batch controls ───────────────────────────────────
    with st.sidebar:
        n_total = len(df)
        n_verified = int((df["status"] == "verified").sum())
        st.metric("Tiến độ duyệt", f"{n_verified} / {n_total} clips")
        st.progress(n_verified / float(max(n_total, 1)))

        filt = st.radio(
            "Lọc danh sách",
            ["Tất cả", "Chưa duyệt (auto)", "Đã duyệt (verified)"],
            index=0,
        )
        if filt == "Chưa duyệt (auto)":
            sub = df[df["status"] != "verified"]
        elif filt == "Đã duyệt (verified)":
            sub = df[df["status"] == "verified"]
        else:
            sub = df

        if len(sub) == 0:
            st.success("Không còn clip nào trong bộ lọc này.")
            sub = df

        clip_ids = sub["clip_id"].tolist()
        if "annot_idx" not in st.session_state or st.session_state["annot_idx"] >= len(clip_ids):
            st.session_state["annot_idx"] = 0

        selected_id = st.selectbox(
            "Chọn clip Fall",
            clip_ids,
            index=int(st.session_state["annot_idx"]),
            key="annot_clip_select",
        )
        st.session_state["annot_idx"] = clip_ids.index(selected_id)

        st.divider()
        if st.button("⚡ Duyệt nhanh tất cả nhãn tự động", use_container_width=True):
            df.loc[df["status"] != "verified", "status"] = "verified"
            _save_df(df, csv_path)
            st.toast("Đã đánh dấu toàn bộ clip là verified!", icon="✅")
            st.rerun()

        if st.button("🔄 Tính lại các dòng chưa duyệt", use_container_width=True):
            generate_fall_timestamps_csv(csv_path, force_overwrite=False)
            st.rerun()

        st.download_button(
            "⬇️ Tải fall_timestamps.csv",
            data=csv_path.read_bytes(),
            file_name="fall_timestamps.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Current clip state & frames ───────────────────────────────────────────
    row_mask = df["clip_id"] == selected_id
    row_idx = int(df.index[row_mask][0])
    row = df.iloc[row_idx]

    video_path = PROJECT_ROOT / str(row["video_rel_path"])
    frames_rgb = _load_all_sampled_frames_rgb(str(video_path)) if video_path.is_file() else []
    total_ticks = len(frames_rgb) if frames_rgb else max(2, int(row["total_ticks"]))
    max_tick = max(1, total_ticks - 1)
    cur_start = max(0, min(int(row["fall_start_tick"]), max_tick))
    cur_end = max(cur_start, min(int(row["fall_end_tick"]), max_tick))

    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    c_info1.metric("Clip", str(selected_id))
    c_info2.metric("Nguồn / Thời lượng", f"{row['dataset_src']} · {row['duration_sec']}s ({total_ticks} ticks)")
    c_info3.metric("Trạng thái", str(row["status"]).upper())
    c_info4.metric("Gợi ý", str(row["notes"]))

    start_tick, end_tick = st.slider(
        "✂️ Thanh trượt 2 đầu — Kéo ĐẦU TRÁI (bắt đầu ngã) và ĐẦU PHẢI (chạm đất xong) @ 10 FPS",
        min_value=0,
        max_value=max_tick,
        value=(cur_start, cur_end),
        step=1,
        key=f"slider_{selected_id}",
        help="Đầu trái: Tick bắt đầu mất thăng bằng. Đầu phải: Tick vừa nằm/chạm sàn xong.",
    )

    pre_n = int(start_tick)
    act_n = int(end_tick - start_tick + 1)
    post_n = max(0, max_tick - int(end_tick))
    st.markdown(
        f"🟢 **Idle (Trước ngã):** `0..{max(0, start_tick - 1)}` ({pre_n} ticks = `{pre_n / 10.0:.1f}s`) &nbsp;|&nbsp; "
        f"🔴 **Fall (Đang ngã):** `{start_tick}..{end_tick}` ({act_n} ticks = `{act_n / 10.0:.1f}s`) &nbsp;|&nbsp; "
        f"🟡 **Lying (Nằm sau ngã):** `{min(max_tick, end_tick + 1)}..{max_tick}` ({post_n} ticks = `{post_n / 10.0:.1f}s`)"
    )

    # ── Video player (left) + Single-Tick Scrubber (right) ────────────────────
    col_vid, col_scrub = st.columns([1.1, 1])
    with col_vid:
        st.markdown("**▶️ Xem toàn bộ video (gắn sẵn số Tick + màu đoạn)**")
        if frames_rgb:
            gif_bytes = _build_annotated_gif_bytes(frames_rgb, int(start_tick), int(end_tick))
            st.image(gif_bytes, use_container_width=True)
        else:
            st.error(f"Không mở được video: `{row['video_rel_path']}`")

    with col_scrub:
        st.markdown("**🔍 Soi chi tiết từng Tick riêng lẻ**")
        mid_default = (int(start_tick) + int(end_tick)) // 2
        inspect_tick = st.slider(
            "Kéo để xem từng khung hình",
            min_value=0,
            max_value=max_tick,
            value=mid_default,
            step=1,
            key=f"inspect_{selected_id}",
        )
        if frames_rgb and int(inspect_tick) < len(frames_rgb):
            ann_single = _annotate_tick_frame_rgb(
                frames_rgb[int(inspect_tick)],
                int(inspect_tick),
                int(start_tick),
                int(end_tick),
            )
            st.image(ann_single, use_container_width=True)

    # ── 3 Milestone Frames ────────────────────────────────────────────────────
    st.markdown("**3 ảnh mốc đang cắt hiện tại**")
    mid_tick = (int(start_tick) + int(end_tick)) // 2
    col1, col2, col3 = st.columns(3)
    for col, t_val, title in (
        (col1, int(start_tick), f"1️⃣ Đầu trái: Bắt đầu ngã (Tick {start_tick})"),
        (col2, int(mid_tick), f"2️⃣ Giữa pha ngã (Tick {mid_tick})"),
        (col3, int(end_tick), f"3️⃣ Đầu phải: Chạm đất xong (Tick {end_tick})"),
    ):
        with col:
            st.caption(title)
            if frames_rgb and t_val < len(frames_rgb):
                st.image(
                    _annotate_tick_frame_rgb(frames_rgb[t_val], t_val, int(start_tick), int(end_tick)),
                    use_container_width=True,
                )

    # ── Visual Filmstrip across whole clip ────────────────────────────────────
    if frames_rgb:
        with st.expander("🎞️ Dải ảnh toàn cảnh clip (chọn nhanh số Tick theo dòng thời gian)", expanded=True):
            n_strip = min(8, len(frames_rgb))
            strip_ticks = sorted({int(round(x)) for x in np.linspace(0, len(frames_rgb) - 1, n_strip)})
            for row_start in range(0, len(strip_ticks), 4):
                cols = st.columns(4)
                for c_idx, t_val in enumerate(strip_ticks[row_start : row_start + 4]):
                    cols[c_idx].image(
                        _annotate_tick_frame_rgb(frames_rgb[t_val], t_val, int(start_tick), int(end_tick)),
                        use_container_width=True,
                    )

    # ── Optional kinematic curve from per_frame/*.npy ─────────────────────────
    stem = str(selected_id).split("/", 1)[-1]
    npy_path = PER_FRAME_FALL_DIR / f"{stem}.npy"
    if npy_path.is_file():
        arr = np.load(npy_path)
        est = estimate_fall_interval_from_keypoints(arr, fps=float(TARGET_FPS))
        with st.expander("📈 Đường cong độ cao vai (sh_y) & độ nghiêng thân (sin_theta)", expanded=False):
            chart_df = pd.DataFrame(
                {
                    "Độ cao vai (sh_y)": est["sh_y"],
                    "Độ nghiêng thân (sin_theta)": est["sin_theta"],
                    "Điểm chuyển động ngã": est["collapse_score"],
                }
            )
            st.line_chart(chart_df)

    # ── Save / Next / Reset buttons ───────────────────────────────────────────
    b1, b2, b3 = st.columns([2, 1.5, 1])
    if b1.button("✅ Lưu & Sang clip tiếp theo", type="primary", use_container_width=True):
        df.at[row_idx, "fall_start_tick"] = int(start_tick)
        df.at[row_idx, "fall_end_tick"] = int(end_tick)
        df.at[row_idx, "fall_start_sec"] = round(float(start_tick) / float(TARGET_FPS), 2)
        df.at[row_idx, "fall_end_sec"] = round(float(end_tick) / float(TARGET_FPS), 2)
        df.at[row_idx, "pre_fall_ticks"] = int(pre_n)
        df.at[row_idx, "active_fall_ticks"] = int(act_n)
        df.at[row_idx, "post_fall_lying_ticks"] = int(post_n)
        df.at[row_idx, "status"] = "verified"
        _save_df(df, csv_path)
        st.session_state["annot_idx"] = (clip_ids.index(selected_id) + 1) % max(len(clip_ids), 1)
        st.rerun()

    if b2.button("💾 Chỉ lưu clip này", use_container_width=True):
        df.at[row_idx, "fall_start_tick"] = int(start_tick)
        df.at[row_idx, "fall_end_tick"] = int(end_tick)
        df.at[row_idx, "fall_start_sec"] = round(float(start_tick) / float(TARGET_FPS), 2)
        df.at[row_idx, "fall_end_sec"] = round(float(end_tick) / float(TARGET_FPS), 2)
        df.at[row_idx, "pre_fall_ticks"] = int(pre_n)
        df.at[row_idx, "active_fall_ticks"] = int(act_n)
        df.at[row_idx, "post_fall_lying_ticks"] = int(post_n)
        df.at[row_idx, "status"] = "verified"
        _save_df(df, csv_path)
        st.toast(f"Đã lưu {selected_id}", icon="💾")

    if b3.button("↺ Gợi ý tự động", use_container_width=True) and npy_path.is_file():
        est_r = estimate_fall_interval_from_keypoints(np.load(npy_path), fps=float(TARGET_FPS))
        s_auto = int(est_r["fall_start_tick"])
        e_auto = int(est_r["fall_end_tick"])
        df.at[row_idx, "fall_start_tick"] = s_auto
        df.at[row_idx, "fall_end_tick"] = e_auto
        df.at[row_idx, "fall_start_sec"] = round(float(s_auto) / float(TARGET_FPS), 2)
        df.at[row_idx, "fall_end_sec"] = round(float(e_auto) / float(TARGET_FPS), 2)
        df.at[row_idx, "pre_fall_ticks"] = s_auto
        df.at[row_idx, "active_fall_ticks"] = e_auto - s_auto + 1
        df.at[row_idx, "post_fall_lying_ticks"] = max(0, max_tick - e_auto)
        df.at[row_idx, "status"] = "auto"
        _save_df(df, csv_path)
        st.rerun()


if __name__ == "__main__":
    main()
