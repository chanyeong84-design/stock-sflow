"""
수급 분석 웹앱 (64비트 Python / pykrx 직접 조회)

- 키움 HTS 불필요 — pykrx로 KRX에서 직접 데이터 조회
- 모바일·외부 네트워크에서 접속 가능
- 투자자 구분: 개인 / 외국인 / 기관 / 기타법인 (4종, KRX 제공 기준)
"""
import sys
import re
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import requests
import pickle as _pickle

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_sflow.app.config_store import ConfigStore
from stock_sflow.app.analysis_service import AnalysisService
from stock_sflow.core.sd_table import process_data as sd_process_data

# GitHub 데이터 저장소 설정
DATA_REPO = "chanyeong84-design/stock-sflow-data"
DATA_BRANCH = "main"

def _gh_token():
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return ""

def _gh_headers():
    token = _gh_token()
    return {"Authorization": f"token {token}"} if token else {}

@st.cache_data(ttl=300, show_spinner=False)
def download_pkl_from_github(filename: str):
    """GitHub 데이터 저장소에서 pkl 파일 다운로드."""
    url = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}/cache/{filename}"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code == 200:
        return _pickle.loads(r.content)
    return None

# =============================================================================
# 색상 / 상수
# =============================================================================
COL_COLORS = {
    "개인":    "#E74C3C", "외국인":  "#3498DB", "기관":    "#2ECC71",
    "금융투자": "#F39C12", "보험":    "#9B59B6", "투신":    "#1ABC9C",
    "기타금융": "#E67E22", "은행":    "#34495E", "연기금":  "#27AE60",
    "사모펀드": "#8E44AD", "국가":    "#2980B9", "기타법인": "#D35400",
    "내외국인": "#C0392B", "세력_1":  "#16A085", "세력_2":  "#7F8C8D",
}
PRICE_COLOR = "#2C3E50"
DEFAULT_VISIBLE = {"개인", "외국인", "기관"}

# =============================================================================
# 초기화
# =============================================================================
st.set_page_config(page_title="수급 분석", page_icon="📈", layout="wide")

cfg = ConfigStore(base_dir=PROJECT_ROOT)
analysis_service = AnalysisService()

for k, v in {"res": None, "avg_df": None, "stock_name": "", "queried_code": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# pykrx 데이터 조회
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_pykrx(code: str, from_date_str: str, to_date_str: str):
    """
    KRX에서 주가(OHLCV) + 투자자 순매수 데이터 조회.
    투자자 데이터는 KRX 인증 문제로 실패할 수 있으며, 실패 시 가격만 반환.
    반환: (DataFrame, 경고메시지 or None)
    """
    from pykrx import stock

    ohlcv = stock.get_market_ohlcv_by_date(from_date_str, to_date_str, code)
    if ohlcv is None or ohlcv.empty:
        return None, f"주가 데이터 없음 — 종목코드({code})를 확인하세요."

    idx = ohlcv.index
    zeros = pd.Series(0, index=idx)
    warn = None

    # 투자자별 순매수 조회 (실패해도 가격 데이터는 유지)
    inv_개인 = zeros.copy()
    inv_외국인 = zeros.copy()
    inv_기관 = zeros.copy()
    inv_기타 = zeros.copy()

    try:
        inv = stock.get_market_trading_volume_by_investor(from_date_str, to_date_str, code)
        if inv is not None and not inv.empty:
            common = idx.intersection(inv.index)
            if len(common) > 0:
                inv_개인   = inv.loc[common, "개인"]      if "개인"      in inv.columns else zeros.loc[common]
                inv_외국인 = inv.loc[common, "외국인합계"] if "외국인합계" in inv.columns else zeros.loc[common]
                inv_기관   = inv.loc[common, "기관합계"]  if "기관합계"  in inv.columns else zeros.loc[common]
                inv_기타   = inv.loc[common, "기타법인"]  if "기타법인"  in inv.columns else zeros.loc[common]
                idx = common
                ohlcv = ohlcv.loc[idx]
    except Exception:
        warn = "⚠️ 투자자 데이터를 가져올 수 없습니다 (KRX 제한). 주가 차트만 표시됩니다."

    df = pd.DataFrame({
        "일자":         idx.strftime("%Y%m%d"),
        "현재가":       ohlcv["종가"],
        "개인투자자":   inv_개인.reindex(idx, fill_value=0),
        "외국인투자자": inv_외국인.reindex(idx, fill_value=0),
        "기관계":       inv_기관.reindex(idx, fill_value=0),
        "금융투자": zeros.reindex(idx, fill_value=0),
        "보험":     zeros.reindex(idx, fill_value=0),
        "투신":     zeros.reindex(idx, fill_value=0),
        "기타금융": zeros.reindex(idx, fill_value=0),
        "은행":     zeros.reindex(idx, fill_value=0),
        "연기금등": zeros.reindex(idx, fill_value=0),
        "사모펀드": zeros.reindex(idx, fill_value=0),
        "국가":     zeros.reindex(idx, fill_value=0),
        "기타법인":     inv_기타.reindex(idx, fill_value=0),
        "내외국인": zeros.reindex(idx, fill_value=0),
    }).reset_index(drop=True)

    return df, warn


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_name(code: str) -> str:
    try:
        from pykrx import stock
        name = stock.get_market_ticker_name(code)
        return name if name else code
    except Exception:
        return code


# =============================================================================
# 캐시 파일 목록
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def list_cache_files():
    """GitHub 데이터 저장소에서 캐시 파일 목록 조회. 토큰 없으면 로컬 파일 사용."""
    token = _gh_token()

    if token:
        # 클라우드: GitHub에서 목록 조회
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/cache"
        try:
            r = requests.get(url, headers=_gh_headers(), timeout=10)
            if r.status_code == 200:
                options = {}
                for f in r.json():
                    name = f.get("name", "")
                    if not name.endswith(".pkl") or "_avg_price" in name:
                        continue
                    m = re.match(r"^([A-Za-z0-9]+)_(\d{8})\.pkl$", name)
                    if m:
                        code, d = m.groups()
                        label = f"[캐시] {code}  ({d[:4]}-{d[4:6]}-{d[6:]})"
                        options[label] = ("github", name)
                return options
        except Exception:
            pass
        return {}

    # 로컬: run_web.bat으로 실행 시 로컬 파일 사용
    options = {}
    cache_dir = cfg.cache_dir
    if not cache_dir.exists():
        return options
    for f in sorted(cache_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True):
        if "_avg_price" in f.name:
            continue
        m = re.match(r"^([A-Za-z0-9]+)_(\d{8})\.pkl$", f.name)
        if m:
            code, d = m.groups()
            label = f"[캐시] {code}  ({d[:4]}-{d[4:6]}-{d[6:]})"
            options[label] = ("local", f)
    return options


# =============================================================================
# 사이드바
# =============================================================================
st.sidebar.title("📈 수급 분석")

tab_direct, tab_cache = st.sidebar.tabs(["🌐 직접 조회", "💾 캐시"])

# ── 직접 조회 탭 ──────────────────────────────────────────────────────────
with tab_direct:
    code_input  = st.text_input("종목 코드", value="005930", key="code_input")
    date_start  = st.date_input("시작일", value=date.today() - timedelta(days=365 * 3), key="d_start")
    date_end    = st.date_input("종료일", value=date.today(), key="d_end")
    fetch_btn   = st.button("🔍 조회 및 분석", use_container_width=True)

    if fetch_btn:
        code = code_input.strip()
        if not code:
            st.error("종목 코드를 입력하세요.")
        elif date_start > date_end:
            st.error("시작일이 종료일보다 늦습니다.")
        else:
            with st.spinner(f"KRX에서 {code} 데이터 조회 중..."):
                df, err = fetch_pykrx(
                    code,
                    date_start.strftime("%Y%m%d"),
                    date_end.strftime("%Y%m%d"),
                )
            if err and df is None:
                st.error(err)
            else:
                if err:
                    st.warning(err)
                with st.spinner("분석 중..."):
                    try:
                        res = analysis_service.run(
                            df,
                            date_start.strftime("%Y%m%d"),
                            date_end.strftime("%Y%m%d"),
                        )
                        st.session_state.res          = res
                        st.session_state.avg_df       = None
                        st.session_state.stock_name   = get_stock_name(code)
                        st.session_state.queried_code = code
                    except Exception as e:
                        st.error(f"분석 오류: {e}")

    st.caption("💡 KRX 제공 데이터: 개인 / 외국인 / 기관 / 기타법인")

# ── 캐시 탭 (키움 앱으로 미리 받은 데이터) ──────────────────────────────
with tab_cache:
    cache_options = list_cache_files()
    if not cache_options:
        st.info("캐시 없음\n\n32비트 `run.py`로\n데이터 조회 후 새로고침")
    else:
        selected_label = st.selectbox("종목 선택", list(cache_options.keys()), key="cache_sel")
        cache_start = st.date_input("시작일", value=date.today() - timedelta(days=365*3), key="c_start")
        cache_end   = st.date_input("종료일", value=date.today(), key="c_end")
        cache_btn   = st.button("📂 캐시로 분석", use_container_width=True)

        if cache_btn:
            with st.spinner("캐시 로드 및 분석 중..."):
                try:
                    source, path_or_name = cache_options[selected_label]

                    if source == "github":
                        result_df   = download_pkl_from_github(path_or_name)
                        avg_name    = path_or_name.replace(".pkl", "_avg_price.pkl")
                        avg_df_full = download_pkl_from_github(avg_name)
                    else:  # local
                        result_df   = pd.read_pickle(path_or_name)
                        avg_path    = Path(str(path_or_name).replace(".pkl", "_avg_price.pkl"))
                        avg_df_full = pd.read_pickle(avg_path) if avg_path.exists() else None

                    if result_df is None:
                        st.error("데이터를 불러올 수 없습니다. sync_cache.bat을 실행했는지 확인하세요.")
                    else:
                        res = analysis_service.run(
                            result_df,
                            cache_start.strftime("%Y%m%d"),
                            cache_end.strftime("%Y%m%d"),
                        )
                        if avg_df_full is not None and not avg_df_full.empty:
                            s, e = res.sliced.index.min(), res.sliced.index.max()
                            avg_df = avg_df_full.loc[(avg_df_full.index >= s) & (avg_df_full.index <= e)]
                        else:
                            avg_df = None

                        st.session_state.res          = res
                        st.session_state.avg_df       = avg_df
                        m = re.match(r"\[캐시\] ([A-Za-z0-9]+)", selected_label)
                        st.session_state.stock_name   = m.group(1) if m else selected_label
                        st.session_state.queried_code = st.session_state.stock_name
                except Exception as e:
                    st.error(f"오류: {e}")

        st.caption("💡 캐시: 13종 투자자 상세 데이터 포함")

# =============================================================================
# Plotly 차트 함수
# =============================================================================
def _vis(col):
    return True if col in DEFAULT_VISIBLE else "legendonly"

def _legend():
    return dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


def make_acc_chart(acc_df, price_df, avg_df=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in acc_df.columns:
        fig.add_trace(go.Scatter(
            x=acc_df.index, y=acc_df[col], name=col,
            line=dict(color=COL_COLORS.get(col, "#808080"), width=2),
            visible=_vis(col),
        ), secondary_y=False)
    if price_df is not None:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df.iloc[:, 0],
            name="현재가", line=dict(color=PRICE_COLOR, width=2),
        ), secondary_y=True)
    if avg_df is not None and not avg_df.empty:
        for col in avg_df.columns:
            color = COL_COLORS.get(col, "#808080")
            series = pd.to_numeric(avg_df[col], errors="coerce")
            for period, dash in {5: "solid", 20: "dash", 60: "dot", 240: "dashdot"}.items():
                fig.add_trace(go.Scatter(
                    x=avg_df.index, y=series.rolling(period, min_periods=1).mean(),
                    name=f"{col} MA{period}",
                    line=dict(color=color, width=1.2, dash=dash),
                    visible="legendonly", legendgroup=f"ma_{col}",
                ), secondary_y=True)
    fig.update_layout(title="매집수량", hovermode="x unified",
                      height=560, legend=_legend(), margin=dict(t=80))
    fig.update_yaxes(title_text="매집수량", secondary_y=False)
    fig.update_yaxes(title_text="가격 (원)", secondary_y=True)
    return fig


def make_ratio_chart(df, price_df, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=COL_COLORS.get(col, "#808080"), width=1.5),
            visible=_vis(col),
        ), secondary_y=False)
    if price_df is not None:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df.iloc[:, 0],
            name="현재가", line=dict(color=PRICE_COLOR, width=2),
        ), secondary_y=True)
    fig.update_layout(title=title, hovermode="x unified",
                      height=480, legend=_legend(), margin=dict(t=80))
    fig.update_yaxes(title_text="비율 (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="가격 (원)", secondary_y=True)
    return fig


def make_avg_price_chart(avg_df, price_df):
    fig = go.Figure()
    if price_df is not None:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df.iloc[:, 0],
            name="현재가", line=dict(color=PRICE_COLOR, width=2.5),
        ))
    for col in avg_df.columns:
        fig.add_trace(go.Scatter(
            x=avg_df.index, y=avg_df[col], name=col,
            line=dict(color=COL_COLORS.get(col, "#808080"), width=1.5),
            visible=_vis(col),
        ))
    fig.update_layout(title="추정 평균가", hovermode="x unified",
                      height=480, legend=_legend(),
                      yaxis_title="가격 (원/주)", margin=dict(t=80))
    return fig


# =============================================================================
# 메인 화면
# =============================================================================
if st.session_state.res is None:
    st.markdown("""
    ## 📱 수급 분석 — 모바일 웹

    ### 🌐 직접 조회 (권장)
    사이드바 **[직접 조회]** 탭에서 종목코드를 입력하고 **[조회 및 분석]** 을 누르세요.
    - 키움 HTS 없이 KRX에서 직접 데이터를 가져옵니다.
    - 투자자 구분: **개인 / 외국인 / 기관 / 기타법인**

    ### 💾 캐시 조회
    PC에서 `run.py`로 미리 조회한 데이터가 있다면 **[캐시]** 탭에서 불러올 수 있습니다.
    - 투자자 구분: **13종 상세** (키움 데이터)

    > 💡 차트에서 범례 항목을 클릭하면 선을 켜고 끌 수 있습니다.
    """)
else:
    res    = st.session_state.res
    avg_df = st.session_state.avg_df
    name   = st.session_state.stock_name
    code   = st.session_state.queried_code

    st.title(f"📊 {name} ({code})")

    tab_acc, tab_drv, tab_hold, tab_sprd, tab_avg, tab_sd = st.tabs([
        "📦 매집수량", "🚀 주가선도비율", "🏦 보유비중",
        "📊 분산추이",  "💰 추정 평균가",  "📋 수급분석표",
    ])

    with tab_acc:
        st.plotly_chart(make_acc_chart(res.acc, res.price, avg_df), use_container_width=True)

    with tab_drv:
        st.plotly_chart(make_ratio_chart(res.drv, res.price, "주가선도비율"), use_container_width=True)

    with tab_hold:
        st.plotly_chart(make_ratio_chart(res.hold, res.price, "보유비중"), use_container_width=True)

    with tab_sprd:
        st.plotly_chart(make_ratio_chart(res.sprd, res.price, "분산추이"), use_container_width=True)

    with tab_avg:
        if avg_df is not None and not avg_df.empty:
            st.plotly_chart(make_avg_price_chart(avg_df, res.price), use_container_width=True)
        else:
            st.info("추정 평균가는 캐시 모드(키움 데이터)에서만 제공됩니다.")

    with tab_sd:
        try:
            df_table = sd_process_data(
                res.sliced.sort_index(ascending=False),
                res.acc.sort_index(ascending=False),
                res.high.sort_index(ascending=False),
            )
            st.dataframe(df_table, use_container_width=True, height=600)
        except Exception as e:
            st.error(f"수급분석표 오류: {e}")
