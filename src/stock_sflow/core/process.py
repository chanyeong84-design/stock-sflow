# =============================================================================
# 목적      : 수급 분석 데이터 전처리/파생지표 유틸리티 (정렬 정책 안전 버전)
# 정책      : ✅ 계산은 항상 오름차순(과거→현재) / ✅ UI·엑셀 출력만 내림차순(최신→과거)
# 작성자    : (작성자명)
# 최종수정  : 2026-01-28
# 버전      : 1.4.0 (Dead Code 제거)
# =============================================================================

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 로깅 설정(핸들러 없을 때만)
# =============================================================================
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# =============================================================================
# 상수 / 컬럼 매핑
# =============================================================================
RENAME_MAP = {
    "개인투자자": "개인",
    "외국인투자자": "외국인",
    "기관계": "기관",
    "금융투자": "금융투자",
    "보험": "보험",
    "투신": "투신",
    "기타금융": "기타금융",
    "은행": "은행",
    "연기금등": "연기금",
    "사모펀드": "사모펀드",
    "국가": "국가",
    "기타법인": "기타법인",
    "내외국인": "내외국인",
}

GROUP1_COLS = ["외국인", "금융투자", "보험", "투신", "기타금융", "은행", "연기금", "사모펀드", "국가", "기타법인"]
GROUP2_COLS = ["외국인", "금융투자", "보험", "투신", "기타금융", "은행", "연기금", "사모펀드", "국가"]

# =============================================================================
# 내부 유틸
# =============================================================================
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    '일자' 컬럼을 datetime으로 변환 후 인덱스로 세팅.
    ✅ 반환은 항상 오름차순(과거→현재)
    """
    if "일자" not in df.columns:
        raise KeyError("'일자' 컬럼이 없습니다.")

    out = df.copy()
    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    out = out.dropna(subset=["일자"]).set_index("일자")
    out = out.sort_index(ascending=True)
    return out


def _to_numeric_df(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """지정 컬럼을 숫자형으로 변환(coerce)하고 NaN은 0으로 치환."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """필수 컬럼 존재 여부 확인."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] 필요한 컬럼 누락: {missing}")


# =============================================================================
# 공개 함수
# =============================================================================
def add_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """세력 합계 컬럼 추가."""
    _check_required_columns(df, GROUP1_COLS, "세력_1 계산")
    _check_required_columns(df, GROUP2_COLS, "세력_2 계산")

    out = df.copy()
    out = _to_numeric_df(out, set(GROUP1_COLS) | set(GROUP2_COLS))
    out["세력_1"] = out[GROUP1_COLS].sum(axis=1)
    out["세력_2"] = out[GROUP2_COLS].sum(axis=1)
    return out


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) 컬럼 리네이밍
    2) '일자' → DatetimeIndex (✅ 오름차순)
    3) 세력_1/세력_2 추가
    """
    if df is None or df.empty:
        raise ValueError("입력 DataFrame이 비어 있습니다.")
    out = df.rename(columns=RENAME_MAP)
    out = _ensure_datetime_index(out)   # ✅ 오름차순 고정
    out = add_fraction(out)
    return out


def slice_data_by_date(
    df: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    *,
    use_nearest: bool = True,
    descending: bool = False,  # ✅ 기본: 계산용 오름차순
) -> pd.DataFrame:
    """
    DatetimeIndex 기반 df에서 날짜 구간 슬라이스.

    ✅ 기본 반환: 오름차순(과거→현재) = 계산 안정성
    필요 시 descending=True로 UI용(최신→과거) 반환 가능

    - use_nearest=True: 인덱스에 없는 날짜는 가장 가까운 날짜로 보정
      (start→크거나 같은, end→작거나 같은)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame 인덱스가 DatetimeIndex가 아닙니다. preprocess_data()를 먼저 호출하세요.")
    if df.empty:
        return df

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    base = df.sort_index(ascending=True)  # ✅ searchsorted/loc 안정성
    idx = base.index

    if use_nearest:
        # 시작 보정 (start 이상)
        if start_ts <= idx[0]:
            s_adj = idx[0]
        else:
            pos = idx.searchsorted(start_ts, side="left")
            s_adj = idx[pos] if pos < len(idx) else idx[-1]

        # 종료 보정 (end 이하)
        if end_ts >= idx[-1]:
            e_adj = idx[-1]
        else:
            pos = idx.searchsorted(end_ts, side="right") - 1
            e_adj = idx[pos] if pos >= 0 else idx[0]

        out = base.loc[s_adj:e_adj]
    else:
        if (start_ts not in idx) or (end_ts not in idx):
            raise ValueError("정확한 시작/종료일자가 인덱스에 존재하지 않습니다.")
        out = base.loc[start_ts:end_ts]

    out = out.sort_index(ascending=not descending)
    return out


def reverse_rolling_sum(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """
    forward-moving sum(현재 행 포함, 이후 window 합)을 만들기 위한 헬퍼.
    ✅ 입력은 오름차순(과거→현재)일 때 의미가 가장 직관적
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return s[::-1].rolling(window=window, min_periods=min_periods).sum()[::-1]


def calculate_forward_moving_sums(
    df: pd.DataFrame,
    investor_cols: Sequence[str],
    windows: Sequence[int] = (5, 20, 60, 120, 240),
) -> pd.DataFrame:
    """
    각 투자자 컬럼에 대해 forward moving sum(현재행 포함, 이후 window 합)을 계산.
    ✅ df 인덱스는 오름차순이어야 정확 → 내부에서 sort_index()로 보장
    """
    _check_required_columns(df, investor_cols, "forward moving sum")
    base = df.sort_index(ascending=True)
    result = pd.DataFrame(index=base.index)

    for col in investor_cols:
        s = pd.to_numeric(base[col], errors="coerce").fillna(0)
        for w in windows:
            result[f"{col}_{w}d"] = reverse_rolling_sum(s, w)

    return result


def get_total_cumsum(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    ✅ 오름차순(과거→현재) 기준으로,
    각 날짜 t에서 '과거(시작)~t까지' 누적합을 계산.

    반환 인덱스는 오름차순 유지.
    """
    _check_required_columns(df, cols, "누적합")

    base = df.sort_index(ascending=True)  # ✅ 정렬 고정
    numeric = _to_numeric_df(base, cols)[list(cols)]

    # ✅ 과거 누적합(일반 누적)
    csum = numeric.cumsum()
    csum.columns = [f"{c}" for c in csum.columns]

    return csum


def get_lowest(cumsum_df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 오름차순 인덱스 기준으로
    시작일부터 현재 날짜 t까지의 '누적 최저값(Expanding Min)'을 계산.
    """
    if cumsum_df.empty:
        return cumsum_df

    # 1. 인덱스 오름차순 정렬 (과거 -> 미래)
    base = cumsum_df.sort_index(ascending=True)

    # 2. 누적 최솟값 계산
    lowest = base.cummin() 
    
    # 3. 컬럼명 변경 (필요 시)
    lowest.columns = [f"{c}" for c in lowest.columns]

    return lowest


def get_accumulation(
    cumsum_df: pd.DataFrame,
    lowest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    매집수량 = 누적합계 - 최저점
    (동일 인덱스/컬럼 기준 가정, 불일치 시 공통 영역 연산)
    """
    if cumsum_df.empty or lowest_df.empty:
        return pd.DataFrame()

    c_base = cumsum_df.sort_index(ascending=True)
    l_base = lowest_df.sort_index(ascending=True)

    if (not c_base.columns.equals(l_base.columns)) or (not c_base.index.equals(l_base.index)):
        logging.warning("get_accumulation: 인덱스/컬럼 불일치 → 공통 영역으로 연산합니다.")

    common_cols = c_base.columns.intersection(l_base.columns)
    common_idx = c_base.index.intersection(l_base.index)

    acc = c_base.loc[common_idx, common_cols] - l_base.loc[common_idx, common_cols]
    return acc


def get_highest(accumulation_df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 오름차순 인덱스(시간순) 기준으로
    시작일부터 현재 시점 t까지의 '누적 최대값(Expanding Max)'을 계산.
    """
    if accumulation_df.empty:
        return accumulation_df

    # 1. 시간 순서대로 정렬 (과거 -> 미래)
    base = accumulation_df.sort_index(ascending=True)

    # 2. 누적 최댓값 계산
    highest = base.cummax()
    
    # 3. 컬럼명 포맷팅
    highest.columns = [f"{c}" for c in highest.columns]

    return highest


def extract_columns(df: pd.DataFrame, selected_columns: Sequence[str]) -> pd.DataFrame:
    """선택 컬럼만 추출. 누락 컬럼은 경고 로그."""
    missing = [c for c in selected_columns if c not in df.columns]
    if missing:
        logging.warning(f"다음 컬럼이 DataFrame에 없습니다: {missing}")
    present = [c for c in selected_columns if c in df.columns]
    return df[present].copy()


def get_common_sum_df(
    cumsum_df: pd.DataFrame,
    selected_cols: Sequence[str],
    sum_cols: Sequence[str],
    new_column_name: str,
) -> pd.DataFrame:
    """부분 프레임 + sum_cols 행 합계를 new_column_name으로 추가."""
    subset = extract_columns(cumsum_df, selected_cols)
    valid_sum_cols = [c for c in sum_cols if c in subset.columns]
    subset[new_column_name] = subset[valid_sum_cols].sum(axis=1)
    return subset


def get_major_fraction_1(cumsum_df: pd.DataFrame) -> pd.DataFrame:
    return extract_columns(cumsum_df, ["개인", "외국인", "기관", "기타법인"])


def get_major_fraction_2(cumsum_df: pd.DataFrame) -> pd.DataFrame:
    return extract_columns(cumsum_df, ["개인", "외국인", "세력_1", "세력_2"])


def get_institution(cumsum_df: pd.DataFrame) -> pd.DataFrame:
    selected_cols = ["기관", "금융투자", "보험", "투신", "기타금융", "은행", "연기금", "사모펀드", "국가"]
    sum_cols = ["금융투자", "보험", "투신", "기타금융", "은행", "연기금", "사모펀드", "국가"]
    return get_common_sum_df(cumsum_df, selected_cols, sum_cols, "기관 합")


def get_price(df: pd.DataFrame) -> pd.DataFrame:
    """'현재가'만 추출(인덱스=일자 가정)."""
    return extract_columns(df, ["현재가"])


def merge_with_price(price_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """
    인덱스(일자) 기준 병합. price_df의 열이 좌측에 먼저 오도록 재배열.
    ✅ 병합 전 오름차순 정렬로 표준화.
    """
    p = price_df.sort_index(ascending=True)
    o = other_df.sort_index(ascending=True)

    merged = pd.concat([p, o], axis=1)
    price_cols = list(p.columns)
    other_cols = [c for c in merged.columns if c not in price_cols]
    return merged[price_cols + other_cols]


def to_view_desc(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ UI/엑셀 표시용: 최신→과거로 변환 (계산 결과를 보여줄 때만 사용)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    return df.sort_index(ascending=False)