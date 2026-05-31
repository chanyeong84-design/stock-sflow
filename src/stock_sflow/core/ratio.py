# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 수급 분석 비율 계산 유틸(분산비율, 보유비중, 주가선도비율, 통합 DF 생성)
# 주요 기능 : 컬럼 정합/정렬 무관(최신 기준), 0 나누기 방지, 결측 안전화, 반올림/클리핑 옵션
# 요구 사항 : Python 3.9+, pandas, numpy, logging
# 작성자    : (작성자명)
# 최종수정  : 2026-02-11
# 버전      : 1.4.1 (코드 중복 제거, applymap→map 변경)
# =============================================================================

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd


# =============================================================================
# 로깅 설정(필요 시 상위에서 재설정 가능)
# =============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# 상수: 컬럼 표준 순서(필요에 맞게 수정)
#  - 주의: 프로젝트 전반에서 '연기금' / '연기금등' 표기가 혼재 가능 → 하나로 통일 권장
# =============================================================================
FINAL_COLUMNS: Sequence[str] = [
    "개인", "외국인", "기관", "금융투자", "보험", "투신",
    "기타금융", "은행", "연기금", "사모펀드", "국가",
    "기타법인", "내외국인", "세력_1", "세력_2",
]


# =============================================================================
# 내부 유틸
# =============================================================================
def _as_percent(
    df_or_s: pd.DataFrame | pd.Series,
    decimals: int = 1,
    clip_0_100: bool = True,
) -> pd.DataFrame | pd.Series:
    """
    소수/NaN 허용 데이터(0~100 가정)를 반올림 및 선택적 클리핑.
    - DataFrame / Series 모두 지원
    """
    out = df_or_s.copy()
    if clip_0_100:
        out = out.clip(lower=0, upper=100)
    return out.round(decimals)


def _align_on_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    두 DF를 공통 컬럼 교집합으로 정렬/정합.
    반환: (left_aligned, right_aligned, common_cols)
    """
    common = left.columns.intersection(right.columns)
    if common.empty:
        raise ValueError("두 DataFrame 간 공통 컬럼이 없습니다.")
    return left[common], right[common], common


def _latest_row(df: pd.DataFrame) -> pd.Series:
    """
    ✅ 정렬 상태와 무관하게 '최신' 행을 반환.
    - DatetimeIndex: index.max() (가장 최신 날짜)
    - 그 외: 마지막 행(df.iloc[-1])을 최신으로 간주
    """
    if df is None or df.empty:
        raise ValueError("입력 DataFrame이 비어 있습니다.")

    if isinstance(df.index, pd.DatetimeIndex):
        return df.loc[df.index.max()]
    return df.iloc[-1]


def _row_to_numeric_series(row: pd.Series) -> pd.Series:
    """행(Series)을 숫자형으로 변환하고 NaN은 0으로 치환."""
    return pd.to_numeric(row, errors="coerce").fillna(0)


def _calculate_row_ratio(
    df: pd.DataFrame,
    *,
    decimals: int = 1,
    clip_0_100: bool = True,
) -> pd.DataFrame:
    """
    각 행의 비율 계산 (공통 로직)
    - 각 행의 각 값을 해당 행의 합계로 나눔
    - 결측은 0으로 간주
    - 행 합계가 0인 경우 해당 행은 NaN 처리
    
    ✅ get_holding_ratio와 get_driving_ratio의 공통 로직
    """
    if df is None or df.empty:
        raise ValueError("DataFrame이 비어 있습니다.")
    
    # 숫자형으로 변환하고 NaN은 0으로 치환
    df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 각 행의 합계 계산 (axis=1: 행 방향 합계)
    row_sums = df_numeric.sum(axis=1)
    
    # 합계가 0인 행은 NaN으로 처리 (0 나누기 방지)
    row_sums = row_sums.replace(0, np.nan)
    
    # 각 행을 해당 행의 합계로 나누고 100 곱하기
    # div(row_sums, axis=0): 각 행을 row_sums의 해당 값으로 나눔
    ratio_df = df_numeric.div(row_sums, axis=0) * 100.0
    
    # 후처리: 반올림 및 클리핑
    ratio_df = _as_percent(ratio_df, decimals=decimals, clip_0_100=clip_0_100)
    
    return ratio_df


# =============================================================================
# 공개 함수
# =============================================================================
def get_spread_ratio(
    accumulation_df: pd.DataFrame,
    highest_df: pd.DataFrame,
    *,
    decimals: int = 1,
    clip_0_100: bool = True,
) -> pd.DataFrame:
    """
    분산비율 = (매집수량 / 매집고점) * 100
    - 0 나누기 방지를 위해 분모 0 → NaN 처리
    - 무한대/±무한대 → NaN
    - 결과는 소수 N째 자리 반올림 및 [0, 100] 클리핑(옵션)

    ✅ 주의: 이 함수는 '시계열 전체(DataFrame)'를 반환한다.
    (표에서 최신 1행만 쓸지는 create_ratio_df에서 결정)
    """
    if accumulation_df is None or highest_df is None or accumulation_df.empty or highest_df.empty:
        raise ValueError("accumulation_df 또는 highest_df가 비어 있습니다.")

    acc_aligned, high_aligned, _ = _align_on_columns(accumulation_df, highest_df)

    denom = high_aligned.replace(0, np.nan)
    spread = (acc_aligned / denom) * 100.0
    spread = spread.replace([np.inf, -np.inf], np.nan)

    spread = _as_percent(spread, decimals=decimals, clip_0_100=clip_0_100)
    return spread


def get_holding_ratio(
    free_float_share_df: pd.DataFrame,
    *,
    decimals: int = 1,
    clip_0_100: bool = True,
) -> pd.DataFrame:
    """
    보유비중 = (각 행의 각 값 / 각 행의 합계) * 100
    - 결측은 0으로 간주
    - 행 합계가 0인 경우 해당 행은 NaN 처리

    ✅ 시계열 전체(DataFrame) 반환:
      - 각 행별로 독립적으로 비율 계산
      - get_spread_ratio와 동일한 형식
    """
    return _calculate_row_ratio(
        free_float_share_df,
        decimals=decimals,
        clip_0_100=clip_0_100,
    )


def get_driving_ratio(
    highest_df: pd.DataFrame,
    *,
    decimals: int = 1,
    clip_0_100: bool = True,
) -> pd.DataFrame:
    """
    주가선도비율 = (각 행의 각 값 / 각 행의 합계) * 100
    - 결측은 0으로 간주
    - 행 합계가 0인 경우 해당 행은 NaN 처리

    ✅ 시계열 전체(DataFrame) 반환:
      - 각 행별로 독립적으로 비율 계산
      - get_spread_ratio와 동일한 형식
    """
    return _calculate_row_ratio(
        highest_df,
        decimals=decimals,
        clip_0_100=clip_0_100,
    )


def create_ratio_df(
    driving_ratio_df: pd.DataFrame,
    holding_ratio_df: pd.DataFrame,
    spread_ratio_df: pd.DataFrame,
    *,
    final_columns: Sequence[str] = FINAL_COLUMNS,
    decimals: int = 1,
    clip_0_100: bool = True,
    add_percent_suffix: bool = False,
) -> pd.DataFrame:
    """
    세 비율을 하나의 표로 결합:
      index: ["주가선도", "보유비중", "분산추이"]
      columns: final_columns 순서로 재배열 (존재하지 않는 컬럼은 NaN)

    - driving_ratio_df / holding_ratio_df / spread_ratio_df: DataFrame (시계열)
      ✅ 표에는 각 DataFrame의 '최신 행'을 사용한다.

    - 옵션: 반올림/클리핑/퍼센트 접미사
    """
    if driving_ratio_df is None or driving_ratio_df.empty:
        raise ValueError("driving_ratio_df가 비어 있습니다.")
    if holding_ratio_df is None or holding_ratio_df.empty:
        raise ValueError("holding_ratio_df가 비어 있습니다.")
    if spread_ratio_df is None or spread_ratio_df.empty:
        raise ValueError("spread_ratio_df가 비어 있습니다.")

    # 1) 각 DataFrame에서 최신 행 추출
    drv_row = _latest_row(driving_ratio_df)
    hld_row = _latest_row(holding_ratio_df)
    spr_row = _latest_row(spread_ratio_df)

    # 2) 숫자형으로 변환 후 표준 컬럼 순서로 reindex (없는 것은 NaN)
    drv = pd.to_numeric(drv_row, errors="coerce").reindex(final_columns)
    hld = pd.to_numeric(hld_row, errors="coerce").reindex(final_columns)
    spr = pd.to_numeric(spr_row, errors="coerce").reindex(final_columns)

    # 3) 후처리: 반올림/클리핑
    drv = _as_percent(drv, decimals=decimals, clip_0_100=clip_0_100)
    hld = _as_percent(hld, decimals=decimals, clip_0_100=clip_0_100)
    spr = _as_percent(spr, decimals=decimals, clip_0_100=clip_0_100)

    out = pd.DataFrame(
        [drv, hld, spr],
        index=["주가선도", "보유비중", "분산추이"],
        columns=final_columns,
    )

    if add_percent_suffix:
        # ✅ applymap → map으로 변경 (pandas 2.1.0+)
        out = out.map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}%")

    return out