# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 키움 OPT10059(종목별 투자자/기관별) 일자별 조회 유틸리티
# 주요 기능 : 날짜 기준 페이지네이션, 진행 콜백, 컬럼 정규화/전처리, 안전장치(max 반복)
# 요구 사항 : Python 3.9+, pandas, pykiwoom
# 작성자    : (작성자명)
# 최종수정  : 2026-01-08
# 버전      : 1.2.2
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Optional, List, Sequence, Union

import pandas as pd
from pykiwoom.kiwoom import Kiwoom


# =============================================================================
# 상수 / 환경 설정
# =============================================================================
DEFAULT_DELAY_SEC: float = 0.2          # TR 호출 간 대기(초)
DEFAULT_MAX_ITERS: int = 1000           # 무한 루프 방지용 최대 반복 회수
REQ_TR_CODE: str = "OPT10059"           # 요청 TR 코드
REQ_OUTPUT: str = "종목별투자자기관별"  # 출력 필드명

# 불필요 컬럼(있으면 삭제)
DROP_COLS: List[str] = ["누적거래대금", "대비기호", "전일대비", "등락율", "누적거래량"]

# 기관별 수량 합이 모두 0이면 종료 판단용 필드
ZERO_STOP_COLS: List[str] = ["개인투자자", "외국인투자자", "기관계"]


# =============================================================================
# 내부 유틸리티
# =============================================================================
def _normalize_numeric_column(series: pd.Series, is_price: bool = False) -> pd.Series:
    """
    지정 시리즈를 숫자형으로 일관되게 변환.
    - 가격컬럼(is_price=True): 맨 앞 '-'만 제거(음수 가격 방지), '+'는 유지
    - 일반 수량/금액: 맨 앞 '+'만 제거, '-'는 음수로 유지
    - 공통: 퍼센트/콤마/공백 제거 후 to_numeric(coerce)
    """
    s = series.astype(str)
    if is_price:
        s = s.str.replace(r"^-", "", regex=True)      # 가격 앞의 음수 기호 제거
    else:
        s = s.str.replace(r"^\+", "", regex=True)     # 수량/금액 앞의 '+' 제거

    s = (
        s.str.replace("%", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace("\u00A0", " ", regex=False)     # NBSP 방어
         .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def _should_stop_by_zero_investors(df: pd.DataFrame) -> bool:
    """기관/개인/외인 합계(ZERO_STOP_COLS)가 모두 0이면 True."""
    need = [c for c in ZERO_STOP_COLS if c in df.columns]
    if not need:
        return False
    inv = df[need].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (inv.sum(axis=0) == 0).all()


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    API 반환 DF 전처리(플로팅/분석 안전 보장):
    - '일자' -> datetime(YYYYMMDD)
    - '현재가'는 가격 규칙, 그 외는 일반 규칙으로 숫자 변환
    - DROP_COLS 제거
    - '현재가'와 '일자'를 제외한 모든 값이 0인 행 제거
    - NaN 포함 행 제거 → float64 강제
    - '일자' 정렬
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 1) 컬럼별 정규화
    for col in list(df.columns):
        if col == "일자":
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
        else:
            df[col] = _normalize_numeric_column(df[col], is_price=(col == "현재가"))

    # 2) 불필요 컬럼 제거
    drop_targets = [c for c in DROP_COLS if c in df.columns]
    if drop_targets:
        df = df.drop(columns=drop_targets)

    # 3) 현재가/일자 제외 전부 0인 행 제거
    num_cols = [c for c in df.columns if c not in ["일자", "현재가"]]
    if num_cols:
        df = df.loc[~(df[num_cols].fillna(0) == 0).all(axis=1)]

    # 4) 숫자형 재보정 → NaN 행 제거 → float64 강제
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if num_cols:
        # NaN이 있으면 float64 강제 변환 시 문제 생길 수 있으므로 먼저 제거
        df = df.dropna(subset=num_cols, how="any").copy()
        df[num_cols] = df[num_cols].astype("float64")

    # 5) 날짜 정렬
    if "일자" in df.columns:
        df = df.dropna(subset=["일자"]).sort_values("일자")

    return df.reset_index(drop=True)


def _as_yyyymmdd(start_date: Union[str, int, datetime]) -> str:
    """start_date를 'YYYYMMDD' 문자열로 정규화."""
    if isinstance(start_date, datetime):
        s = start_date.strftime("%Y%m%d")
    else:
        s = str(start_date)
    s = s.strip().replace("-", "")
    # 간단 검증(8자리 숫자 형태)
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"start_date 형식이 올바르지 않습니다: {start_date!r} (예: 20250101)")
    return s


# =============================================================================
# 공개 함수
# =============================================================================
def query_stock_data(
    kiwoom: Kiwoom,
    stock_code: str,
    start_date: Union[str, int, datetime],
    *,
    delay_sec: float = DEFAULT_DELAY_SEC,
    max_iters: int = DEFAULT_MAX_ITERS,
    progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    """
    키움 OPT10059 TR(종목별투자자기관별)을 사용해 start_date부터 과거 방향으로
    페이지네이션 조회하여 정규화된 DataFrame을 반환.

    Parameters
    ----------
    kiwoom : Kiwoom
        pykiwoom.Kiwoom 인스턴스(로그인/접속 완료 상태여야 함)
    stock_code : str
        종목코드(예: '005930')
    start_date : str | int | datetime
        조회 기준 시작일(YYYYMMDD, 정수, 또는 datetime)
    delay_sec : float, optional
        호출 간 대기(초), 기본 0.2
    max_iters : int, optional
        최대 반복 횟수(무한루프 방지), 기본 1000
    progress_callback : Callable[[int, Optional[str]], None], optional
        진행 상황 콜백(인자: 반복회수, 기준일자)

    Returns
    -------
    pd.DataFrame
        전처리 및 정규화된 결과.
        컬럼 예시: ['일자', '현재가', '개인투자자', '외국인투자자', '기관계', ...]
        (API/계정 환경에 따라 컬럼 구성은 달라질 수 있음)

    Notes
    -----
    - 페이지 간 중복 행이 생길 수 있어, 최종 병합 후 drop_duplicates로 중복 제거함.
    - ZERO_STOP_COLS 합계가 모두 0이면 추가 조회를 중단.
    - API가 빈 DF를 반환하면 종료.
    """
    if not stock_code or not str(stock_code).strip():
        raise ValueError("stock_code가 비어 있습니다.")

    current_date_str = _as_yyyymmdd(start_date)

    # 키움 API 파라미터에서 종목코드는 'CODE_AL' 형태를 사용(기존 코드 유지)
    request_code = f"{stock_code.strip()}_AL"

    all_pages: List[pd.DataFrame] = []
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iters:
            break

        if progress_callback:
            progress_callback(iteration, current_date_str)

        params = {
            "일자": current_date_str,
            "종목코드": request_code,
            "금액수량구분": 2,   # (기존 값 유지)
            "매매구분": 0,      # (기존 값 유지)
            "단위구분": 1,      # (기존 값 유지)
            "next": 0,
            "output": REQ_OUTPUT,
        }

        try:
            df_raw = kiwoom.block_request(REQ_TR_CODE, **params)
        except Exception:
            break

        if df_raw is None or df_raw.empty:
            break

        # 페이지는 원본 그대로 저장(중복 제거는 병합 후 일괄 처리)
        df_page = df_raw

        # 종료 조건: 투자자 합계가 모두 0인 경우
        if _should_stop_by_zero_investors(df_page):
            break

        all_pages.append(df_page)

        # 다음 기준일자 설정(마지막 행의 일자)
        try:
            new_date_raw = df_raw.iloc[-1]["일자"]
        except Exception:
            break

        new_date_str = str(new_date_raw).strip()
        if new_date_str == current_date_str:
            break

        current_date_str = new_date_str
        time.sleep(max(0.0, float(delay_sec)))

    if not all_pages:
        return pd.DataFrame()

    # 병합
    result = pd.concat(all_pages, ignore_index=True)

    # 병합 직후 중복 제거(전체 행 동일 기준)
    result = result.drop_duplicates()

    # 전처리(숫자/날짜 강제, NaN 제거 등)
    result = _preprocess_df(result)

    return result.reset_index(drop=True)


def query_investor_tr(
    kiwoom: Kiwoom,
    stock_code: str,
    start_date: Union[str, int, datetime],
    *,
    amount_qty_div: int = 2,
    trade_div: int = 0,
    delay_sec: float = DEFAULT_DELAY_SEC,
    max_iters: int = DEFAULT_MAX_ITERS,
    progress_callback: Optional[Callable[[int, Optional[str]], None]] = None,
) -> pd.DataFrame:
    """
    OPT10059 TR 범용 호출. 금액수량구분/매매구분을 직접 지정 가능.

    Parameters
    ----------
    amount_qty_div : int
        1=금액, 2=수량
    trade_div : int
        0=순매수(매수-매도), 1=매수, 2=매도
    """
    if not stock_code or not str(stock_code).strip():
        raise ValueError("stock_code가 비어 있습니다.")

    current_date_str = _as_yyyymmdd(start_date)
    request_code = f"{stock_code.strip()}_AL"

    all_pages: List[pd.DataFrame] = []
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iters:
            break

        if progress_callback:
            progress_callback(iteration, current_date_str)

        params = {
            "일자": current_date_str,
            "종목코드": request_code,
            "금액수량구분": amount_qty_div,
            "매매구분": trade_div,
            "단위구분": 1,
            "next": 0,
            "output": REQ_OUTPUT,
        }

        try:
            df_raw = kiwoom.block_request(REQ_TR_CODE, **params)
        except Exception:
            break

        if df_raw is None or df_raw.empty:
            break

        if _should_stop_by_zero_investors(df_raw):
            break

        all_pages.append(df_raw)

        try:
            new_date_raw = df_raw.iloc[-1]["일자"]
        except Exception:
            break

        new_date_str = str(new_date_raw).strip()
        if new_date_str == current_date_str:
            break

        current_date_str = new_date_str
        time.sleep(max(0.0, float(delay_sec)))

    if not all_pages:
        return pd.DataFrame()

    result = pd.concat(all_pages, ignore_index=True)
    result = result.drop_duplicates()
    result = _preprocess_df(result)
    return result.reset_index(drop=True)
