# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 데이터 조회 책임 분리 (캐시 우선 + Kiwoom API)
# 주요 기능 : Kiwoom 연결, 캐시 로드/저장, DF 필수 컬럼 검증
# 요구 사항 : pandas, pykiwoom
# 작성자    : (작성자명 기입)
# 최종수정  : 2025-12-19
# 버전      : 1.0.0
# =============================================================================

# =============================== 표준 라이브러리 ==============================
from typing import Optional, Callable, Tuple, List
from pathlib import Path

# =============================== 서드파티 라이브러리 ==========================
import numpy as np
import pandas as pd
from pykiwoom.kiwoom import Kiwoom

# =============================== 프로젝트 내부 모듈 ===========================
from stock_sflow.core import mykiwoom
from stock_sflow.core.process import RENAME_MAP
from stock_sflow.utils import general


# OPT10059 단위구분=1 기준
# 금액수량구분=1(금액): 백만원 단위 → 원 환산 계수
# 금액수량구분=2(수량): 주(株) 단위
# 추정 평균가(원/주) = 누적금액(백만원) * AVG_PRICE_AMT_SCALE / 누적수량(주)
# ⚠️ 실제 API 응답 단위가 다를 경우 이 값을 조정하세요 (예: 1 또는 1_000)
AVG_PRICE_AMT_SCALE: int = 1_000_000

_INVESTOR_COLS: List[str] = [
    "개인투자자", "외국인투자자", "기관계", "금융투자", "보험",
    "투신", "기타금융", "은행", "연기금등", "사모펀드",
    "국가", "기타법인", "내외국인",
]


def _compute_estimated_avg_price(
    amount_df: pd.DataFrame,
    qty_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    투자자별 추정 평균가 = 누적매수금액 * AVG_PRICE_AMT_SCALE / 누적매수수량.
    반환 DataFrame의 인덱스는 DatetimeIndex(일자), 컬럼명은 RENAME_MAP 적용 후.
    """
    amt = amount_df.copy()
    qty = qty_df.copy()

    if "일자" in amt.columns:
        amt = amt.set_index("일자")
    if "일자" in qty.columns:
        qty = qty.set_index("일자")

    amt = amt.sort_index(ascending=True)
    qty = qty.sort_index(ascending=True)

    common_idx = amt.index.intersection(qty.index)
    inv_cols = [c for c in _INVESTOR_COLS if c in amt.columns and c in qty.columns]

    result = pd.DataFrame(index=common_idx)
    for col in inv_cols:
        daily_amt = pd.to_numeric(amt.loc[common_idx, col], errors="coerce").fillna(0)
        daily_qty = pd.to_numeric(qty.loc[common_idx, col], errors="coerce").fillna(0)
        # 당일 매수수량이 0인 날은 NaN 처리
        result[col] = (daily_amt * AVG_PRICE_AMT_SCALE / daily_qty.replace(0, np.nan)).round(0)

    return result.rename(columns=RENAME_MAP)


class DataService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.kiwoom: Optional[Kiwoom] = None
        self.stock_name: str = ""  # 마지막으로 조회한 종목 이름

    def connect(self, log: Callable[[str], None]) -> bool:
        try:
            if self.kiwoom is None:
                self.kiwoom = Kiwoom()

            try:
                if hasattr(self.kiwoom, "GetConnectState") and self.kiwoom.GetConnectState() == 1:
                    return True
            except Exception:
                pass

            log("Kiwoom API 연결 중...")
            self.kiwoom.CommConnect(block=True)
            log("로그인 완료.")
            return True
        except Exception as e:
            log(f"Kiwoom 연결 실패: {e}")
            return False

    def fetch(
        self,
        code: str,
        api_date_yyyymmdd: str,
        *,
        log: Callable[[str], None],
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """
        캐시 우선 로드, 실패 시 API 조회.
        반환: (df, cache_path)
        종목 이름은 self.stock_name 에 저장됨.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{code}_{api_date_yyyymmdd}.pkl"

        df: Optional[pd.DataFrame] = None
        if cache_path.exists():
            try:
                log(f"캐시 발견: {cache_path}")
                df = pd.read_pickle(cache_path)
            except Exception as e:
                log(f"캐시 로드 실패, API 재조회 시도: {e}")

        # 종목 이름 조회 (캐시/API 공통)
        try:
            if self.kiwoom is None:
                self.connect(log)
            name = self.kiwoom.GetMasterCodeName(code) if self.kiwoom else ""
            self.stock_name = name.strip() if name else code
            if name:
                log(f"종목 이름: {self.stock_name}")
        except Exception:
            self.stock_name = code

        if df is None:
            if not self.connect(log):
                return None, None

            log(f"API 조회일: {api_date_yyyymmdd[:4]}-{api_date_yyyymmdd[4:6]}-{api_date_yyyymmdd[6:]}")

            df = mykiwoom.query_stock_data(
                self.kiwoom, code, api_date_yyyymmdd, progress_callback=progress_callback
            )
        
            if df is None or df.empty:
                log("조회된 자료가 없습니다.")
                return None, None

            try:
                df.to_pickle(cache_path)
                log(f"데이터 캐시 완료: {cache_path}")
            except Exception as e:
                log(f"캐시 저장 실패(무시 가능): {e}")

        required = [
            "개인투자자", "외국인투자자", "기관계", "금융투자", "보험", "투신", "기타금융", "은행",
            "연기금등", "사모펀드", "국가", "기타법인", "내외국인", "일자", "현재가"
        ]
        general.validate_dataframe(df, required)
        return df, cache_path

    def fetch_avg_price(
        self,
        code: str,
        api_date_yyyymmdd: str,
        *,
        log: Callable[[str], None],
        progress_callback: Optional[Callable] = None,
    ) -> Optional[pd.DataFrame]:
        """
        투자자별 추정 평균가 조회.
        매수금액(금액수량구분=1, 매매구분=1)과 매수수량(금액수량구분=2, 매매구분=1)을
        각각 조회한 뒤, 날짜별 누적비율로 추정 평균가를 계산합니다.
        결과 컬럼: 개인, 외국인, 기관, ... (DatetimeIndex)

        ⚠️ API 2회 호출로 시간이 소요됩니다. 캐시 이후에는 즉시 반환합니다.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{code}_{api_date_yyyymmdd}_avg_price.pkl"

        if cache_path.exists():
            try:
                log(f"추정 평균가 캐시 발견: {cache_path}")
                return pd.read_pickle(cache_path)
            except Exception as e:
                log(f"캐시 로드 실패, 재조회: {e}")

        if not self.connect(log):
            return None

        log("추정 평균가 조회 ① 매수금액 조회 중...")
        amount_df = mykiwoom.query_investor_tr(
            self.kiwoom, code, api_date_yyyymmdd,
            amount_qty_div=1, trade_div=1,
            progress_callback=progress_callback,
        )

        log("추정 평균가 조회 ② 매수수량 조회 중...")
        qty_df = mykiwoom.query_investor_tr(
            self.kiwoom, code, api_date_yyyymmdd,
            amount_qty_div=2, trade_div=1,
            progress_callback=progress_callback,
        )

        if amount_df.empty or qty_df.empty:
            log("추정 평균가: 데이터를 가져오지 못했습니다.")
            return None

        avg_df = _compute_estimated_avg_price(amount_df, qty_df)

        try:
            avg_df.to_pickle(cache_path)
            log(f"추정 평균가 캐시 완료: {cache_path}")
        except Exception as e:
            log(f"캐시 저장 실패(무시): {e}")

        return avg_df
