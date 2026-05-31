# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 분석 파이프라인 분리 (전처리/슬라이스/지표/비율/머지)
# 주요 기능 : core.process / core.ratio 호출을 한 곳에서 관리
# 요구 사항 : pandas 
# 작성자    : (작성자명 기입)
# 최종수정  : 2025-12-19
# 버전      : 1.0.0
# =============================================================================

# =============================== 표준 라이브러리 ==============================
from dataclasses import dataclass
from typing import Optional, List

# =============================== 서드파티 라이브러리 ==========================
import pandas as pd

# =============================== 프로젝트 내부 모듈 ===========================
from stock_sflow.core import process
from stock_sflow.core import ratio


@dataclass
class AnalysisResult:
    sliced: pd.DataFrame
    price: pd.DataFrame
    cumsum: pd.DataFrame
    acc: pd.DataFrame
    high: pd.DataFrame
    drv: pd.DataFrame 
    hold: pd.DataFrame  
    sprd: pd.DataFrame
    rdf: pd.DataFrame
    price_merged_sprd: pd.DataFrame
    price_merged_cumsum: pd.DataFrame
    right_cols: List[str]


class AnalysisService:
    def run(
        self,
        raw_df: pd.DataFrame,
        raw_start_yyyymmdd: str,
        raw_end_yyyymmdd: str,
    ) -> AnalysisResult:
        added = process.preprocess_data(raw_df)
        # raw_df.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\raw_df.xlsx")
        
        sliced = process.slice_data_by_date(
            added,
            raw_start_yyyymmdd,
            raw_end_yyyymmdd,
            use_nearest=True,
            descending=False,  # 계산용 오름차순 명시
        ).sort_index(ascending=True)       # ✅ 서비스 레벨에서 한 번 더 고정
        # sliced.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\sliced.xlsx")
        
        price = process.get_price(sliced)
  
        # ✅ 표준 컬럼 재사용 (중복/누락 방지)
        cols = list(ratio.FINAL_COLUMNS)
   
        cumsum = process.get_total_cumsum(sliced, cols)
        # cumsum.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\cumsum.xlsx")
        low = process.get_lowest(cumsum)
        #low.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\low.xlsx")
        acc = process.get_accumulation(cumsum, low)
        #acc.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\acc.xlsx")
        high = process.get_highest(acc)
        #high.to_excel(r"C:\Users\hushh\Desktop\python\stock_sflow\high.xlsx")
        
        drv = ratio.get_driving_ratio(high)
        hold = ratio.get_holding_ratio(acc)
        sprd = ratio.get_spread_ratio(acc, high)
        rdf = ratio.create_ratio_df(drv, hold, sprd)
   
        price_merged_sprd = process.merge_with_price(price, sprd)
        price_merged_cumsum = process.merge_with_price(price, cumsum)

        # ✅ "가격 컬럼 제외 = 지표 컬럼"을 정확히 분리
        right_cols = [c for c in price_merged_sprd.columns if c not in price.columns]

        return AnalysisResult(
            sliced=sliced,
            price=price,
            cumsum=cumsum,
            acc=acc,
            high=high,
            drv=drv,     
            hold=hold,
            sprd=sprd,
            rdf=rdf,
            price_merged_sprd=price_merged_sprd,
            price_merged_cumsum=price_merged_cumsum,
            right_cols=right_cols,
        )

