# =============================================================================
# 수급 분석표 생성/표시 (PyQt5 + pandas)
# - 일/주/월/분기/연 집계 + 현재/최대 보유량 행 추가
# - 숫자화/결측/문자 혼입 안전 처리
# - 배경 밴딩/양수(빨강)/음수(파랑) 컬러링, 평균단가 정수 포맷
# - 기존 UI 느낌(고정폭 + 여유폭) 유지
# =============================================================================

from __future__ import annotations
import sys
from typing import List, Sequence

import numpy as np
import pandas as pd

try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout
    )
    from PyQt5.QtGui import QColor, QBrush
    from PyQt5.QtCore import Qt
    _PYQT5_AVAILABLE = True
except ImportError:
    QWidget = object  # 웹앱 환경에서 클래스 정의가 실패하지 않도록
    _PYQT5_AVAILABLE = False


# 기대 컬럼(순서 보장용). 누락되면 NaN 컬럼으로 자동 채움
EXPECTED_COLUMNS: List[str] = [
    "일자", "평균단가", "개인", "외국인", "기관", "금융투자", "보험",
    "투신", "기타금융", "은행", "연기금", "사모펀드",
    "국가", "기타법인", "내외국인", "세력_1", "세력_2",
]


# ================================ 유틸 함수 ================================

def _ensure_expected_columns(df: pd.DataFrame, expected: Sequence[str]) -> pd.DataFrame:
    """기대 컬럼이 없으면 NaN으로 추가하고, 컬럼 순서를 기대 순서로 맞춘다."""
    out = df.copy()
    for c in expected:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(expected)]


def _to_numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """지정 컬럼들을 숫자형(coerce)로 변환(일자는 제외)."""
    out = df.copy()
    for c in cols:
        if c in out.columns and c != "일자":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _safe_slice_iloc(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """iloc 슬라이스 안전 처리(범위 자동 클램프)."""
    start = max(0, start)
    end = max(0, min(len(df), end))
    if start >= end:
        return df.iloc[0:0].copy()
    return df.iloc[start:end].copy()


# ================================ 집계 함수 ================================

def aggregate_block(df: pd.DataFrame, start: int, end: int, label: str) -> dict:
    """
    블록 집계: '평균단가'는 평균, 나머지 수치 컬럼은 합계.
    - 범위 초과/빈 블록도 안전 처리
    - 숫자화(coerce) 후 집계
    """
    block = _safe_slice_iloc(df, start, end)
    agg_row = {"일자": label}
    if block.empty:
        agg_row["평균단가"] = np.nan
        for col in df.columns[2:]:
            agg_row[col] = np.nan
        return agg_row

    num_block = _to_numeric_cols(block, block.columns)
    agg_row["평균단가"] = num_block["평균단가"].mean(skipna=True) if "평균단가" in num_block.columns else np.nan
    for col in num_block.columns:
        if col in ("일자", "평균단가"):
            continue
        agg_row[col] = num_block[col].sum(skipna=True)
    return agg_row


def _build_period_df(main_df: pd.DataFrame, days_per_period: int, n_periods: int, label_suffix: str) -> pd.DataFrame:
    """규칙적 블록 집계 DataFrame 생성(데이터 부족 시 남는 블록은 스킵)."""
    rows = []
    for i in range(n_periods):
        s = i * days_per_period
        e = (i + 1) * days_per_period
        if s >= len(main_df):
            break
        
        # # --- 검증용 코드 시작 ---
        # subset = main_df.iloc[s:e]
        # if not subset.empty:
        #     print(f"[{i+1}{label_suffix}] 구간: {subset['일자'].iloc[0]} ~ {subset['일자'].iloc[-1]} (행수: {len(subset)})")
        # # --- 검증용 코드 끝 ---
        
        rows.append(aggregate_block(main_df, s, e, f"{i + 1}{label_suffix}"))
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return _ensure_expected_columns(res, EXPECTED_COLUMNS)


def create_row_dict(row_series: pd.Series, label: str, expected_columns: Sequence[str]) -> dict:
    """단일 행 시리즈를 라벨 포함 dict로 매핑(없는 컬럼은 NaN, 평균단가는 없음)."""
    row_dict = {}
    for col in expected_columns:
        if col == "일자":
            row_dict[col] = label
        elif col == "평균단가":
            row_dict[col] = np.nan
        else:
            row_dict[col] = pd.to_numeric(row_series.get(col, np.nan), errors="coerce")
    return row_dict


# ================================ 메인 파이프 ================================

def process_data(main_df: pd.DataFrame, accumulation_df: pd.DataFrame, highest_df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 데이터프레임(main_df, accumulation_df, highest_df)을 전처리 및 집계하여
    최종 데이터프레임(df_final)을 반환.
      1) main_df: 인덱스 복구 → '현재가'를 '평균단가'로 리네이밍 → 기대 컬럼 강제 → 숫자화
      2) 일/주/월/분기/연 집계(5/20/60/240일 단위, 남는 블록은 자동 스킵)
      3) accumulation_df, highest_df 첫 행으로 '현재 보유량', '최대 보유량' 추가
    """
    if main_df.empty:
        raise ValueError("main_df가 비어 있습니다.")
    if accumulation_df.empty or highest_df.empty:
        raise ValueError("accumulation_df 또는 highest_df가 비어 있습니다.")

    # 1) 메인 데이터
    m = main_df.copy()
    m = m.reset_index().rename(columns={"현재가": "평균단가"})
    m = _ensure_expected_columns(m, EXPECTED_COLUMNS)
    m = _to_numeric_cols(m, m.columns)

    # 일일: 최근 5행 미리보기(주/월 등과 구분용)
    daily_df = m.head(5).copy()

    # 2) 블록 집계
    weekly_df    = _build_period_df(m, days_per_period=5,   n_periods=4,  label_suffix="주")
    monthly_df   = _build_period_df(m, days_per_period=20,  n_periods=4,  label_suffix="달")
    quarterly_df = _build_period_df(m, days_per_period=60,  n_periods=4,  label_suffix="분기")
    yearly_df    = _build_period_df(m, days_per_period=240, n_periods=10, label_suffix="년")

    parts = [df for df in [daily_df, weekly_df, monthly_df, quarterly_df, yearly_df] if df is not None and not df.empty]
    df_final = pd.concat(parts, ignore_index=True) if parts else daily_df.copy()

    # 3) 현재/최대 보유량 행(첫 행 기준)
    row_current = accumulation_df.iloc[0]
    row_max = highest_df.iloc[0]

    current_hold_dict = create_row_dict(row_current, "현재 보유량", EXPECTED_COLUMNS)
    max_hold_dict = create_row_dict(row_max, "최대 보유량", EXPECTED_COLUMNS)
    new_rows = pd.DataFrame([current_hold_dict, max_hold_dict], columns=EXPECTED_COLUMNS)
    new_rows = new_rows.dropna(axis=1, how='all')

    df_final = pd.concat([df_final, new_rows], ignore_index=True)
    return df_final


# ================================ 테이블 UI ================================

class TableWindow(QWidget):
    def __init__(self, dataframe: pd.DataFrame, title: str = "수급 분석표"):
        super().__init__()
        self.df = dataframe
        self._title = title
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self._title)
        layout = QVBoxLayout(self)
        table = QTableWidget(self)

        n_rows, n_cols = len(self.df), len(self.df.columns)
        table.setRowCount(n_rows)
        table.setColumnCount(n_cols)
        table.setHorizontalHeaderLabels([str(c) for c in self.df.columns.tolist()])

        # 구간별 배경색 (라벨 패턴으로 그룹 구분)
        labels = self.df["일자"].astype(str)
        num_daily     = (labels.str.contains("-", na=False)).sum()
        num_weekly    = (labels.str.contains("주", na=False)).sum()
        num_monthly   = (labels.str.contains("달", na=False)).sum()
        num_quarterly = (labels.str.contains("분기", na=False)).sum()
        num_yearly    = (labels.str.contains("년", na=False)).sum()

        start_daily     = 0
        start_weekly    = start_daily + num_daily
        start_monthly   = start_weekly + num_weekly
        start_quarterly = start_monthly + num_monthly
        start_yearly    = start_quarterly + num_quarterly
        start_extra     = start_yearly + num_yearly

        color_daily     = QColor("lightgray")
        color_weekly    = QColor("lightyellow")
        color_monthly   = QColor("lightblue")
        color_quarterly = QColor("lightgreen")
        color_yearly    = QColor("lightpink")
        color_extra     = QColor("lightcyan")

        for row in range(n_rows):
            if row < start_weekly:
                bg_color = color_daily
            elif row < start_monthly:
                bg_color = color_weekly
            elif row < start_quarterly:
                bg_color = color_monthly
            elif row < start_yearly:
                bg_color = color_quarterly
            elif row < start_extra:
                bg_color = color_yearly
            else:
                bg_color = color_extra

            for col in range(n_cols):
                value = self.df.iat[row, col]
                col_name = self.df.columns[col]

                # 표출 문자열 구성
                if col_name == "일자":
                    # datetime → 'YYYY-MM-DD', 그 외 문자열은 그대로
                    try:
                        text = pd.to_datetime(value).strftime("%Y-%m-%d")
                    except Exception:
                        text = str(value)
                elif col_name == "평균단가":
                    try:
                        num = float(value)
                        text = "" if np.isnan(num) else f"{int(round(num)):,}"
                    except Exception:
                        text = ""
                else:
                    # 일반 수치: 천단위(정수면 0f), 소수 있으면 그대로
                    try:
                        num = float(value)
                        if np.isnan(num):
                            text = ""
                        else:
                            text = f"{num:,.0f}" if num.is_integer() else f"{num:,}"
                    except Exception:
                        text = str(value) if value is not None else ""

                item = QTableWidgetItem(text)

                # 컬러링: 평균단가는 검정, 일반 수치 양수=빨강, 음수=파랑
                if col_name == "평균단가":
                    item.setForeground(QBrush(QColor("black")))
                elif col_name != "일자":
                    try:
                        num = float(value)
                        if not np.isnan(num):
                            if num < 0:
                                item.setForeground(QBrush(QColor("blue")))
                            elif num > 0:
                                item.setForeground(QBrush(QColor("red")))
                    except Exception:
                        pass

                item.setBackground(QBrush(bg_color))
                table.setItem(row, col, item)

        # 폭/높이: 원 코드를 따라 ‘내용 맞춤 → 여유 20픽셀’ 방식
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        for col in range(table.columnCount()):
            current_width = table.columnWidth(col)
            table.setColumnWidth(col, current_width + 20)

        total_width = table.verticalHeader().width()
        for col in range(table.columnCount()):
            total_width += table.columnWidth(col)

        total_height = table.horizontalHeader().height()
        for row in range(table.rowCount()):
            total_height += table.rowHeight(row)

        self.resize(total_width + 50, total_height + 50)

        layout.addWidget(table)
        self.setLayout(layout)


# ============================== 외부 진입 함수 ==============================

def show_supply_analysis_table(
    main_df: pd.DataFrame,
    accumulation_df: pd.DataFrame,
    highest_df: pd.DataFrame,
    title: str = "수급 분석표",
) -> QWidget:
    """
    세 개의 데이터프레임(main_df, accumulation_df, highest_df)을 입력받아 전처리 후,
    수급 분석표 창(QWidget)을 반환합니다. (여기서는 QApplication을 생성하지 않음)
    """
    main_df = main_df.sort_index(ascending=False)
    accumulation_df = accumulation_df.sort_index(ascending=False)
    highest_df = highest_df.sort_index(ascending=False)

    processed_df = process_data(main_df, accumulation_df, highest_df)
    return TableWindow(processed_df, title=title)
