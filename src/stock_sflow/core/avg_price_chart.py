# =============================================================================
# 목적  : 투자자별 추정 평균가 시계열 차트
# 설명  : 현재가와 투자자별 추정 평균가(누적매수금액/누적매수수량)를 동일 가격 축에 표시.
#         현재가 선이 특정 투자자의 추정 평균가를 상회하면 해당 투자자 수익 구간.
# =============================================================================
from __future__ import annotations

from typing import Optional, List
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QCheckBox, QScrollBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import pyqtgraph as pg


pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

PRICE_COLOR = "#2C3E50"
PRICE_WIDTH = 2.0

COL_COLORS: dict[str, str] = {
    "개인":    "#E74C3C",
    "외국인":  "#3498DB",
    "기관":    "#2ECC71",
    "금융투자": "#F39C12",
    "보험":    "#9B59B6",
    "투신":    "#1ABC9C",
    "기타금융": "#E67E22",
    "은행":    "#34495E",
    "연기금":  "#27AE60",
    "사모펀드": "#8E44AD",
    "국가":    "#2980B9",
    "기타법인": "#D35400",
    "내외국인": "#C0392B",
}

COL_WIDTHS: dict[str, float] = {col: 1.5 for col in COL_COLORS}
COL_WIDTHS.update({"개인": 2.0, "외국인": 2.0, "기관": 2.0})

DEFAULT_VISIBLE = ["개인", "외국인", "기관"]
SCROLL_RESOLUTION = 10000


class AvgPriceChartWidget(QWidget):
    """
    투자자별 추정 평균가 + 현재가를 단일 가격 축(원/주)에 표시.

    레이아웃:
    ┌─────────────────────────────────────────────────────┐
    │  ☑개인 ☑외국인 ☑기관 ☐금융투자 ...  ☑현재가       │ ← 체크박스
    ├─────────────────────────────────────────────────────┤
    │                    차트 (가격 원/주)                  │
    ├─────────────────────────────────────────────────────┤
    │  ←────────────── 스크롤바 ─────────────────→         │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        avg_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        title: str = "추정 평균가",
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.avg_df = avg_df
        self.price_df = price_df
        self.title = title
        self._default_visible = default_visible if default_visible is not None else DEFAULT_VISIBLE

        self._checkboxes: dict[str, QCheckBox] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._price_curve: Optional[pg.PlotDataItem] = None
        self._cb_price: Optional[QCheckBox] = None

        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._scrollbar_updating: bool = False

        self._build_ui()
        self._plot_all()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(36)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        cb_container = QWidget()
        cb_layout = QHBoxLayout(cb_container)
        cb_layout.setContentsMargins(2, 0, 2, 0)
        cb_layout.setSpacing(10)

        small_font = QFont()
        small_font.setPointSize(8)

        for col in self.avg_df.columns:
            cb = QCheckBox(col)
            cb.setFont(small_font)
            cb.setChecked(col in self._default_visible)
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[col] = cb
            cb_layout.addWidget(cb)

        if self.price_df is not None:
            self._cb_price = QCheckBox("현재가")
            self._cb_price.setFont(small_font)
            self._cb_price.setChecked(True)
            self._cb_price.stateChanged.connect(self._on_price_checkbox_changed)
            cb_layout.addWidget(self._cb_price)

        cb_layout.addStretch()
        scroll.setWidget(cb_container)
        root.addWidget(scroll)

        date_axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(
            axisItems={"bottom": date_axis},
            title=f"<b>{self.title}</b>",
        )
        self.plot_widget.setLabel("left", "가격 (원/주)")
        self.plot_widget.setLabel("bottom", "날짜")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.plot_widget.plotItem.vb.setMouseEnabled(x=True, y=False)
        root.addWidget(self.plot_widget)

        self._scrollbar = QScrollBar(Qt.Horizontal)
        self._scrollbar.setMinimum(0)
        self._scrollbar.setMaximum(SCROLL_RESOLUTION)
        self._scrollbar.setValue(SCROLL_RESOLUTION)
        self._scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        root.addWidget(self._scrollbar)

    def _ts(self, df: pd.DataFrame) -> np.ndarray:
        return df.index.astype("int64") // 10 ** 9

    def _plot_all(self):
        self.plot_widget.clear()
        self._curves.clear()
        self._price_curve = None

        ts_avg = self._ts(self.avg_df)
        all_ts = [ts_avg]
        if self.price_df is not None:
            all_ts.append(self._ts(self.price_df))
        combined = np.concatenate(all_ts)
        if len(combined) > 0:
            self._x_min = float(combined.min())
            self._x_max = float(combined.max())

        # 투자자별 추정 평균가 커브
        for col, cb in self._checkboxes.items():
            if col not in self.avg_df.columns:
                continue
            color = COL_COLORS.get(col, "#808080")
            width = COL_WIDTHS.get(col, 1.5)
            pen = pg.mkPen(color=color, width=width)
            vals = pd.to_numeric(self.avg_df[col], errors="coerce").to_numpy()
            curve = self.plot_widget.plot(x=ts_avg, y=vals, pen=pen, connect="finite")
            curve.setVisible(cb.isChecked())
            self._curves[col] = curve

        # 현재가 커브 (굵은 검정 실선)
        if self.price_df is not None:
            price_col = self.price_df.columns[0]
            price_ts = self._ts(self.price_df)
            price_vals = pd.to_numeric(
                self.price_df[price_col], errors="coerce"
            ).fillna(0).to_numpy()
            pen_price = pg.mkPen(color=PRICE_COLOR, width=PRICE_WIDTH)
            self._price_curve = self.plot_widget.plot(
                x=price_ts, y=price_vals, pen=pen_price
            )
            if self._cb_price is not None:
                self._price_curve.setVisible(self._cb_price.isChecked())

        self._update_scrollbar_range()
        if self._x_max > self._x_min:
            self.plot_widget.plotItem.vb.setXRange(self._x_min, self._x_max, padding=0)

    def _update_scrollbar_range(self):
        if self._x_max <= self._x_min:
            return
        vb = self.plot_widget.plotItem.vb
        x_range = vb.viewRange()[0]
        view_width = x_range[1] - x_range[0]
        total_width = self._x_max - self._x_min
        if total_width <= 0:
            return
        page_step = max(1, int(SCROLL_RESOLUTION * view_width / total_width))
        self._scrollbar.setPageStep(page_step)
        self._scrollbar.setMaximum(SCROLL_RESOLUTION - page_step)

    def _on_x_range_changed(self, vb, x_range):
        if self._scrollbar_updating or self._x_max <= self._x_min:
            return
        self._scrollbar_updating = True
        try:
            self._update_scrollbar_range()
            total_width = self._x_max - self._x_min
            ratio = (x_range[0] - self._x_min) / total_width
            new_val = max(0, min(int(ratio * SCROLL_RESOLUTION), self._scrollbar.maximum()))
            self._scrollbar.setValue(new_val)
        finally:
            self._scrollbar_updating = False

    def _on_scrollbar_changed(self, value: int):
        if self._scrollbar_updating or self._x_max <= self._x_min:
            return
        total_width = self._x_max - self._x_min
        vb = self.plot_widget.plotItem.vb
        view_width = vb.viewRange()[0][1] - vb.viewRange()[0][0]
        new_left = self._x_min + (value / SCROLL_RESOLUTION) * total_width
        new_right = new_left + view_width
        if new_right > self._x_max:
            new_right = self._x_max
            new_left = new_right - view_width
        self._scrollbar_updating = True
        vb.setXRange(new_left, new_right, padding=0)
        self._scrollbar_updating = False

    def _on_checkbox_changed(self):
        for col, cb in self._checkboxes.items():
            if col in self._curves:
                self._curves[col].setVisible(cb.isChecked())

    def _on_price_checkbox_changed(self):
        if self._price_curve is not None and self._cb_price is not None:
            self._price_curve.setVisible(self._cb_price.isChecked())

    def update_df(
        self,
        avg_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
    ):
        self.avg_df = avg_df
        if price_df is not None:
            self.price_df = price_df
        self._plot_all()


class AvgPriceChartWindow(QMainWindow):
    def __init__(
        self,
        avg_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        title: str = "추정 평균가",
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1600, 650)
        self._widget = AvgPriceChartWidget(
            avg_df=avg_df,
            price_df=price_df,
            title=title,
            default_visible=default_visible,
        )
        self.setCentralWidget(self._widget)

    def update_data(
        self,
        avg_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
    ):
        self._widget.update_df(avg_df, price_df=price_df)


def show_avg_price_chart(
    avg_df: pd.DataFrame,
    *,
    price_df: Optional[pd.DataFrame] = None,
    title: str = "추정 평균가",
    default_visible: Optional[List[str]] = None,
    parent: Optional[QWidget] = None,
) -> AvgPriceChartWindow:
    return AvgPriceChartWindow(
        avg_df=avg_df,
        price_df=price_df,
        title=title,
        default_visible=default_visible,
        parent=parent,
    )
