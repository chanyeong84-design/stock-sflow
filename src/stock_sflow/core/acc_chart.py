# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 컬럼별 매집수량을 체크박스로 선택해 시각화, 가격 우측 2축 오버레이
# 주요 기능 : 투자자 체크박스 필터, 왼쪽 Y축(매집수량) / 오른쪽 Y축(가격+추정평균가MA)
# 요구 사항 : Python 3.9+ (32비트 호환), PyQt5, pyqtgraph, pandas
# 최종수정  : 2026-06-01
# 버전      : 1.1.0
# =============================================================================

from __future__ import annotations

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QCheckBox, QScrollBar, QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import pyqtgraph as pg


# =============================================================================
# 상수
# =============================================================================

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

# 가격선 색상 / 두께
PRICE_COLOR = "#2C3E50"
PRICE_WIDTH = 1.5

# 추정 평균가 이동평균 설정
MA_PERIODS = [5, 20, 60, 240]
MA_LABELS  = {5: "5일MA", 20: "20일MA", 60: "60일MA", 240: "240일MA"}
MA_STYLES  = {5: Qt.SolidLine, 20: Qt.DashLine, 60: Qt.DotLine, 240: Qt.DashDotLine}
MA_WIDTHS  = {5: 1.2, 20: 1.5, 60: 1.8, 240: 2.0}
MA_DEFAULT_CHECKED = {5: False, 20: False, 60: False, 240: False}

# 기본 체크 컬럼
DEFAULT_VISIBLE = ["개인", "외국인", "기관"]

# 컬럼별 색상
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
    "세력_1":  "#16A085",
    "세력_2":  "#7F8C8D",
}

# 컬럼별 두께
COL_WIDTHS: dict[str, float] = {
    "개인":    2.0,
    "외국인":  2.0,
    "기관":    2.0,
    "금융투자": 1.5,
    "보험":    1.5,
    "투신":    1.5,
    "기타금융": 1.5,
    "은행":    1.5,
    "연기금":  1.5,
    "사모펀드": 1.5,
    "국가":    1.5,
    "기타법인": 1.5,
    "내외국인": 1.5,
    "세력_1":  2.0,
    "세력_2":  1.5,
}

SCROLL_RESOLUTION = 10000


# =============================================================================
# 메인 위젯
# =============================================================================
class AccChartWidget(QWidget):
    """
    매집수량 + 가격 + 투자자별 추정평균가 이동평균 차트.

    레이아웃:
    ┌──────────────────────────────────────────────────────────────┐
    │  ☑개인 ☑외국인 ☑기관 ...  │ ☑가격 │ ☐5일MA ☐20일MA ...   │ ← 체크박스
    ├──────────────────────────────────────────────────────────────┤
    │ 매집수량 │              차트                │ 가격(+MA)      │
    ├──────────────────────────────────────────────────────────────┤
    │  ←──────────────────── 스크롤바 ──────────────────────→      │
    └──────────────────────────────────────────────────────────────┘
    왼쪽 Y축 : 매집수량
    오른쪽 Y축: 현재가 + 투자자별 추정평균가 이동평균(MA)
    MA 색상   : 해당 투자자 색상과 동일, 기간별로 선 스타일 구분
    """

    def __init__(
        self,
        acc_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        avg_df: Optional[pd.DataFrame] = None,
        title: str = "매집수량",
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.acc_df = acc_df
        self.price_df = price_df
        self.avg_df = avg_df
        self.title = title
        self._default_visible = default_visible if default_visible is not None else DEFAULT_VISIBLE

        self._checkboxes: dict[str, QCheckBox] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._price_curve: Optional[pg.PlotDataItem] = None
        self._price_vb: Optional[pg.ViewBox] = None
        self._cb_price: Optional[QCheckBox] = None
        # (investor_col, period) → PlotDataItem
        self._ma_curves: dict[Tuple[str, int], pg.PlotDataItem] = {}
        self._cb_ma_investors: dict[str, QCheckBox] = {}  # MA 투자자별 선택
        self._cb_ma_periods: dict[int, QCheckBox] = {}    # MA 기간 선택

        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._scrollbar_updating: bool = False

        self._build_ui()
        self._plot_all()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        small_font = QFont()
        small_font.setPointSize(8)

        def _make_scroll_row() -> tuple:
            """스크롤 가능한 가로 체크박스 행을 반환 (scroll_area, h_layout)."""
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFixedHeight(34)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(2, 0, 2, 0)
            layout.setSpacing(8)
            sa.setWidget(container)
            return sa, layout

        # ── 1행: 매집수량 투자자 선택 + 가격 ──────────────────────────────
        row1_sa, row1 = _make_scroll_row()

        lbl1 = QLabel("매집:")
        lbl1.setFont(small_font)
        row1.addWidget(lbl1)

        for col in self.acc_df.columns:
            cb = QCheckBox(col)
            cb.setFont(small_font)
            cb.setChecked(col in self._default_visible)
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[col] = cb
            row1.addWidget(cb)

        if self.price_df is not None:
            sep = QLabel("┃")
            sep.setFont(small_font)
            row1.addWidget(sep)
            self._cb_price = QCheckBox("가격")
            self._cb_price.setFont(small_font)
            self._cb_price.setChecked(True)
            self._cb_price.stateChanged.connect(self._on_price_checkbox_changed)
            row1.addWidget(self._cb_price)

        row1.addStretch()
        root.addWidget(row1_sa)

        # ── 2행: MA 투자자 선택 + MA 기간 선택 ───────────────────────────
        if self.avg_df is not None and not self.avg_df.empty:
            row2_sa, row2 = _make_scroll_row()

            lbl2 = QLabel("MA:")
            lbl2.setFont(small_font)
            row2.addWidget(lbl2)

            for col in self.avg_df.columns:
                cb = QCheckBox(col)
                cb.setFont(small_font)
                cb.setChecked(col in self._default_visible)
                cb.stateChanged.connect(self._on_ma_investor_changed)
                self._cb_ma_investors[col] = cb
                row2.addWidget(cb)

            sep2 = QLabel("┃")
            sep2.setFont(small_font)
            row2.addWidget(sep2)

            for period in MA_PERIODS:
                cb = QCheckBox(MA_LABELS[period])
                cb.setFont(small_font)
                cb.setChecked(MA_DEFAULT_CHECKED[period])
                cb.stateChanged.connect(self._on_ma_period_changed)
                self._cb_ma_periods[period] = cb
                row2.addWidget(cb)

            row2.addStretch()
            root.addWidget(row2_sa)

        # 메인 차트 (왼쪽 Y축: 매집수량)
        date_axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(
            axisItems={"bottom": date_axis},
            title=f"<b>{self.title}</b>",
        )
        self.plot_widget.setLabel("left", "매집수량")
        self.plot_widget.setLabel("bottom", "날짜")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.plot_widget.plotItem.vb.setMouseEnabled(x=True, y=False)

        # 우측 Y축 ViewBox (현재가 + MA)
        if self.price_df is not None:
            self._price_vb = pg.ViewBox()
            self.plot_widget.plotItem.vb.scene().addItem(self._price_vb)

            right_axis = pg.AxisItem("right")
            right_axis.setLabel("가격 (원)")
            self.plot_widget.plotItem.layout.addItem(right_axis, 2, 3)
            right_axis.linkToView(self._price_vb)

            self._price_vb.setXLink(self.plot_widget.plotItem.vb)
            self.plot_widget.plotItem.vb.sigResized.connect(self._update_price_vb_geometry)

        root.addWidget(self.plot_widget)

        self._scrollbar = QScrollBar(Qt.Horizontal)
        self._scrollbar.setMinimum(0)
        self._scrollbar.setMaximum(SCROLL_RESOLUTION)
        self._scrollbar.setValue(SCROLL_RESOLUTION)
        self._scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        root.addWidget(self._scrollbar)

    def _update_price_vb_geometry(self):
        if self._price_vb is not None:
            self._price_vb.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())

    # --------------------------------------------------------------- 차트 --
    def _timestamps(self) -> np.ndarray:
        return self.acc_df.index.astype("int64") // 10 ** 9

    def _plot_all(self):
        self.plot_widget.clear()
        self._curves.clear()
        self._price_curve = None
        self._ma_curves.clear()

        if self._price_vb is not None:
            self._price_vb.clear()

        ts = self._timestamps()

        all_ts = [ts]
        if self.price_df is not None:
            all_ts.append(self.price_df.index.astype("int64") // 10 ** 9)
        combined = np.concatenate(all_ts)
        if len(combined) > 0:
            self._x_min = float(combined.min())
            self._x_max = float(combined.max())

        # 매집수량 커브 (왼쪽 축)
        for col, cb in self._checkboxes.items():
            if col not in self.acc_df.columns:
                continue
            color = COL_COLORS.get(col, "#808080")
            width = COL_WIDTHS.get(col, 1.5)
            pen = pg.mkPen(color=color, width=width)
            vals = pd.to_numeric(self.acc_df[col], errors="coerce").fillna(0).to_numpy()
            curve = self.plot_widget.plot(x=ts, y=vals, pen=pen)
            curve.setVisible(cb.isChecked())
            self._curves[col] = curve

        if self.price_df is not None and self._price_vb is not None:
            # 현재가 커브
            price_col = self.price_df.columns[0]
            price_ts = self.price_df.index.astype("int64") // 10 ** 9
            price_vals = pd.to_numeric(
                self.price_df[price_col], errors="coerce"
            ).fillna(0).to_numpy()
            pen_price = pg.mkPen(color=PRICE_COLOR, width=PRICE_WIDTH)
            self._price_curve = pg.PlotDataItem(x=price_ts, y=price_vals, pen=pen_price)
            self._price_vb.addItem(self._price_curve)
            if self._cb_price is not None:
                self._price_curve.setVisible(self._cb_price.isChecked())

            # 투자자별 추정 평균가 이동평균 커브
            if self.avg_df is not None and not self.avg_df.empty:
                avg_ts = self.avg_df.index.astype("int64") // 10 ** 9
                for col, cb_inv in self._checkboxes.items():
                    if col not in self.avg_df.columns:
                        continue
                    color = COL_COLORS.get(col, "#808080")
                    series = pd.to_numeric(self.avg_df[col], errors="coerce")
                    cb_ma_inv = self._cb_ma_investors.get(col)
                    for period, cb_period in self._cb_ma_periods.items():
                        ma_vals = series.rolling(window=period, min_periods=1).mean().to_numpy()
                        pen_ma = pg.mkPen(
                            color=color,
                            width=MA_WIDTHS[period],
                            style=MA_STYLES[period],
                        )
                        curve = pg.PlotDataItem(
                            x=avg_ts.to_numpy(), y=ma_vals, pen=pen_ma, connect="finite"
                        )
                        self._price_vb.addItem(curve)
                        visible = (
                            cb_ma_inv is not None and cb_ma_inv.isChecked()
                            and cb_period.isChecked()
                        )
                        curve.setVisible(visible)
                        self._ma_curves[(col, period)] = curve

        self._update_price_vb_geometry()
        self._update_scrollbar_range()
        if self._x_max > self._x_min:
            self.plot_widget.plotItem.vb.setXRange(self._x_min, self._x_max, padding=0)

    # --------------------------------------------------------- 스크롤바 연동 --
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

    # -------------------------------------------------------- 이벤트 핸들러 --
    def _on_checkbox_changed(self):
        # 매집수량 선만 제어 (MA는 별도 체크박스로 제어)
        for col, cb in self._checkboxes.items():
            if col in self._curves:
                self._curves[col].setVisible(cb.isChecked())

    def _on_price_checkbox_changed(self):
        if self._price_curve is not None and self._cb_price is not None:
            self._price_curve.setVisible(self._cb_price.isChecked())

    def _on_ma_investor_changed(self):
        # 투자자 MA 체크박스 변경 → 해당 투자자의 모든 MA 기간 업데이트
        for col, cb_ma_inv in self._cb_ma_investors.items():
            for period, cb_period in self._cb_ma_periods.items():
                curve = self._ma_curves.get((col, period))
                if curve is not None:
                    curve.setVisible(cb_ma_inv.isChecked() and cb_period.isChecked())

    def _on_ma_period_changed(self):
        # MA 기간 체크박스 변경 → 해당 기간의 모든 투자자 업데이트
        for period, cb_period in self._cb_ma_periods.items():
            for col, cb_ma_inv in self._cb_ma_investors.items():
                curve = self._ma_curves.get((col, period))
                if curve is not None:
                    curve.setVisible(cb_ma_inv.isChecked() and cb_period.isChecked())

    # ----------------------------------------------------------- 데이터 교체 --
    def update_df(
        self,
        acc_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        avg_df: Optional[pd.DataFrame] = None,
    ):
        self.acc_df = acc_df
        if price_df is not None:
            self.price_df = price_df
        if avg_df is not None:
            self.avg_df = avg_df
        self._plot_all()


# =============================================================================
# 메인 창
# =============================================================================
class AccChartWindow(QMainWindow):
    def __init__(
        self,
        acc_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        avg_df: Optional[pd.DataFrame] = None,
        title: str = "매집수량",
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1600, 650)
        self._widget = AccChartWidget(
            acc_df=acc_df,
            price_df=price_df,
            avg_df=avg_df,
            title=title,
            default_visible=default_visible,
        )
        self.setCentralWidget(self._widget)

    def update_data(
        self,
        acc_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        avg_df: Optional[pd.DataFrame] = None,
    ):
        self._widget.update_df(acc_df, price_df=price_df, avg_df=avg_df)


# =============================================================================
# 진입 함수
# =============================================================================
def show_acc_chart(
    acc_df: pd.DataFrame,
    *,
    price_df: Optional[pd.DataFrame] = None,
    avg_df: Optional[pd.DataFrame] = None,
    title: str = "매집수량",
    default_visible: Optional[List[str]] = None,
    parent: Optional[QWidget] = None,
) -> AccChartWindow:
    return AccChartWindow(
        acc_df=acc_df,
        price_df=price_df,
        avg_df=avg_df,
        title=title,
        default_visible=default_visible,
        parent=parent,
    )
