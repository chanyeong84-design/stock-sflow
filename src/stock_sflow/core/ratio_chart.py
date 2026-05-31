# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : driving / holding / spread 비율 DataFrame을 탭 차트로 시각화
# 주요 기능 : 3개 탭(주가선도/보유비중/분산추이), 투자자 체크박스 필터, Y축 0~100 고정
#             가격 우측 2축 오버레이, X축 가로 스크롤바
# 요구 사항 : Python 3.9+ (32비트 호환), PyQt5, pyqtgraph, pandas
# 최종수정  : 2026-03-13
# 버전      : 1.4.0
# =============================================================================

from __future__ import annotations

from typing import Optional, List
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QScrollArea, QCheckBox, QScrollBar,
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

TAB_LABELS = ["주가선도비율", "보유비중", "분산추이"]

DEFAULT_VISIBLE = ["개인", "외국인", "기관"]

# 가격선 색상 / 두께
PRICE_COLOR = "#2C3E50"
PRICE_WIDTH = 1.5

# 컬럼별 색상 (hex — 이 블록만 수정하면 전체 색상 변경)
COL_COLORS: dict[str, str] = {
    "개인":    "#E74C3C",  # 빨강
    "외국인":  "#3498DB",  # 파랑
    "기관":    "#2ECC71",  # 초록
    "금융투자": "#F39C12",  # 주황
    "보험":    "#9B59B6",  # 보라
    "투신":    "#1ABC9C",  # 청록
    "기타금융": "#E67E22",  # 진주황
    "은행":    "#34495E",  # 남색
    "연기금":  "#27AE60",  # 진초록
    "사모펀드": "#8E44AD",  # 진보라
    "국가":    "#2980B9",  # 진파랑
    "기타법인": "#D35400",  # 갈색
    "내외국인": "#C0392B",  # 진빨강
    "세력_1":  "#16A085",  # 청록
    "세력_2":  "#7F8C8D",  # 회색
}

# 컬럼별 두께 (이 블록만 수정하면 전체 두께 변경)
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
# 단일 탭 위젯
# =============================================================================
class _RatioTab(QWidget):
    """
    하나의 비율 DataFrame + 가격 DataFrame을 받아 pyqtgraph 라인 차트로 렌더링.

    레이아웃:
    ┌──────────────────────────────────────────────┐
    │  ☑개인 ☑외국인 ☑기관 ☐금융투자 ...  ☑가격  │ ← 체크박스
    ├──────────────────────────────────────────────┤
    │ 비율(%) │              차트           │ 가격 │
    ├──────────────────────────────────────────────┤
    │  ←──────────────── 스크롤바 ───────────────→ │
    └──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        df: pd.DataFrame,
        title: str,
        price_df: Optional[pd.DataFrame] = None,
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.df = df
        self.title = title
        self.price_df = price_df
        self._default_visible = default_visible if default_visible is not None else DEFAULT_VISIBLE
        self._checkboxes: dict[str, QCheckBox] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._price_curve: Optional[pg.PlotDataItem] = None
        self._price_vb: Optional[pg.ViewBox] = None
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

        # ---- 체크박스 ----
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

        for col in self.df.columns:
            cb = QCheckBox(col)
            cb.setFont(small_font)
            cb.setChecked(col in self._default_visible)
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[col] = cb
            cb_layout.addWidget(cb)

        if self.price_df is not None:
            self._cb_price = QCheckBox("가격")
            self._cb_price.setFont(small_font)
            self._cb_price.setChecked(True)
            self._cb_price.stateChanged.connect(self._on_price_checkbox_changed)
            cb_layout.addWidget(self._cb_price)

        cb_layout.addStretch()
        scroll.setWidget(cb_container)
        root.addWidget(scroll)

        # ---- 메인 차트 ----
        date_axis = pg.DateAxisItem(orientation="bottom")
        self.plot_widget = pg.PlotWidget(
            axisItems={"bottom": date_axis},
            title=f"<b>{self.title}</b>",
        )
        self.plot_widget.setLabel("left", "비율 (%)")
        self.plot_widget.setLabel("bottom", "날짜")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.plotItem.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        # X축만 줌/드래그
        self.plot_widget.plotItem.vb.setMouseEnabled(x=True, y=False)

        # ---- 우측 Y축 ViewBox ----
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

        # ---- 가로 스크롤바 ----
        self._scrollbar = QScrollBar(Qt.Horizontal)
        self._scrollbar.setMinimum(0)
        self._scrollbar.setMaximum(SCROLL_RESOLUTION)
        self._scrollbar.setValue(SCROLL_RESOLUTION)
        self._scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        root.addWidget(self._scrollbar)

    def _update_price_vb_geometry(self):
        if self._price_vb is not None:
            self._price_vb.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())

    def _timestamps(self) -> np.ndarray:
        return self.df.index.astype("int64") // 10 ** 9

    def _plot_all(self):
        self.plot_widget.clear()
        self._curves.clear()
        self._price_curve = None

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

        for col, cb in self._checkboxes.items():
            if col not in self.df.columns:
                continue
            color = COL_COLORS.get(col, "#808080")
            width = COL_WIDTHS.get(col, 1.5)
            pen = pg.mkPen(color=color, width=width)
            vals = pd.to_numeric(self.df[col], errors="coerce").fillna(0).to_numpy()
            curve = self.plot_widget.plot(x=ts, y=vals, pen=pen)
            curve.setVisible(cb.isChecked())
            self._curves[col] = curve

        if self.price_df is not None and self._price_vb is not None:
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

        self._update_price_vb_geometry()
        self._update_scrollbar_range()
        # 데이터 범위로 X축 초기화 (빈 여백 제거)
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

    def update_df(self, df: pd.DataFrame, price_df: Optional[pd.DataFrame] = None):
        self.df = df
        if price_df is not None:
            self.price_df = price_df
        self._plot_all()


# =============================================================================
# 메인 창 (탭 3개)
# =============================================================================
class RatioChartWindow(QMainWindow):

    def __init__(
        self,
        driving_df: pd.DataFrame,
        holding_df: pd.DataFrame,
        spread_df: pd.DataFrame,
        title: str = "수급 비율",
        price_df: Optional[pd.DataFrame] = None,
        default_visible: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1600, 650)
        self._price_df = price_df
        self._default_visible = default_visible if default_visible is not None else DEFAULT_VISIBLE
        self._build_ui(driving_df, holding_df, spread_df)

    def _build_ui(self, driving_df, holding_df, spread_df):
        tab_widget = QTabWidget()
        for label, df in zip(TAB_LABELS, [driving_df, holding_df, spread_df]):
            tab_widget.addTab(
                _RatioTab(
                    df=df,
                    title=label,
                    price_df=self._price_df,
                    default_visible=self._default_visible,
                ),
                label,
            )
        self.setCentralWidget(tab_widget)

    def update_data(
        self,
        driving_df: pd.DataFrame,
        holding_df: pd.DataFrame,
        spread_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
    ):
        tab_widget: QTabWidget = self.centralWidget()
        for i, df in enumerate([driving_df, holding_df, spread_df]):
            tab: _RatioTab = tab_widget.widget(i)
            tab.update_df(df, price_df=price_df)


# =============================================================================
# 진입 함수
# =============================================================================
def show_ratio_chart(
    driving_df: pd.DataFrame,
    holding_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    *,
    title: str = "수급 비율",
    price_df: Optional[pd.DataFrame] = None,
    default_visible: Optional[List[str]] = None,
    parent: Optional[QWidget] = None,
) -> RatioChartWindow:
    return RatioChartWindow(
        driving_df, holding_df, spread_df,
        title=title,
        price_df=price_df,
        default_visible=default_visible,
        parent=parent,
    )