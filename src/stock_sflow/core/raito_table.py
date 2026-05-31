# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 퍼센트 값이 들어있는 DataFrame을 QTableWidget으로 시각화(셀 내 바 차트)
# 주요 기능 : 퍼센트 파싱(문자/NaN/경계값 안전), 구간별 색상, 선택 하이라이트, 툴팁
# 요구 사항 : Python 3.9+, PyQt5, pandas, numpy(선택)
# 작성자    : (작성자명)
# 최종수정  : 2025-10-02
# 버전      : 1.2.0
# =============================================================================

# 표준 라이브러리
from __future__ import annotations
import sys
from typing import Optional, Sequence

# 서드파티
import pandas as pd

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QStyledItemDelegate, QStyle
)
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QRect


# =============================================================================
# 상수 / 설정
# =============================================================================
# 퍼센트 구간 경계(이 값 이하일 때 적용)
P_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)  # 0~100 스케일 가정
# 구간 색상(경계에 대응): 파랑, 연녹, 노랑, 주황, 빨강
P_COLORS_HEX = ("#0000FF", "#90EE90", "#FFFF00", "#FFA500", "#FF0000")

# 테이블 기본 셀 폭
DEFAULT_COL_WIDTH = 100
# 퍼센트 표시 포맷
P_FORMAT = "{:.1f}%"


# =============================================================================
# 내부 유틸
# =============================================================================
def _parse_percent(value) -> float:
    """
    퍼센트 형태(예: '45%', ' 12.3 ', 55 등)를 float[0,100]로 변환.
    - 변환 실패 시 0.0 반환
    - 범위를 벗어나면 0~100으로 클램프
    """
    try:
        if isinstance(value, str):
            v = value.strip()
            if v.endswith("%"):
                v = v[:-1]
            pct = float(v)
        else:
            pct = float(value)
    except Exception:
        return 0.0

    if pct != pct:  # NaN 체크
        return 0.0
    if pct < 0.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def _pick_color(pct: float) -> QColor:
    """
    퍼센트 구간에 따라 색상을 선택.
    """
    t1, t2, t3, t4 = P_THRESHOLDS
    c1, c2, c3, c4, c5 = P_COLORS_HEX
    if pct <= t1:
        return QColor(c1)
    elif pct <= t2:
        return QColor(c2)
    elif pct <= t3:
        return QColor(c3)
    elif pct <= t4:
        return QColor(c4)
    return QColor(c5)


# =============================================================================
# 델리게이트
# =============================================================================
class BarDelegate(QStyledItemDelegate):
    """
    테이블 셀 내부에 퍼센트 바를 그리는 델리게이트.
    - 선택 상태에서는 시스템 하이라이트 배경을 우선 적용
    - 값은 중앙 정렬 텍스트로 함께 표시
    """

    def paint(self, painter: QPainter, option, index):
        painter.save()

        # 표시 텍스트 & 퍼센트값 파싱
        text = index.data(Qt.DisplayRole)
        pct = _parse_percent(text)

        rect: QRect = option.rect
        # 선택 상태 배경 그리기
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        else:
            painter.fillRect(rect, option.palette.base())

        # 퍼센트 바 너비 계산
        fill_w = int(rect.width() * (pct / 100.0))
        fill_rect = QRect(rect.x(), rect.y(), fill_w, rect.height())

        # 퍼센트 바 그리기
        fill_color = _pick_color(pct)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(fill_rect, QBrush(fill_color))

        # 외곽선
        painter.setPen(Qt.black)
        painter.drawRect(rect)

        # 텍스트 표시(일관된 포맷)
        display = text if isinstance(text, str) else P_FORMAT.format(pct)
        painter.setPen(option.palette.highlightedText().color()
                       if option.state & QStyle.State_Selected else Qt.black)
        painter.drawText(rect, Qt.AlignCenter, display)

        painter.restore()

    # 선택적으로 크기 힌트 조정이 필요할 때 오버라이드 가능
    # def sizeHint(self, option, index):
    #     return super().sizeHint(option, index)


# =============================================================================
# 메인 뷰 생성 함수
# =============================================================================
def show_percentage_table(df: pd.DataFrame, *, title: str = "퍼센트 테이블",
                          parent: Optional[QWidget] = None) -> QMainWindow:
    """
    퍼센트 값(또는 퍼센트 문자열)을 포함한 DataFrame을 테이블로 표시하는 창(QMainWindow)을 생성.
    - 각 셀에 퍼센트 바를 렌더링
    - 헤더 볼드
    - 기본 열 너비 고정
    - 각 셀 툴팁: 원본 값과 파싱된 퍼센트 표시

    Returns
    -------
    QMainWindow : 호출자가 .show() 호출하여 띄우는 것을 권장
    """
    # QApplication 보장
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = QMainWindow(parent)
    win.setWindowTitle(title)

    table = QTableWidget()
    n_rows, n_cols = df.shape
    table.setRowCount(n_rows)
    table.setColumnCount(n_cols)

    # 헤더 레이블
    table.setHorizontalHeaderLabels([str(c) for c in df.columns.tolist()])
    table.setVerticalHeaderLabels([str(i) for i in df.index.tolist()])

    # 헤더 볼드
    bold = QFont()
    bold.setBold(True)
    table.horizontalHeader().setFont(bold)
    table.verticalHeader().setFont(bold)

    # 델리게이트 설정
    delegate = BarDelegate()
    table.setItemDelegate(delegate)

    # 데이터 채우기
    for r in range(n_rows):
        for c in range(n_cols):
            raw = df.iat[r, c]
            pct = _parse_percent(raw)
            display = P_FORMAT.format(pct)
            item = QTableWidgetItem(display)
            item.setTextAlignment(Qt.AlignCenter)
            # 툴팁: 원본 값과 파싱 결과를 함께
            item.setToolTip(f"원본: {raw}\n해석된 퍼센트: {display}")
            # 정렬/소팅을 고려한다면 사용자 역할 데이터로 순수 수치 저장
            item.setData(Qt.UserRole, pct)
            table.setItem(r, c, item)

    # 컬럼 너비/행 높이
    for col in range(n_cols):
        table.setColumnWidth(col, DEFAULT_COL_WIDTH)
    table.resizeRowsToContents()

    # 창 크기 계산(여유 여백 포함)
    total_w = table.verticalHeader().width() + sum(table.columnWidth(i) for i in range(n_cols)) + 24
    total_h = table.horizontalHeader().height() + sum(table.rowHeight(i) for i in range(n_rows)) + 24

    win.setCentralWidget(table)
    win.resize(total_w, total_h)

    return win
