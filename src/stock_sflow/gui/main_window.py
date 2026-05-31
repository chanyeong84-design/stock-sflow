# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 메인 GUI (입력/버튼/로그/창 관리)
# 주요 기능 : 데이터 조회 -> 분석 실행 -> 테이블/그래프 창 표시
# 요구 사항 : QtPy, pandas, pykiwoom
# 작성자    : (작성자명 기입)
# 최종수정  : 2025-12-30
# 버전      : 1.1.0
# =============================================================================

# =============================== 표준 라이브러리 ==============================
from __future__ import annotations
import shutil
from typing import Optional, Tuple

# =============================== 서드파티 라이브러리 ==========================
import pandas as pd
from qtpy import QtWidgets, QtCore, QtGui

# =============================== 프로젝트 내부 모듈 ===========================
from stock_sflow.app.config_store import ConfigStore
from stock_sflow.app.data_service import DataService
from stock_sflow.app.analysis_service import AnalysisService
from stock_sflow.utils.qt_helpers import qdate_to_yyyymmdd, qdate_from_yyyy_mm_dd

from stock_sflow.core import sd_table
from stock_sflow.core import raito_table as ratio_table  # NOTE: 기존 오타 통일
from stock_sflow.core import ratio_chart
from stock_sflow.core import acc_chart
from stock_sflow.core import avg_price_chart

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Optional[ConfigStore] = None):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("통합 수급 및 분석 프로그램")

        # 의존성(설정/서비스) - ✅ 외부 주입(DI) 지원
        self.cfg = cfg or ConfigStore()

        # ✅ config 기본 구조 보장 (첫 실행 KeyError 방지)
        self.cfg.data.setdefault("window_positions", {})
        self.cfg.data.setdefault("plot_settings", {})
      
        self.data_service = DataService(cache_dir=self.cfg.cache_dir)
        self.analysis_service = AnalysisService()

        # 상태
        self.result_df: Optional[pd.DataFrame] = None
        self.last_fetch_key: Tuple[Optional[str], Optional[str]] = (None, None)

        # 자식 창
        self.sd_win = None
        self.ratio_win = None
        self.ratio_chart_win = None
        self.acc_chart_win = None
        self.avg_price_win = None
        self.stock_name: str = ""
        
        self.init_ui()
        self.restore_last_state()
        self.load_history()
        self.restore_window_geometry()
        
        # ✅ 앱 시작 시 키움 연결 (비동기로 띄우면 UI가 먼저 뜸)
        QtCore.QTimer.singleShot(500, self._connect_on_startup)
        
    
    # ------------------------------- Kiwoom 증권 API 연결 -------------------------------
    def _connect_on_startup(self):
        self.log("키움 API 연결 시도 중...")
        ok = self.data_service.connect(log=self.log)
        if ok:
            self.log("✅ 키움 연결 완료. 종목 조회 준비됨.")
        else:
            self.log("❌ 키움 연결 실패. 수동으로 재시도하세요.")

    # ------------------------------- UI 구성 -------------------------------
    def init_ui(self):
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        form = QtWidgets.QFormLayout()

        self.code_input = QtWidgets.QComboBox()
        self.code_input.setEditable(True)
        form.addRow("종목 코드:", self.code_input)

        self.api_date_edit = QtWidgets.QDateEdit(calendarPopup=True)
        self.api_date_edit.setDate(QtCore.QDate.currentDate())
        self.api_date_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow("API 조회일:", self.api_date_edit)

        self.slice_start_edit = QtWidgets.QDateEdit(calendarPopup=True)
        self.slice_start_edit.setDate(QtCore.QDate.currentDate().addYears(-3))
        self.slice_start_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow("슬라이스 시작일:", self.slice_start_edit)

        self.slice_end_edit = QtWidgets.QDateEdit(calendarPopup=True)
        self.slice_end_edit.setDate(QtCore.QDate.currentDate())
        self.slice_end_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow("슬라이스 종료일:", self.slice_end_edit)

        self.fetch_btn = QtWidgets.QPushButton("데이터 조회")
        self.analyze_btn = QtWidgets.QPushButton("분석 실행")
        self.clear_cache_btn = QtWidgets.QPushButton("캐시 비우기")

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.fetch_btn)
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.clear_cache_btn)

        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)

        layout = QtWidgets.QVBoxLayout(container)
        layout.addLayout(form)
        layout.addLayout(btn_layout)
        layout.addWidget(self.log_output)

        self.fetch_btn.clicked.connect(self.on_fetch)
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.clear_cache_btn.clicked.connect(self.on_clear_cache)

    # ------------------------------- 로깅 -------------------------------
    def log(self, message: str):
        ts = QtCore.QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_output.append(f"[{ts}] {message}")

    def update_progress(self, iteration, date_str=None):
        text = f"조회 {iteration}회차"
        if date_str:
            text += f" (기준일: {date_str})"
        self.statusBar().showMessage(text)

    # ----------------------------- 설정/상태 ------------------------------
    def _current_last_state(self) -> dict:
        return {
            "code": self.code_input.currentText().strip(),
            "api_date": self.api_date_edit.date().toString("yyyy-MM-dd"),
            "slice_start": self.slice_start_edit.date().toString("yyyy-MM-dd"),
            "slice_end": self.slice_end_edit.date().toString("yyyy-MM-dd"),
        }

    def restore_last_state(self):
        ls = self.cfg.data.get("last_state", {})
        code = ls.get("code", "")
        if code:
            if self.code_input.findText(code) < 0:
                self.code_input.addItem(code)
            self.code_input.setCurrentText(code)

        # self.api_date_edit.setDate(qdate_from_yyyy_mm_dd(ls.get("api_date", "")))
        self.slice_start_edit.setDate(qdate_from_yyyy_mm_dd(ls.get("slice_start", "")))
        self.slice_end_edit.setDate(qdate_from_yyyy_mm_dd(ls.get("slice_end", "")))

    def load_history(self):
        for code in self.cfg.data.get("history", []):
            self.code_input.addItem(code)

    def save_history(self, code: str):
        hist = self.cfg.data.setdefault("history", [])
        if code and code not in hist:
            hist.append(code)
            self.code_input.addItem(code)

    def restore_window_geometry(self):
        geom = self.cfg.data.get("window_positions", {}).get("main")
        if geom:
            self.setGeometry(*geom)

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[name-defined]
        # ✅ 기본 구조 보장
        self.cfg.data.setdefault("window_positions", {})
        self.cfg.data.setdefault("plot_settings", {})

        g = self.geometry()
        self.cfg.data["window_positions"]["main"] = [g.x(), g.y(), g.width(), g.height()]

        for key, win in [
            ("sd", self.sd_win),
            ("ratio", self.ratio_win),
            ("ratio_chart", self.ratio_chart_win),
            ("acc_chart", self.acc_chart_win),
            ("avg_price", self.avg_price_win),
        ]:
            if win is not None and hasattr(win, "geometry"):
                gg = win.geometry()
                self.cfg.data["window_positions"][key] = [gg.x(), gg.y(), gg.width(), gg.height()]
        self.cfg.data["last_state"] = self._current_last_state()
        
        if self.data_service.kiwoom is not None:
            try:
                self.data_service.kiwoom.CommTerminate()  # 연결 종료
            except Exception:
                pass
        self.cfg.save()
        super().closeEvent(event)

    # ------------------------------- 액션 -------------------------------
    def on_clear_cache(self):
        try:
            if self.cfg.cache_dir.is_dir():
                shutil.rmtree(self.cfg.cache_dir)
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            self.result_df = None
            self.last_fetch_key = (None, None)
            self.log("캐시를 모두 삭제했습니다.")
        except Exception as e:
            self.log(f"캐시 삭제 실패: {e}")

    def on_fetch(self):
        self.cfg.data["last_state"] = self._current_last_state()
        self.cfg.save()

        code = self.code_input.currentText().strip()
        if not code:
            self.log("종목 코드를 입력하세요.")
            return

        api_date = qdate_to_yyyymmdd(self.api_date_edit.date())
        df, _ = self.data_service.fetch(
            code, api_date, log=self.log, progress_callback=self.update_progress
        )
        if df is None:
            return

        self.result_df = df
        self.last_fetch_key = (code, api_date)
        self.stock_name = self.data_service.stock_name
        self.log(f"데이터 로드 완료: {len(df)}건")
        self.save_history(code)
        self.cfg.save()

    def on_analyze(self):
        self.cfg.data["last_state"] = self._current_last_state()
        self.cfg.save()

        code = self.code_input.currentText().strip()
        api_date = qdate_to_yyyymmdd(self.api_date_edit.date())

        if self.result_df is None or (code, api_date) != self.last_fetch_key:
            self.log("데이터 없거나 변경됨 → 자동 조회 실행")
            self.on_fetch()
            if self.result_df is None:
                return

        raw_start = qdate_to_yyyymmdd(self.slice_start_edit.date())
        raw_end = qdate_to_yyyymmdd(self.slice_end_edit.date())
        if raw_start > raw_end:
            self.log("슬라이스 기간이 올바르지 않습니다. (시작일 ≤ 종료일)")
            return

        try:
            self.log("전처리/분석 시작...")
            res = self.analysis_service.run(self.result_df, raw_start, raw_end)

            # -------------------- 추정 평균가 조회 (차트 생성 전) --------------------
            self.log("추정 평균가 조회 중... (API 2회 추가 호출)")
            avg_df = self.data_service.fetch_avg_price(
                code, api_date,
                log=self.log,
                progress_callback=self.update_progress,
            )
            # 슬라이스 기간에 맞춰 필터링
            avg_df_sliced = None
            if avg_df is not None and not avg_df.empty:
                sliced_start = res.sliced.index.min()
                sliced_end   = res.sliced.index.max()
                avg_df_sliced = avg_df.loc[
                    (avg_df.index >= sliced_start) & (avg_df.index <= sliced_end)
                ]

            # -------------------- 수급분석표 --------------------
            self.log("수급분석표 생성")
            if self.sd_win is not None and hasattr(self.sd_win, "close"):
                self.sd_win.close()
            self.sd_win = sd_table.show_supply_analysis_table(
                res.sliced, res.acc, res.high,
                title=f"수급 분석표({self.stock_name})",
            )
            sd_pos = self.cfg.data.get("window_positions", {}).get("sd")
            if sd_pos:
                self.sd_win.setGeometry(*sd_pos)
            self.sd_win.show()

            # -------------------- 비율표 --------------------
            self.log("비율표 생성")
            if self.ratio_win is not None and hasattr(self.ratio_win, "close"):
                self.ratio_win.close()
            self.ratio_win = ratio_table.show_percentage_table(
                res.rdf,
                title=f"수급 테이블({self.stock_name})",
            )
            ratio_pos = self.cfg.data.get("window_positions", {}).get("ratio")
            if ratio_pos:
                self.ratio_win.setGeometry(*ratio_pos)
            self.ratio_win.show()

            # -------------------- 비율차트 --------------------
            self.log("비율차트 생성")
            if self.ratio_chart_win is not None and hasattr(self.ratio_chart_win, "close"):
                self.ratio_chart_win.close()
            self.ratio_chart_win = ratio_chart.show_ratio_chart(
                res.drv,
                res.hold,
                res.sprd,
                title=f"수급 비율({self.stock_name})",
                price_df=res.price,
                default_visible=self.cfg.data["plot_settings"].get(
                    "ratio_chart_visible", ["개인", "외국인", "기관"]
                ),
            )
            ratio_chart_pos = self.cfg.data.get("window_positions", {}).get("ratio_chart")
            if ratio_chart_pos:
                self.ratio_chart_win.setGeometry(*ratio_chart_pos)
            self.ratio_chart_win.show()

            # -------------------- 매집수량 차트 (추정 평균가 MA 포함) --------------------
            self.log("매집수량 차트 생성")
            if self.acc_chart_win is not None and hasattr(self.acc_chart_win, "close"):
                self.acc_chart_win.close()
            self.acc_chart_win = acc_chart.show_acc_chart(
                res.acc,
                price_df=res.price,
                avg_df=avg_df_sliced,
                title=f"매집수량({self.stock_name})",
                default_visible=self.cfg.data["plot_settings"].get(
                    "ratio_chart_visible", ["개인", "외국인", "기관"]
                ),
            )
            acc_chart_pos = self.cfg.data.get("window_positions", {}).get("acc_chart")
            if acc_chart_pos:
                self.acc_chart_win.setGeometry(*acc_chart_pos)
            self.acc_chart_win.show()

            # -------------------- 추정 평균가 차트 --------------------
            if avg_df is not None and not avg_df.empty:
                if self.avg_price_win is not None and hasattr(self.avg_price_win, "close"):
                    self.avg_price_win.close()
                self.avg_price_win = avg_price_chart.show_avg_price_chart(
                    avg_df,
                    price_df=res.price,
                    title=f"추정 평균가({self.stock_name})",
                    default_visible=self.cfg.data["plot_settings"].get(
                        "ratio_chart_visible", ["개인", "외국인", "기관"]
                    ),
                )
                avg_price_pos = self.cfg.data.get("window_positions", {}).get("avg_price")
                if avg_price_pos:
                    self.avg_price_win.setGeometry(*avg_price_pos)
                self.avg_price_win.show()
                self.log("추정 평균가 차트 완료.")
            else:
                self.log("추정 평균가 데이터 없음.")

        except Exception as e:
            self.log(f"분석 중 오류: {e}")
