# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : GUI 실행 엔트리 제공 (run())
# 주요 기능 : QApplication 생성 후 MainWindow 실행
# 요구 사항 : Python 3.9+, QtPy
# 작성자    : (작성자명 기입)
# 최종수정  : 2025-12-30
# 버전      : 1.1.0
# =============================================================================

# =============================== 표준 라이브러리 ==============================
import sys
from typing import Optional

# =============================== 서드파티 라이브러리 ==========================
from qtpy.QtWidgets import QApplication

# =============================== 프로젝트 내부 모듈 ===========================
from stock_sflow.app.config_store import ConfigStore
from stock_sflow.gui.main_window import MainWindow

# =============================================================================
# 엔트리 포인트 함수
# =============================================================================
def run(config_store: Optional[ConfigStore] = None):
    """
    GUI를 실행한다.

    Args:
        config_store: 외부에서 주입받는 ConfigStore.
            None이면 기본 ConfigStore()를 내부에서 생성한다.
    """
    app = QApplication(sys.argv)
    cfg = config_store or ConfigStore()
    win = MainWindow(cfg)   # ✅ MainWindow가 cfg를 받도록 변경 필요
    win.resize(980, 680)
    win.show()

    # 이벤트 루프 종료 코드를 OS로 전달
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
