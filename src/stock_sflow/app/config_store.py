# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : config.json 저장/복원 책임 분리 + 경로 정책(프로젝트 루트) 표준화 + 마이그레이션
# 주요 기능 : 기본 설정 제공, 누락 키 보정, 버전 마이그레이션, 안전 저장(atomic)
# 요구 사항 : Python 3.9+
# 작성자    : (작성자명 기입)
# 최종수정  : 2025-12-29
# 버전      : 1.3.1
# =============================================================================

# =============================== 표준 라이브러리 ==============================
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# 상수 / 기본 설정
# =============================================================================
SCHEMA_VERSION: int = 1
HISTORY_MAX: int = 50

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "history": [],
    "window_positions": {"main": None, "sd": None, "ratio": None, "combined": None},
    "plot_settings": {"combined": {"colors": {}, "visible": []},
                        "ratio_chart_visible": ["개인", "외국인", "기관"],
    },
    "app_version": "1.3.1",
    "last_state": {"code": "", "api_date": "", "slice_start": "", "slice_end": ""},
}


# =============================================================================
# 경로 유틸
# =============================================================================
def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 반환.

    목표(선택지 A):
    - 프로젝트 루트(= run.py가 있는 폴더)에 config.json / cache / logs를 둔다.
    - 현재 프로젝트는 src-layout이므로 config_store.py는 보통:
        <PROJECT_ROOT>/src/stock_sflow/app/config_store.py
      형태이다.

    구현:
    - 현재 파일 위치에서 위로 올라가며 run.py를 찾는다.
    - 찾으면 해당 폴더를 프로젝트 루트로 간주한다.
    - 못 찾으면 안전장치로 현재 작업 폴더(Path.cwd())를 반환한다.
      (단, 실제 운영에선 run.py에서 base_dir를 주입하는 것을 권장)
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "run.py").is_file():
            return p

    # fallback: 개발 환경/실행 방식에 따라 run.py 탐색이 실패할 수 있으므로 보수적으로 처리
    return Path.cwd().resolve()


class ConfigStore:
    """
    - config.json 로드/저장(atomic)
    - 누락 키 보정(스키마 정규화)
    - schema_version/app_version 기반 마이그레이션 훅 제공
    - cache/logs 디렉토리 생성 및 경로 제공

    저장 위치 정책(선택지 A):
    - base_dir(프로젝트 루트)/
        - config.json
        - cache/
        - logs/
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        # 선택지 A: 프로젝트 루트(run.py가 있는 폴더)를 기본 기준으로 삼는다.
        self.base_dir: Path = (base_dir or get_project_root()).resolve()

        # 파일/폴더 경로 표준화 (프로젝트 루트에 생성)
        self.config_path: Path = self.base_dir / "config.json"
        self.cache_dir: Path = self.base_dir / "cache"
        self.logs_dir: Path = self.base_dir / "logs"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.data: Dict[str, Any] = self.load()

    # -------------------------------------------------------------------------
    # Public helpers (경로)
    # -------------------------------------------------------------------------
    def get_cache_path(self, filename: str) -> Path:
        """cache 폴더 내부 파일 경로."""
        return self.cache_dir / filename

    def get_log_path(self, filename: str) -> Path:
        """logs 폴더 내부 파일 경로."""
        return self.logs_dir / filename

    # -------------------------------------------------------------------------
    # Load / Save
    # -------------------------------------------------------------------------
    def deep_default(self) -> Dict[str, Any]:
        return copy.deepcopy(DEFAULT_CONFIG)

    def load(self) -> Dict[str, Any]:
        if not self.config_path.is_file():
            cfg = self.deep_default()
            # 최초 실행 시에도 파일을 만들어두고 싶으면 여기서 self.data=cfg; self.save() 호출 가능
            return cfg

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            # 깨진 config면 기본값으로 복구
            return self.deep_default()

        cfg = self._normalize(cfg)
        cfg = self._migrate_if_needed(cfg)
        cfg = self._normalize(cfg)

        # history 과다 방지
        cfg["history"] = list(cfg.get("history", []))[-HISTORY_MAX:]
        return cfg

    def save(self) -> None:
        """
        안전 저장:
        - config.json.tmp에 먼저 쓰고
        - 성공하면 config.json으로 교체
        """
        try:
            self.data = self._normalize(self.data)
            tmp_path = self.config_path.with_suffix(".json.tmp")

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)

            tmp_path.replace(self.config_path)
        except Exception:
            # GUI에서 로그로 처리할 수도 있으니 여기서는 조용히
            pass

    # -------------------------------------------------------------------------
    # Normalize / Migration
    # -------------------------------------------------------------------------
    def _normalize(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        누락 키 보정 + 타입이 이상한 경우 기본값으로 복구.
        """
        if not isinstance(cfg, dict):
            return self.deep_default()

        cfg.setdefault("schema_version", SCHEMA_VERSION)
        cfg.setdefault("app_version", DEFAULT_CONFIG["app_version"])

        cfg.setdefault("history", [])
        if not isinstance(cfg["history"], list):
            cfg["history"] = []

        cfg.setdefault("window_positions", {})
        if not isinstance(cfg["window_positions"], dict):
            cfg["window_positions"] = {}
        for k in ["main", "sd", "ratio", "combined", "ratio_chart", "acc_chart"]:
            cfg["window_positions"].setdefault(k, None)

        cfg.setdefault("plot_settings", {})
        if not isinstance(cfg["plot_settings"], dict):
            cfg["plot_settings"] = {}
        cfg["plot_settings"].setdefault("combined", {"colors": {}, "visible": []})
        cfg["plot_settings"].setdefault(  # ← 추가
            "ratio_chart_visible",
            ["개인", "외국인", "기관"]
        )
        cfg.setdefault("last_state", dict(DEFAULT_CONFIG["last_state"]))
        if not isinstance(cfg["last_state"], dict):
            cfg["last_state"] = dict(DEFAULT_CONFIG["last_state"])

        return cfg

    def _migrate_if_needed(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        schema_version 기반 마이그레이션.
        - 스키마가 늘어나거나 키 이름이 바뀌는 경우 여기서 흡수.
        """
        current = int(cfg.get("schema_version", 0))

        if current < 1:
            # 예시: 과거 버전에 schema_version이 없었다면 1로 올림
            cfg["schema_version"] = 1

        # 앞으로 스키마 버전이 늘면:
        # if current < 2: ...
        # if current < 3: ...

        return cfg

    # -------------------------------------------------------------------------
    # High-level setters/getters (GUI에서 자주 쓰는 것만)
    # -------------------------------------------------------------------------
    def push_history(self, item: Any) -> None:
        hist = list(self.data.get("history", []))
        hist.append(item)
        self.data["history"] = hist[-HISTORY_MAX:]

    def set_last_state(self, **kwargs: str) -> None:
        st = dict(self.data.get("last_state", {}))
        st.update(kwargs)
        self.data["last_state"] = st

    def set_window_position(self, key: str, pos: Optional[Dict[str, int]]) -> None:
        """
        pos 예시: {"x": 10, "y": 20, "w": 1200, "h": 800}
        """
        if key not in self.data["window_positions"]:
            return
        self.data["window_positions"][key] = pos

    def get_window_position(self, key: str) -> Optional[Dict[str, int]]:
        return self.data.get("window_positions", {}).get(key)
