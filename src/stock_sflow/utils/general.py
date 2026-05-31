# =============================================================================
# 모듈 헤더
# -----------------------------------------------------------------------------
# 목적      : 데이터 입출력 및 DataFrame 검증/정렬 유틸리티
# 주요 기능 : 엑셀 로드/저장, 필수 컬럼 검증, 날짜 인덱스 정리
# 요구 사항 : Python 3.9+, pandas, openpyxl(xlsx 사용 시)
# 작성자    : (작성자명)
# 최종수정  : 2025-09-28
# 버전      : 1.2.0
# =============================================================================

# 표준 라이브러리
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

# 서드파티
import pandas as pd

# 프로젝트 내부 모듈
# (없음)


# =============================================================================
# 상수 / 로깅 설정
# =============================================================================
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    # 라이브러리/모듈로도 쓸 수 있게 root 설정에 의존하되, 미설정 시 기본 포맷 제공
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# 내부 유틸
# =============================================================================
def _ensure_parent_dir(path: Path) -> None:
    """파일 경로의 부모 디렉터리가 없으면 생성."""
    path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 공개 API
# =============================================================================
def load_excel(
    file_path: str | Path,
    *,
    sheet_name: str | int | list[int | str] | None = 0,
    dtype: Optional[dict] = None,
    engine: Optional[str] = None,
    header: int | Sequence[int] | None = 0,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    엑셀 파일을 로드하여 DataFrame(또는 시트별 dict)으로 반환합니다.

    Parameters
    ----------
    file_path : str | Path
        엑셀 파일 경로(.xlsx 권장).
    sheet_name : str | int | list | None, default 0
        읽을 시트(여러 시트면 dict 반환). None이면 모든 시트.
    dtype : dict, optional
        컬럼별 dtype 강제 변환 사전.
    engine : str, optional
        판다스 엔진(openpyxl 등) 명시. None이면 pandas 기본 결정 사용.
    header : int | Sequence[int] | None, default 0
        헤더 행(멀티헤더 지원).

    Returns
    -------
    pd.DataFrame | dict[str, pd.DataFrame]
        시트가 하나면 DataFrame, 여러 개면 dict(시트명→DataFrame).

    Raises
    ------
    FileNotFoundError
        경로가 없을 때.
    ValueError
        파일 내용이 비정상일 때.
    Exception
        판다스 내부 예외 등.
    """
    path = Path(file_path)
    if not path.exists():
        LOGGER.error(f"파일을 찾을 수 없습니다: {path}")
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    try:
        df_or_dict = pd.read_excel(
            path,
            sheet_name=sheet_name,
            dtype=dtype,
            engine=engine,
            header=header,
        )
        LOGGER.info(f"엑셀 로드 성공: {path}")
        return df_or_dict
    except Exception as e:
        LOGGER.error(f"엑셀 로드 오류: {e}")
        raise


def save_to_excel(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    filename: str | Path,
    *,
    index: bool = True,
    engine: Optional[str] = None,
    if_sheet_exists: Optional[str] = None,
) -> None:
    """
    DataFrame(또는 시트명→DataFrame dict)을 엑셀 파일로 저장합니다.

    Parameters
    ----------
    df : DataFrame | dict[str, DataFrame]
        저장할 데이터. dict면 각 key가 시트명, value가 DataFrame입니다.
    filename : str | Path
        저장 파일 경로.
    index : bool, default True
        인덱스 포함 여부.
    engine : str, optional
        엑셀 엔진(openpyxl 등). None이면 pandas 기본.
    if_sheet_exists : {"error","new","replace","overlay"}, optional
        기존 파일/시트에 저장할 때 모드 (pandas>=1.5의 ExcelWriter 옵션).

    Raises
    ------
    Exception
        저장 과정에서 발생한 예외.
    """
    path = Path(filename)
    _ensure_parent_dir(path)

    try:
        # dict를 전달받으면 멀티시트로 저장
        if isinstance(df, dict):
            with pd.ExcelWriter(path, engine=engine, if_sheet_exists=if_sheet_exists) as writer:
                for sheet, subdf in df.items():
                    if not isinstance(subdf, pd.DataFrame):
                        raise TypeError(f"시트 '{sheet}' 값이 DataFrame이 아닙니다.")
                    subdf.to_excel(writer, sheet_name=str(sheet), index=index)
        else:
            # 단일 시트 저장
            df.to_excel(path, index=index, engine=engine)
        LOGGER.info(f"엑셀 저장 완료: {path}")
    except Exception as e:
        LOGGER.error(f"엑셀 저장 오류: {e}")
        raise


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    strict: bool = False,
) -> list[str]:
    """
    DataFrame에 required_columns가 모두 존재하는지 확인.

    Parameters
    ----------
    df : DataFrame
        검사 대상.
    required_columns : Iterable[str]
        필요한 컬럼 목록.
    strict : bool, default False
        True면 누락 컬럼 존재 시 예외(ValueError)를 발생.

    Returns
    -------
    list[str]
        누락된 컬럼 목록(없으면 빈 리스트).

    Raises
    ------
    ValueError
        df가 비어 있거나(strict=True인 경우) 필수 컬럼 누락 시.
    """
    if df is None or df.empty:
        LOGGER.error("입력 DataFrame이 비어 있습니다.")
        raise ValueError("입력 DataFrame이 비어 있습니다.")

    req = list(required_columns)
    missing = [col for col in req if col not in df.columns]

    if missing:
        msg = f"누락된 컬럼: {missing}"
        if strict:
            LOGGER.error(msg)
            raise ValueError(msg)
        else:
            LOGGER.warning(msg)
    else:
        LOGGER.info("필수 컬럼 검증 통과.")

    return missing


def ensure_datetime_index(
    df: pd.DataFrame,
    *,
    date_column: str = "일자",
    ascending: bool = False,
    inplace: bool = False,
    coerce_errors: bool = True,
) -> pd.DataFrame:
    """
    날짜 컬럼을 datetime으로 변환한 뒤 정렬하고 인덱스로 설정합니다.

    Parameters
    ----------
    df : DataFrame
        대상 DataFrame.
    date_column : str, default "일자"
        날짜 컬럼명.
    ascending : bool, default False
        True면 오래된→최신, False면 최신→오래된 순으로 정렬.
    inplace : bool, default False
        True면 원본 수정, False면 사본을 반환.
    coerce_errors : bool, default True
        True면 변환 실패 값을 NaT로 강제(errors='coerce').

    Returns
    -------
    DataFrame
        정리된 DataFrame(인플레이스면 동일 객체).

    Raises
    ------
    KeyError
        date_column이 존재하지 않을 때.
    ValueError
        모든 날짜가 변환 실패(NaT)일 때.
    """
    if date_column not in df.columns:
        raise KeyError(f"'{date_column}' 컬럼을 찾을 수 없습니다.")

    target = df if inplace else df.copy()

    try:
        errors = "coerce" if coerce_errors else "raise"
        target[date_column] = pd.to_datetime(target[date_column], errors=errors)

        if target[date_column].isna().all():
            raise ValueError(f"모든 '{date_column}' 값이 datetime으로 변환되지 않았습니다.")

        target.sort_values(by=date_column, ascending=ascending, inplace=True)
        target.set_index(date_column, inplace=True)
        LOGGER.info(
            f"'{date_column}' → datetime 변환 및 정렬({ '오름차순' if ascending else '내림차순' }) 후 인덱스 설정 완료."
        )
        return target
    except Exception as e:
        LOGGER.error(f"'{date_column}' 처리 중 오류: {e}")
        raise
