from __future__ import annotations

__all__ = ["inf", "nan"]


class md_nan:
    def __init__(self) -> None:
        self.value = "not a number!"

    def __float__(self) -> float:
        return float("nan")

    def __eq__(self, other: object) -> bool:
        return bool(isinstance(other, md_nan))

    def __repr__(self) -> str:
        return "nan"


class md_infp:
    def __init__(self) -> None:
        self.value = "infinity!"

    def __eq__(self, other: object) -> bool:
        return self  # type: ignore[return-value]

    def __gt__(self, other: object) -> bool:
        return True

    def __lt__(self, other: object) -> bool:
        return False

    def __mul__(self, other: object) -> md_infp:
        return inf  # type: ignore[return-value]

    def __int__(self) -> int:
        return self  # type: ignore[return-value]

    def __float__(self) -> float:
        return float("inf")

    def __repr__(self) -> str:
        return "inf"


class md_infn:
    def __init__(self) -> None:
        self.value = "-infinity!"

    def __eq__(self, other: object) -> bool:
        return self  # type: ignore[return-value]

    def __gt__(self, other: object) -> bool:
        return False

    def __lt__(self, other: object) -> bool:
        return True

    def __mul__(self, other: object) -> md_infp:
        return inf  # type: ignore[return-value]

    def __int__(self) -> int:
        return self  # type: ignore[return-value]

    def __float__(self) -> float:
        return float("-inf")

    def __repr__(self) -> str:
        return "-inf"


_infp = md_infp()
_infn = md_infn()


class md_inf(md_infp):
    def __init__(self) -> None:
        super().__init__()

    def __eq__(self, other: object) -> bool:
        return bool(isinstance(other, md_inf))

    def __gt__(self, other: object) -> bool:
        return not isinstance(other, md_inf)

    def __lt__(self, other: object) -> bool:
        return False

    def __mul__(self, other: object) -> md_infp | md_infn:
        if other < 0:  # type: ignore[operator]
            return _infn
        else:
            return _infp

    def __repr__(self) -> str:
        return "inf"


nan = md_nan()
inf = md_inf()
