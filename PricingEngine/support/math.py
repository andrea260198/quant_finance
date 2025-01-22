from typing import TypeVar


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def calc_cartesian_product(x: iter[T1], y: iter[T2]) -> list[list[tuple[T1, T2]]]:  # type: ignore
    I = len(x)
    J = len(y)
    X = [[(x[i], y[j]) for j in range(J)] for i in range(I)]  # type: ignore
    return X