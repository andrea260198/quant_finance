from core.interest_rates.plot_yield_curve import get_yield_curves


def test_plot_yield_curve() -> None:
    TT = [1., 2., 3.]
    get_yield_curves(TT)