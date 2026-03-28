from pricing_engine.core.contract import PricingMethod
from pricing_engine.core.interest_rates.plot_yield_curve import get_approx_and_exact_yield_curves, get_yield_curve
from pricing_engine.core.interest_rates.short_rate_models import VasicekModel


def test_get_yield_curve() -> None:
    TT = [1., 2., 3.]
    short_rate_model = VasicekModel(
            dt= 0.01,
            a= 0.1,
            b= 0.07,
            r0= 0.02,
            sigma= 0.02,
    )
    yy = get_yield_curve(TT, short_rate_model, PricingMethod.EXACT)
    assert yy == [0.022356817111394145, 0.024452539955681075, 0.026320995225858134]


def test_plot_yield_curve() -> None:
    TT = [1., 2., 3.]
    yy_approx, yy_exact = get_approx_and_exact_yield_curves(TT)