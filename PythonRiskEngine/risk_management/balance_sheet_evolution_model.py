from risk_management.balance_sheet import BalanceSheet
from support.quant_dataclass import QuantDataclass
import polars as pl


class BalanceSheetEvolutionModel(QuantDataclass):
    starting_balance_sheet: BalanceSheet
    scenarios: pl.DataFrame

    def run(self) -> BalanceSheet:
        df = pl.DataFrame()
        return BalanceSheet(time=1, df=df)