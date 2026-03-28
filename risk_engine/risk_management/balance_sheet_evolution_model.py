from risk_management.balance_sheet import BalanceSheet, EndingBalanceSheet
from pricing_engine.support.quant_dataclass import ImmutableDataclass
import polars as pl


class BalanceSheetEvolutionModel(ImmutableDataclass):
    starting_balance_sheet: BalanceSheet
    scenarios: pl.DataFrame

    def run(self) -> BalanceSheet:
        df = self.starting_balance_sheet.df

        # TODO: Update underlying, expiry, r... values based on scenarios for every contract

        df = pl.DataFrame()
        return EndingBalanceSheet(df=df)