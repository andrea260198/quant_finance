from risk_management.balance_sheet import EndingBalanceSheet, StartingBalanceSheet
from support.quant_dataclass import ImmutableDataclass
import polars as pl

class Results(ImmutableDataclass):
    df: pl.dataframe


class ResultsProcessor(ImmutableDataclass):
    starting_balance_sheet: StartingBalanceSheet
    ending_balance_sheet: EndingBalanceSheet

    def run(self):
        self.starting_balance_sheet.price()
        self.ending_balance_sheet.price()

        results = self.ending_balance_sheet.df["Price"] - self.starting_balance_sheet.df["Price"]

        return Results(df=results)