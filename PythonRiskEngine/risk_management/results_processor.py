from risk_management.balance_sheet import EndingBalanceSheet, StartingBalanceSheet
from support.quant_dataclass import QuantDataclass
import polars as pl

class Results(QuantDataclass):
    df: pl.dataframe


class ResultsProcessor(QuantDataclass):
    starting_balance_sheet: StartingBalanceSheet
    ending_balance_sheet: EndingBalanceSheet

    def run(self):
        df = pl.DataFrame()
        return Results(df=df)