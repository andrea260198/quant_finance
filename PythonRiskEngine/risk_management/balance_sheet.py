from support.quant_dataclass import QuantDataclass
import polars as pl


class BalanceSheet(QuantDataclass):
    time: int
    df: pl.DataFrame

class StartingBalanceSheet(BalanceSheet):
    time: int = 0

class EndingBalanceSheet(BalanceSheet):
    time: int = 1