from risk_management.balance_sheet import StartingBalanceSheet
from support.quant_dataclass import QuantDataclass
import polars as pl


class BalanceSheetReader(QuantDataclass):
    filename: str

    def run(self) -> StartingBalanceSheet:
        df = pl.read_csv(self.filename)
        return StartingBalanceSheet(df=df)
