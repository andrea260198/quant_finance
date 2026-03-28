from risk_management.balance_sheet import StartingBalanceSheet
from pricing_engine.support.quant_dataclass import ImmutableDataclass
import polars as pl

from pricing_engine.support.underlying import Underlying


class BalanceSheetReader(ImmutableDataclass):
    filename: str

    def run(self) -> StartingBalanceSheet:
        # TODO: obtain contract values from input spreadsheet.
        df = pl.read_csv(self.filename)

        option1 = {'S_0': 100.0, 'T': 10.0, 'r': 0.2, 'sigma': 0.2, 'strike': 100.0, 'type': 'EuropeanCallOption'}
        option2 = {'S_0': 100.0, 'T': 10.0, 'r': 0.2, 'sigma': 0.2, 'strike': 120.0, 'type': 'EuropeanCallOption'}

        df = pl.DataFrame(data={"BalanceSheetItem": [option1, option2]})

        return StartingBalanceSheet(df=df)
