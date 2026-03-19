from support.quant_dataclass import ImmutableDataclass
import polars as pl


class BalanceSheet(ImmutableDataclass):
    time: int
    df: pl.DataFrame

    def price(self):
        # TODO: Fix the preliminary sketch below
        # A solution could be to add a column with the price, next to every BSI
        # Price shall be negative for liabilities
        # Remember to calculate the present value as of t=0
        for row in self.df:
            row[1] = row[0].price()




class StartingBalanceSheet(BalanceSheet):
    time: int = 0

class EndingBalanceSheet(BalanceSheet):
    time: int = 1