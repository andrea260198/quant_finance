from risk_management.balance_sheet_evolution_model import BalanceSheetEvolutionModel
from risk_management.balance_sheet_reader import BalanceSheetReader
from risk_management.results_processor import ResultsProcessor
from scenario_generation.scenario_generator import ScenarioGenerator


def run_risk_model():
    scenarios = ScenarioGenerator().run()

    starting_balance_sheet = BalanceSheetReader(filename='inputs/balance_sheet.csv').run()

    ending_balance_sheet = BalanceSheetEvolutionModel(starting_balance_sheet=starting_balance_sheet, scenarios=scenarios).run()

    results = ResultsProcessor(
        starting_balance_sheet=starting_balance_sheet,
        ending_balance_sheet=ending_balance_sheet
    ).run()

    # TODO: Save the results in an output spreadsheet


if __name__ == '__main__':
    run_risk_model()