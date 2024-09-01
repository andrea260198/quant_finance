from risk_management.risk_model import MyRiskModel


if __name__ == '__main__':
    model = MyRiskModel()

    results = model.run()

    print(results)