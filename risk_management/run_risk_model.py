from risk_management.risk_model import MyRiskModel, ModelParameters

if __name__ == '__main__':
    model_parameters = ModelParameters(10_000)

    model = MyRiskModel(model_parameters)

    results = model.run()

    print(results)