from .inference import IrisInference, IrisPrediction
from .train import get_classes, load_model


def get_input_data(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
) -> IrisInference:
    # Input data via pydantic model
    input_data = IrisInference(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
    )
    return input_data


def predict_results(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
) -> IrisPrediction:
    try:
        # Input data via pydantic model
        input_data = get_input_data(
            sepal_length, sepal_width, petal_length, petal_width
        )

        # Load model
        model = load_model("app/artifacts/iris_model.joblib")

        # Perform predictions
        data = [
            [
                input_data.sepal_length,
                input_data.sepal_width,
                input_data.petal_length,
                input_data.petal_width,
            ]
        ]

        classes = get_classes()

        pred = model.predict(data)[0]
        label = str(classes[pred])

        probabilities = model.predict_proba(data)[0]

        return IrisPrediction(
            prediction=label,
            probability={
                classes[i]: round(prob, 4) for i, prob in enumerate(probabilities)
            },
        )

    except Exception as e:
        print(f"Exception occured : {e}")


if __name__ == "__main__":
    # Example usage
    results = predict_results(6.6, 3.0, 5.0, 1.7)
    print(results)
