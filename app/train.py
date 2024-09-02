from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


def train_and_save_model():
    # Load iris data
    iris = load_iris()

    # Get X and y features
    X, y = iris.data, iris.target

    # Train test split
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Create pipeline
    pipe = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression()
    )

    # Fit the pipeline
    pipe.fit(xtrain, ytrain)

    # Save model as joblib
    path = "./artifacts/iris_model.joblib"
    joblib.dump(pipe, path)
    print(f"Model saved successfully in path : {path}")


def get_classes() -> list[str]:
    iris = load_iris()
    return iris.target_names


def load_model(path: str = "./artifacts/iris_model.joblib"):
    model = joblib.load(path)
    return model


if __name__ == "__main__":
    print(get_classes())
    train_and_save_model()
