from train import load_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


def get_test_report():
    iris = load_iris()
    xtrain, xtest, ytrain, ytest = train_test_split(
        iris.data, iris.target, test_size=0.33, random_state=42
    )
    model = load_model()
    ypred_test = model.predict(xtest)
    report = classification_report(ytest, ypred_test)
    print(report)
    cv_scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    print(f"\n5 fold Cross validated average F1 Macro is {cv_scores.mean():.4f}")


if __name__ == "__main__":
    get_test_report()
