import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from catboost import CatBoostClassifier
from pprint import pprint
from statsmodels.stats.outliers_influence import variance_inflation_factor

def handle_categorical(data, cat_col_names, categories):
    for col, category in categories.items():
        data[col] = pd.Categorical(data[col], categories=category)
    
    data = pd.get_dummies(data, columns=cat_col_names, dtype=np.int8)
    return data

def run_classifier(
    data, classifier, categorical_features, verbose=False, fit_kwargs={}
):
    X = data.drop(["Target"], axis=1)
    Y = data["Target"].to_numpy()
    categories = {}
    for feature in categorical_features:
        categories[feature] = X[feature].cat.categories
    

    model = classifier(n_estimators=100, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    convert_categorical = lambda x: handle_categorical(x, categorical_features, categories)


    model.fit(convert_categorical(X_train), Y_train, **fit_kwargs)


    # predict on test set
    Y_pred = model.predict(convert_categorical(X_test))
    # print(Y_pred)
    # calculate f1 score
    base_f1 = f1_score(Y_test, Y_pred, average="weighted")
    permuted_f1s = {}
    for feature in X_test.columns:
        X_test_permuted = X_test.copy(True)
        X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])
        Y_pred = model.predict(
            convert_categorical(X_test_permuted)
        )
        f1 = f1_score(Y_test, Y_pred, average="weighted")
        permuted_f1s[feature] = f1

    top_k_imp_features = [[i[0], base_f1 - i[1]] for i in permuted_f1s.items()]
    top_k_imp_features = sorted(top_k_imp_features, key=lambda x: x[1], reverse=True)[:7]

    if verbose:
        acc_score = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average="weighted")
        recall = recall_score(Y_test, Y_pred, average="weighted")
        print("Accuracy score: ", acc_score)
        print("Precision score: ", precision)
        print("Recall score: ", recall)
        print("F1 score: ", base_f1)
        print(classification_report(Y_test, Y_pred))
        print(confusion_matrix(Y_test, Y_pred))

    return {
        "base_f1": base_f1,
        "permuted_f1s": permuted_f1s,
        "top_k_imp_features": top_k_imp_features,
        "classifier": classifier.__name__,
    }

def make_bar_plot(ret, title=None):
    classifier_name = ret["classifier"]
    top_k_imp_features = ret["top_k_imp_features"][::-1]
    pprint(top_k_imp_features)
    plt.figure(figsize=(6, 5))
    plt.barh([i[0] for i in top_k_imp_features], [i[1] for i in top_k_imp_features])
    plt.tight_layout(pad=2)
    plt.title(f"{classifier_name}")
    plt.savefig(f"top_k_imp_features_{classifier_name}.png")
    plt.clf()
    plt.close()

def feature_selection(data, categorical_features, threshold=5):
    # Perform feature selection using VIF (thresholod = 5)
    X = data.drop(["Target"], axis=1)
    # Convert binary catergorical features to binary numerical and drop the rest
    for feature in categorical_features:
        if len(X[feature].unique()) == 2:
            X[feature] = pd.Categorical(X[feature]).codes
        else:
            X = X.drop([feature], axis=1)
    
    # Calculate VIF
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    vif = vif.sort_values("VIF Factor", ascending=False)
    # print(vif)

    # Drop features with VIF > FS_THRESHOLD
    while vif["VIF Factor"].max() > threshold:
        idx = vif["VIF Factor"].idxmax()
        X = X.drop(X.columns[idx], axis=1)
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
    
    # Add target column back
    X["Target"] = data["Target"]

    print(f"Features selected: {X.columns}")

def main():
    data = pd.read_csv("dataset.csv")
    categorical_features: list[str] = pickle.load(
        open("categorical_features.pkl", "rb")
    )
    for feature in categorical_features:
        data[feature] = pd.Categorical(data[feature])
    
    # # make correlation matrix and plot heatmap
    # cormat = data.drop(["Target"], axis=1).corr()
    # plt.subplots(figsize=(10, 10))
    # sns.heatmap(cormat, linewidth=0, yticklabels = cormat.columns, xticklabels = cormat.columns, cmap="RdBu")
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig("correlation_matrix.png")
    # plt.close()
    # plt.clf()


    cormat = data.drop(["Target"], axis=1).corr()
    plt.subplots(figsize=(10, 10))
    sns.heatmap(cormat, linewidth=0, yticklabels = cormat.columns, xticklabels = cormat.columns, cmap="RdBu")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()
    plt.clf()

    categorical_features.remove("Target")

    ret = run_classifier(data, RandomForestClassifier, categorical_features, verbose=True)
    pprint(ret)
    make_bar_plot(ret)

    ret = run_classifier(data, GradientBoostingClassifier, categorical_features, verbose=True)
    pprint(ret)
    make_bar_plot(ret)    

    ret = run_classifier(data, CatBoostClassifier, categorical_features, fit_kwargs={"verbose": False}, verbose=True)
    make_bar_plot(ret)
    pprint(ret)

    # Feature Selection
    FS_THRESHOLD = 5
    print(f"FS_THRESHOLD: {FS_THRESHOLD}")
    feature_selection(data, categorical_features, threshold=FS_THRESHOLD)

if __name__ == "__main__":
    main()


