import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub

dagshub.init(repo_owner='gaur8126', repo_name='Mlfow-Experiment-Tracking', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/gaur8126/Mlfow-Experiment-Tracking.mlflow')

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

wine = load_wine()

x= wine.data
y = wine.target

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.10, random_state=42)

max_depth = 5
n_estimators = 25

mlflow.set_experiment("new_exp")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train,y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimator", n_estimators)

    ## Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Metrix")

    # save plot
    plt.savefig("Confusion-metrix.png")


    #log artifacts using mlflow
    mlflow.log_artifact("Confusion-metrix.png")
    mlflow.log_artifact(__file__)

    #Tags 
    mlflow.set_tags({"Author":"Vikash","Project":"Wine Classification"})

    #Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    

    print(accuracy)
