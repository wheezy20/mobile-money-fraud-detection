"""End-to-end training pipeline with conditional deployment."""
from zenml import pipeline
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

from pipelines.steps.data_loader import load_data
from pipelines.steps.preprocessor import preprocess_data
from pipelines.steps.trainer import train_model
from pipelines.steps.evaluator import evaluate_model
from pipelines.steps.deployer import deployment_trigger


@pipeline(enable_cache=True)
def training_pipeline(
    data_path: str = "data/raw/transactions.csv",
    n_estimators: int = 200,
    max_depth: int = 15,
    min_pr_auc: float = 0.85,
):
    """Load -> Preprocess -> Train -> Evaluate -> (Conditional Deploy)"""
    df = load_data(data_path=data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df=df)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    pr_auc = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    deploy_decision = deployment_trigger(pr_auc=pr_auc, min_pr_auc=min_pr_auc)
    
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=1,
        timeout=120,
    )
