"""Step 5: Deployment trigger gates model promotion on quality threshold."""
from zenml import step


@step
def deployment_trigger(pr_auc: float, min_pr_auc: float = 0.85) -> bool:
    """Gate deployment on PR-AUC threshold."""
    deploy = pr_auc >= min_pr_auc
    print(f"PR-AUC: {pr_auc:.4f} | Threshold: {min_pr_auc} | Deploy: {deploy}")
    return deploy
