"""Entry point for ZenML pipelines."""
import click
from pipelines.training_pipeline import training_pipeline


@click.command()
@click.option("--data-path", default="data/raw/transactions.csv")
@click.option("--n-estimators", default=200, type=int)
@click.option("--max-depth", default=15, type=int)
@click.option("--min-pr-auc", default=0.85, type=float)
def main(data_path: str, n_estimators: int, max_depth: int, min_pr_auc: float):
    training_pipeline(
        data_path=data_path,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_pr_auc=min_pr_auc,
    )
    print("\n[OK] Pipeline complete. Run `zenml up` for dashboard.")


if __name__ == "__main__":
    main()
