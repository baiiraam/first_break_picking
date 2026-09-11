#!/usr/bin/env python3
# file location: scripts/verify_mlflow_routing.py

"""
Verify MLflow routing and run-ID propagation (C.1.1).

Checks:
    1. get_mlflow_manager() is a singleton — same args → same instance
    2. run_id is accessible from the manager after start_run()
    3. The run lands in the correct experiment (seismic-fbp-training)
    4. get_mlflow_manager() with a DIFFERENT experiment name returns
       a DIFFERENT instance (no false singleton)

Does not require training. Runs in ~2 seconds.

Usage:
    python scripts/verify_mlflow_routing.py
"""

import os
import sys
from datetime import datetime, timezone

import click
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import (
    EXPERIMENT_EVALUATION,
    EXPERIMENT_TRAINING,
)

# ============================================================
# CHECKS
# ============================================================


def check_singleton() -> bool:
    print()
    print("=" * 80)
    print("TEST 1 — get_mlflow_manager() is a singleton")
    print("=" * 80)

    a = get_mlflow_manager(EXPERIMENT_TRAINING)
    b = get_mlflow_manager(EXPERIMENT_TRAINING)
    c = get_mlflow_manager(EXPERIMENT_EVALUATION)
    d = get_mlflow_manager(EXPERIMENT_EVALUATION)

    passed = True

    if a is b:
        print("  ✅ Same experiment → same instance")
    else:
        print("  ❌ Same experiment returned DIFFERENT instances")
        passed = False

    if c is d:
        print("  ✅ Same (different) experiment → same instance")
    else:
        print(
            "  ❌ Different experiment returned DIFFERENT instances for the same name"
        )
        passed = False

    if a is not c:
        print("  ✅ Different experiments → different instances")
    else:
        print("  ❌ Different experiments returned the SAME instance (bug)")
        passed = False

    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


def check_run_id_propagation() -> bool:
    print()
    print("=" * 80)
    print("TEST 2 — run_id is accessible after start_run()")
    print("=" * 80)

    manager = get_mlflow_manager(EXPERIMENT_TRAINING)

    # Start a dummy run
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = manager.start_run(
        config_dict={"purpose": "verify_mlflow_routing"},
        run_name=f"verify_routing_{ts}",
        tags={
            "verify_smoke_test": "true",
            "dataset": "_internal_",
            "model_type": "VerifyMLflowRouting",
            "phase": "smoke-test",
            "env": "research",
        },
    )

    passed = True

    if run_id:
        print(f"  ✅ start_run returned run_id: {run_id}")
    else:
        print("  ❌ start_run returned empty run_id")
        passed = False

    # The SAME instance should now have that run_id
    same_manager = get_mlflow_manager(EXPERIMENT_TRAINING)
    if same_manager.run_id == run_id:
        print("  ✅ Same manager instance exposes run_id")
    else:
        print(
            f"  ❌ Manager returned run_id={same_manager.run_id!r}, expected {run_id!r}"
        )
        passed = False

    # End the run cleanly
    manager.end_run()

    # After end_run, run_id should be cleared or kept; either way,
    # the important part is that we had access to it DURING the run.

    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed, run_id


def check_routing(run_id: str) -> bool:
    print()
    print("=" * 80)
    print("TEST 3 — Run landed in the correct experiment")
    print("=" * 80)

    client = mlflow.MlflowClient()

    try:
        run = client.get_run(run_id)
    except mlflow.MlflowException as e:
        print(f"  ❌ Could not fetch run: {e}")
        return False

    experiment_id = run.info.experiment_id
    experiment = client.get_experiment(experiment_id)
    experiment_name = experiment.name

    print(f"  Run ID:          {run_id}")
    print(f"  Experiment ID:   {experiment_id}")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Expected:        {EXPERIMENT_TRAINING}")

    passed = experiment_name == EXPERIMENT_TRAINING
    if passed:
        print("  ✅ Run is in the correct experiment")
    else:
        print("  ❌ Run is in WRONG experiment")

    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--keep-smoke-run",
    is_flag=True,
    default=False,
    help="Keep the dummy smoke-test run in MLflow (default: keep anyway, just tagged).",
)
def main(keep_smoke_run: bool) -> None:
    print("=" * 80)
    print("VERIFY MLFLOW ROUTING (C.1.1)")
    print("=" * 80)

    if not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"  Training experiment:   {EXPERIMENT_TRAINING}")
    print(f"  Evaluation experiment: {EXPERIMENT_EVALUATION}")

    results = []

    # 1. Singleton behavior
    ok_singleton = check_singleton()
    results.append(("get_mlflow_manager() is a singleton", ok_singleton))

    # 2. Run ID propagation
    ok_run_id, smoke_run_id = check_run_id_propagation()
    results.append(("run_id propagated to manager", ok_run_id))

    # 3. Routing
    if smoke_run_id:
        ok_routing = check_routing(smoke_run_id)
        results.append(("run routed to correct experiment", ok_routing))
    else:
        results.append(("run routed to correct experiment", False))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    if smoke_run_id:
        print()
        print(f"  Smoke-test run ID: {smoke_run_id}")
        print("  (tagged verify_smoke_test=true; filter or delete it in the UI)")

    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED — C.1.1 is correct.")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
