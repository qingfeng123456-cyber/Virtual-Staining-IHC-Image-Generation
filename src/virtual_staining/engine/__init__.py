"""Training, validation, inference, checkpoint, and ensemble utilities."""

from .ablation_budget import (
    AblationBudgetPlan,
    EvidenceRun,
    bind_evidence_records,
    build_promotion_provenance,
    resolve_ablation_budget,
)
from .checkpoint import load_checkpoint, resume_from_checkpoint, save_checkpoint
from .complexity_report import ComplexityRunInput, generate_complexity_report
from .ema import ExponentialMovingAverage
from .gradient_monitor import (
    GradientCosineMonitor,
    compute_gradient_cosine_report,
    compute_task_gradient_vectors,
)
from .inferencer import Inferencer, apply_d4, invert_d4
from .pretrainer import transfer_local_encoder, transfer_local_encoder_from_checkpoint
from .prototype_diagnostics import (
    PrototypeDiagnosticsAggregator,
    write_prototype_diagnostics,
)
from .prototype_monitor import PrototypeUsageMonitor
from .trainer import Trainer, collect_optimizer_parameters, loss_optimizer_parameters
from .validator import Validator

__all__ = [
    "AblationBudgetPlan",
    "EvidenceRun",
    "ExponentialMovingAverage",
    "ComplexityRunInput",
    "GradientCosineMonitor",
    "Inferencer",
    "PrototypeDiagnosticsAggregator",
    "PrototypeUsageMonitor",
    "Trainer",
    "Validator",
    "apply_d4",
    "bind_evidence_records",
    "build_promotion_provenance",
    "collect_optimizer_parameters",
    "compute_gradient_cosine_report",
    "compute_task_gradient_vectors",
    "invert_d4",
    "generate_complexity_report",
    "load_checkpoint",
    "loss_optimizer_parameters",
    "resume_from_checkpoint",
    "resolve_ablation_budget",
    "save_checkpoint",
    "transfer_local_encoder",
    "transfer_local_encoder_from_checkpoint",
    "write_prototype_diagnostics",
]
