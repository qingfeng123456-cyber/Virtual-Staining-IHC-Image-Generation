"""Data discovery, manifests, datasets, auditing, and paired transforms."""

from .activity import compute_training_activity
from .dataset import VirtualStainingDataset, marker_path_column
from .discovery import DiscoveryResult, discover_data_root
from .grouped_folds import (
    InnerFoldAssignment,
    ROIGroupedFoldResult,
    build_authoritative_roi_folds,
)
from .loader_helpers import (
    ActivitySamplingPlan,
    AuthoritativeFoldSplit,
    prepare_activity_sampling,
    prepare_authoritative_fold_split,
    require_verified_filename_grid,
)
from .manifest import ManifestBuildResult, build_manifests
from .neighborhood import NeighborhoodDataset, NeighborhoodInferenceDataset, context_offsets
from .roi_index import (
    BoundaryContinuityAudit,
    CoordinateParseResult,
    PatchCoordinate,
    ROIGridAudit,
    ROIGridSummary,
    ROIIndex,
    assess_context_eligibility,
    audit_roi_grid,
    parse_patch_coordinate,
    parse_patch_coordinate_status,
)
from .samplers import ActivityStratifiedSampler
from .transforms import (
    ContextPairedTransform,
    DeterministicContextTransform,
    PairedTransform,
    TrainPairedTransform,
    apply_context_d4,
    apply_d4,
    invert_d4,
    transform_context_offsets,
    transform_context_sample,
)

__all__ = [
    "ActivityStratifiedSampler",
    "ActivitySamplingPlan",
    "AuthoritativeFoldSplit",
    "BoundaryContinuityAudit",
    "ContextPairedTransform",
    "CoordinateParseResult",
    "DeterministicContextTransform",
    "DiscoveryResult",
    "InnerFoldAssignment",
    "ManifestBuildResult",
    "NeighborhoodDataset",
    "NeighborhoodInferenceDataset",
    "PairedTransform",
    "PatchCoordinate",
    "ROIGridAudit",
    "ROIGridSummary",
    "ROIGroupedFoldResult",
    "ROIIndex",
    "TrainPairedTransform",
    "VirtualStainingDataset",
    "apply_context_d4",
    "apply_d4",
    "assess_context_eligibility",
    "audit_roi_grid",
    "build_authoritative_roi_folds",
    "build_manifests",
    "context_offsets",
    "compute_training_activity",
    "discover_data_root",
    "invert_d4",
    "marker_path_column",
    "parse_patch_coordinate",
    "parse_patch_coordinate_status",
    "prepare_activity_sampling",
    "prepare_authoritative_fold_split",
    "require_verified_filename_grid",
    "transform_context_offsets",
    "transform_context_sample",
]
