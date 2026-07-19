# ROI context design and leakage rules

An authoritative patch name has the form `ROI000_00_00.jpg`.  Parsing is strict,
case-insensitive for the ROI prefix, accepts coordinates wider than two digits,
and never derives coordinates from a numeric stem.  Each coordinate belongs to
exactly one organ, split, and ROI.

The default neighborhood is a row-major 3×3 grid.  A missing position receives a
copy of the center tensor, a false valid-mask entry, and its real relative
offset.  Only DAPI paths are indexed.  A training or validation target is always
the center target; neighbor labels are never loaded as context.

Grid audit reports coordinate ranges, holes, duplicates, split overlap, and
border-strip continuity.  Horizontal candidates compare right-to-left borders;
vertical candidates compare bottom-to-top borders.  Transpose and flip
hypotheses are evaluated explicitly.  Context remains disabled unless filename
coordinates and direction evidence agree.

Horizontal flip negates the column offset, vertical flip negates the row offset,
and 90-degree rotations transform both axes and reorder tiles/masks.  Geometry is
shared by center, context, and target.  DAPI intensity augmentation uses one set
of parameters for the complete neighborhood.

The current anonymous sample has no authoritative coordinates and contains
spatially continuous train/validation boundaries.  Its edge graph is audit
evidence only and cannot enable a promotable context run.

