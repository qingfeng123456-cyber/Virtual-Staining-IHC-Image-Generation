"""Synchronized geometric transforms and input-only intensity augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.nn.functional as functional

TensorTargets = TypeVar("TensorTargets", torch.Tensor, dict[str, torch.Tensor])


def _validate_image_tensor(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2:
        raise ValueError(f"Image tensor needs at least two dimensions, got {tuple(tensor.shape)}")


def apply_d4(tensor: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Apply one of the eight exact dihedral transforms on the last two axes."""

    _validate_image_tensor(tensor)
    if transform_id not in range(8):
        raise ValueError(f"D4 transform_id must be in [0, 7], got {transform_id}")
    value = torch.flip(tensor, dims=(-1,)) if transform_id >= 4 else tensor
    return torch.rot90(value, k=transform_id % 4, dims=(-2, -1))


def invert_d4(tensor: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Invert :func:`apply_d4` exactly, including for non-square tensors."""

    if transform_id not in range(8):
        raise ValueError(f"D4 transform_id must be in [0, 7], got {transform_id}")
    inverse_id = (-transform_id) % 4 if transform_id < 4 else transform_id
    return apply_d4(tensor, inverse_id)


def d4_transform(tensor: torch.Tensor, transform_id: int) -> torch.Tensor:
    return apply_d4(tensor, transform_id)


def d4_inverse(tensor: torch.Tensor, transform_id: int) -> torch.Tensor:
    return invert_d4(tensor, transform_id)


def _random_float(generator: torch.Generator | None, device: torch.device) -> float:
    sampling_device = device if device.type == "cuda" else torch.device("cpu")
    return float(torch.rand((), generator=generator, device=sampling_device).item())


def _random_int(
    low: int,
    high: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    sampling_device = device if device.type == "cuda" else torch.device("cpu")
    return int(torch.randint(low, high, (), generator=generator, device=sampling_device).item())


def _uniform(
    low: float,
    high: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> float:
    return low + (high - low) * _random_float(generator, device)


def _translate(tensor: torch.Tensor, dx: int, dy: int, padding_mode: str = "reflect") -> torch.Tensor:
    if dx == 0 and dy == 0:
        return tensor
    _validate_image_tensor(tensor)
    height, width = tensor.shape[-2:]
    margin = max(abs(dx), abs(dy))
    if margin >= min(height, width):
        raise ValueError(f"Translation {dx, dy} is too large for image size {width}x{height}")
    leading_shape = tensor.shape[:-2]
    value = tensor.reshape(-1, 1, height, width)
    padded = functional.pad(value, (margin, margin, margin, margin), mode=padding_mode)
    y_start = margin - dy
    x_start = margin - dx
    shifted = padded[..., y_start : y_start + height, x_start : x_start + width]
    return shifted.reshape(*leading_shape, height, width)


def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return tensor
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    channels = tensor.shape[-3] if tensor.ndim >= 3 else 1
    weight = kernel.expand(channels, 1, -1, -1)
    original_shape = tensor.shape
    value = tensor.reshape(-1, channels, tensor.shape[-2], tensor.shape[-1])
    blurred = functional.conv2d(value, weight, padding=radius, groups=channels)
    return blurred.reshape(original_shape)


def _map_targets(targets: TensorTargets, operation: object) -> TensorTargets:
    if not callable(operation):
        raise TypeError("Target operation must be callable")
    if isinstance(targets, torch.Tensor):
        return operation(targets)  # type: ignore[return-value]
    if isinstance(targets, Mapping):
        return {marker: operation(tensor) for marker, tensor in targets.items()}  # type: ignore[return-value]
    raise TypeError(f"Targets must be a tensor or mapping, got {type(targets).__name__}")


@dataclass
class PairedTransform:
    """Training augmentation with synchronized geometry and DAPI-only intensity."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_probability: float = 1.0
    max_translation: int = 0
    gamma_range: tuple[float, float] = (0.9, 1.1)
    gamma_probability: float = 0.5
    brightness_delta: float = 0.05
    brightness_probability: float = 0.5
    contrast_range: tuple[float, float] = (0.95, 1.05)
    contrast_probability: float = 0.5
    noise_std: float = 0.005
    noise_probability: float = 0.25
    blur_sigma_range: tuple[float, float] = (0.2, 0.6)
    blur_probability: float = 0.0

    def __post_init__(self) -> None:
        probabilities = {
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "vertical_flip_probability": self.vertical_flip_probability,
            "rotate_probability": self.rotate_probability,
            "gamma_probability": self.gamma_probability,
            "brightness_probability": self.brightness_probability,
            "contrast_probability": self.contrast_probability,
            "noise_probability": self.noise_probability,
            "blur_probability": self.blur_probability,
        }
        invalid = {name: value for name, value in probabilities.items() if not 0.0 <= value <= 1.0}
        if invalid:
            raise ValueError(f"Probabilities must lie in [0, 1]: {invalid}")
        if self.max_translation < 0:
            raise ValueError("max_translation cannot be negative")
        for name, bounds in (("gamma_range", self.gamma_range), ("contrast_range", self.contrast_range)):
            if bounds[0] <= 0.0 or bounds[1] < bounds[0]:
                raise ValueError(f"Invalid {name}: {bounds}")

    def __call__(
        self,
        image: torch.Tensor,
        targets: TensorTargets,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, TensorTargets]:
        _validate_image_tensor(image)
        device = image.device
        transformed_image = image
        transformed_targets = targets
        if _random_float(generator, device) < self.horizontal_flip_probability:
            transformed_image = torch.flip(transformed_image, dims=(-1,))
            transformed_targets = _map_targets(transformed_targets, lambda value: torch.flip(value, dims=(-1,)))
        if _random_float(generator, device) < self.vertical_flip_probability:
            transformed_image = torch.flip(transformed_image, dims=(-2,))
            transformed_targets = _map_targets(transformed_targets, lambda value: torch.flip(value, dims=(-2,)))
        if _random_float(generator, device) < self.rotate_probability:
            turns = _random_int(0, 4, generator, device)
            transformed_image = torch.rot90(transformed_image, k=turns, dims=(-2, -1))
            transformed_targets = _map_targets(
                transformed_targets,
                lambda value: torch.rot90(value, k=turns, dims=(-2, -1)),
            )
        if self.max_translation > 0:
            dx = _random_int(-self.max_translation, self.max_translation + 1, generator, device)
            dy = _random_int(-self.max_translation, self.max_translation + 1, generator, device)
            transformed_image = _translate(transformed_image, dx, dy)
            transformed_targets = _map_targets(
                transformed_targets,
                lambda value: _translate(value, dx, dy),
            )

        intensity = transformed_image.to(dtype=torch.float32)
        if _random_float(generator, device) < self.gamma_probability:
            gamma = _uniform(*self.gamma_range, generator, device)
            intensity = intensity.clamp_min(0.0).pow(gamma)
        if _random_float(generator, device) < self.contrast_probability:
            contrast = _uniform(*self.contrast_range, generator, device)
            mean = intensity.mean(dim=(-2, -1), keepdim=True)
            intensity = (intensity - mean) * contrast + mean
        if _random_float(generator, device) < self.brightness_probability:
            brightness = _uniform(-self.brightness_delta, self.brightness_delta, generator, device)
            intensity = intensity + brightness
        if self.noise_std > 0.0 and _random_float(generator, device) < self.noise_probability:
            noise = torch.randn(
                intensity.shape,
                dtype=intensity.dtype,
                device=intensity.device,
                generator=generator,
            )
            intensity = intensity + noise * self.noise_std
        if _random_float(generator, device) < self.blur_probability:
            sigma = _uniform(*self.blur_sigma_range, generator, device)
            intensity = _gaussian_blur(intensity, sigma)
        return intensity.clamp(0.0, 1.0), transformed_targets


TrainPairedTransform = PairedTransform


def transform_context_offsets(offsets: torch.Tensor, transform_id: int) -> torch.Tensor:
    """Transform ``(row_offset, col_offset)`` vectors with a D4 operation."""

    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError("context offsets must be a [N, 2] tensor")
    if transform_id not in range(8):
        raise ValueError(f"D4 transform_id must be in [0, 7], got {transform_id}")
    transformed = offsets.clone()
    row = transformed[:, 0]
    col = transformed[:, 1]
    if transform_id >= 4:
        col = -col
    for _ in range(transform_id % 4):
        row, col = -col, row
    return torch.stack((row, col), dim=1)


def apply_context_d4(
    context_tiles: torch.Tensor,
    context_valid_mask: torch.Tensor,
    context_offsets: torch.Tensor,
    transform_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply D4 to context pixels and reorder tiles/mask into canonical slots.

    The returned offsets retain the input slot order.  For a canonical 3x3 grid,
    this means an hflip exchanges left/right tiles, a vflip exchanges top/bottom
    tiles, and rotations move every tile and mask to its geometrically correct slot.
    """

    if context_tiles.ndim < 3:
        raise ValueError("context_tiles must have shape [N, ..., H, W]")
    count = context_tiles.shape[0]
    if context_valid_mask.ndim != 1 or context_valid_mask.shape[0] != count:
        raise ValueError("context_valid_mask must have shape [N]")
    if context_offsets.shape != (count, 2):
        raise ValueError("context_offsets must have shape [N, 2]")
    slot_by_offset: dict[tuple[int, int], int] = {}
    for index, offset in enumerate(context_offsets.detach().cpu().tolist()):
        key = (int(offset[0]), int(offset[1]))
        if key in slot_by_offset:
            raise ValueError(f"Duplicate context offset: {key}")
        slot_by_offset[key] = index
    moved_offsets = transform_context_offsets(context_offsets, transform_id)
    transformed_pixels = apply_d4(context_tiles, transform_id)
    output_tiles = torch.empty_like(transformed_pixels)
    output_mask = torch.empty_like(context_valid_mask)
    assigned: set[int] = set()
    for source_index, moved in enumerate(moved_offsets.detach().cpu().tolist()):
        key = (int(moved[0]), int(moved[1]))
        if key not in slot_by_offset:
            raise ValueError(
                "Context offsets are not closed under the requested D4 transform: "
                f"missing slot {key}"
            )
        destination_index = slot_by_offset[key]
        output_tiles[destination_index] = transformed_pixels[source_index]
        output_mask[destination_index] = context_valid_mask[source_index]
        assigned.add(destination_index)
    if len(assigned) != count:
        raise ValueError("Context transform did not assign every output slot exactly once")
    return output_tiles, output_mask, context_offsets.clone()


def transform_context_sample(
    image: torch.Tensor,
    context_tiles: torch.Tensor,
    context_valid_mask: torch.Tensor,
    context_offsets: torch.Tensor,
    targets: TensorTargets,
    transform_id: int,
) -> dict[str, object]:
    """Synchronously transform center, targets, context pixels, offsets, and mask."""

    transformed_tiles, transformed_mask, transformed_offsets = apply_context_d4(
        context_tiles,
        context_valid_mask,
        context_offsets,
        transform_id,
    )
    return {
        "input": apply_d4(image, transform_id),
        "targets": _map_targets(targets, lambda value: apply_d4(value, transform_id)),
        "context_tiles": transformed_tiles,
        "context_valid_mask": transformed_mask,
        "context_offsets": transformed_offsets,
    }


@dataclass
class ContextPairedTransform:
    """Random synchronized D4 geometry and shared-parameter DAPI augmentation."""

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotate_probability: float = 1.0
    gamma_range: tuple[float, float] = (0.9, 1.1)
    gamma_probability: float = 0.5
    brightness_delta: float = 0.05
    brightness_probability: float = 0.5
    contrast_range: tuple[float, float] = (0.95, 1.05)
    contrast_probability: float = 0.5
    noise_std: float = 0.005
    noise_probability: float = 0.25
    blur_sigma_range: tuple[float, float] = (0.2, 0.6)
    blur_probability: float = 0.0

    def __post_init__(self) -> None:
        self._intensity_transform = PairedTransform(
            horizontal_flip_probability=0.0,
            vertical_flip_probability=0.0,
            rotate_probability=0.0,
            gamma_range=self.gamma_range,
            gamma_probability=self.gamma_probability,
            brightness_delta=self.brightness_delta,
            brightness_probability=self.brightness_probability,
            contrast_range=self.contrast_range,
            contrast_probability=self.contrast_probability,
            noise_std=self.noise_std,
            noise_probability=self.noise_probability,
            blur_sigma_range=self.blur_sigma_range,
            blur_probability=self.blur_probability,
        )
        for name in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "rotate_probability",
        ):
            probability = float(getattr(self, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {probability}")

    def __call__(
        self,
        image: torch.Tensor,
        context_tiles: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_offsets: torch.Tensor,
        targets: TensorTargets,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, object]:
        device = image.device
        horizontal = _random_float(generator, device) < self.horizontal_flip_probability
        vertical = _random_float(generator, device) < self.vertical_flip_probability
        turns = (
            _random_int(0, 4, generator, device)
            if _random_float(generator, device) < self.rotate_probability
            else 0
        )
        if horizontal and vertical:
            transform_id = (turns + 2) % 4
        elif horizontal:
            transform_id = 4 + turns
        elif vertical:
            transform_id = 4 + ((turns + 2) % 4)
        else:
            transform_id = turns
        sample = transform_context_sample(
            image,
            context_tiles,
            context_valid_mask,
            context_offsets,
            targets,
            transform_id,
        )
        combined = torch.cat(
            (
                sample["input"].unsqueeze(0),  # type: ignore[union-attr]
                sample["context_tiles"],  # type: ignore[arg-type]
            ),
            dim=0,
        )
        augmented, _ = self._intensity_transform(combined, {}, generator=generator)
        sample["input"] = augmented[0]
        sample["context_tiles"] = augmented[1:]
        return sample


@dataclass(frozen=True)
class DeterministicContextTransform:
    """Deterministic context geometry for validation and unit tests."""

    transform_id: int = 0

    def __post_init__(self) -> None:
        if self.transform_id not in range(8):
            raise ValueError("transform_id must be in [0, 7]")

    def __call__(
        self,
        image: torch.Tensor,
        context_tiles: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_offsets: torch.Tensor,
        targets: TensorTargets,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, object]:
        del generator
        return transform_context_sample(
            image,
            context_tiles,
            context_valid_mask,
            context_offsets,
            targets,
            self.transform_id,
        )


@dataclass
class DeterministicPairedTransform(PairedTransform):
    """Geometry-only deterministic transform useful for tests and validation."""

    transform_id: int = 0

    def __post_init__(self) -> None:
        if self.transform_id not in range(8):
            raise ValueError("transform_id must be in [0, 7]")

    def __call__(
        self,
        image: torch.Tensor,
        targets: TensorTargets,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, TensorTargets]:
        del generator
        return apply_d4(image, self.transform_id), _map_targets(
            targets, lambda value: apply_d4(value, self.transform_id)
        )
