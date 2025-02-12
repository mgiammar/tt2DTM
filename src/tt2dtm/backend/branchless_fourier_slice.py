"""Implementation extract slices function from torch_fourier_slice that is branchless"""

import torch
import torch.nn.functional as F


def setup_slice_coordinates(
    volume_shape: tuple[int, int, int], batch_size: int
) -> torch.Tensor:
    """Returns coordinates to rotate around for a given volume shape.

    Note that the volume is assumed to be RFFT-ed and FFT-shifted.

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        The shape of the volume.
    batch_size : int
        The batch size of the orientations (how many Fourier slices to take at once).

    Returns
    -------
    torch.Tensor
        The coordinates to rotate around.
    """
    d, h, w = volume_shape

    x = torch.arange(0, w, dtype=torch.float32)
    y = torch.arange(-h // 2, h // 2, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    coordinates = torch.stack((torch.zeros_like(xx), yy, xx), dim=-1)
    coordinates = coordinates.repeat(batch_size, 1, 1, 1)

    return coordinates


def extract_central_slices_rfft_3d_branchless(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,  # (b, 3, 3)
    coordinates: torch.Tensor,
    conjugate_mask: torch.Tensor,  # (b, h, w)
    inside_mask: torch.Tensor,  # (b * h * w)
    outslice: torch.Tensor,  # (b, h, w)
):
    """Branchless 'torch_fourier_filter.extract_central_slices_rfft_3d' function.

    Makes some assumptions, namely that 'volume_rfft' is a rfft-ed and fftshift-ed
    tensor, and that the values to slice are on the grid defined by 'image_shape'.

    Rotation matrices should be passed in xyz format, that is, Rv = v' where v is
    a vector of (x, y, z) coordinates.

    Parameters
    ----------
    volume_rfft : torch.Tensor
        A rfft-ed and fftshift-ed volumetric tensor. Has shape (d, h, w).
    rotation_matrices : torch.Tensor
        A tensor of rotation matrices of shape (batch, 3, 3).
    coordinates : torch.Tensor
        A tensor of coordinates to rotate and use for sampling the volume.
    conjugate_mask : torch.Tensor
        A tensor to store the mask of where the x-coordinate is negative. Has shape
        (batch, h, w).
    inside_mask : torch.Tensor
        A tensor to store the mask of where the coordinates are inside the volume.
        Is flat, that is, has shape (batch * h * w).
    out : torch.Tensor
        A tensor to store the extracted slices. Has shape (batch, h, w).
    """
    # Unpack shapes into helpful variables
    batch = rotation_matrices.shape[0]
    d, h, w = volume_rfft.shape

    # TODO: Remove flip from xyz to zyx to directly sample coordinates after
    # rotation. Some testing will be necessary.

    # Make rotation matrices rotate zyx coordinates (from xyz format)
    rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # Rotate coordinates by rotation matrices
    coordinates = coordinates.reshape(batch, -1, 3)
    coordinates = rotation_matrices.unsqueeze(1) @ coordinates.unsqueeze(-1)
    coordinates = coordinates.squeeze(-1)
    coordinates = coordinates.view(-1, 3)  # flatten into a single dimension

    # Find where half-dim is negative and flip
    conjugate_mask = coordinates[..., 2] < 0
    torch.where(
        conjugate_mask.unsqueeze(1).expand(-1, 3),
        -coordinates,
        coordinates,
        out=coordinates,
    )

    # Shift the coordinates to center based on shape
    coordinates[..., 0] += d // 2
    coordinates[..., 1] += h // 2
    # no shift for x since rfft

    # Create mask to keep only coordinates within the volume (for later use)
    inside_mask = (
        (coordinates[..., 0] <= (d - 1))
        & (coordinates[..., 1] <= (h - 1))
        & (coordinates[..., 2] <= (w - 1))
        & (coordinates[..., 0] >= 0)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 2] >= 0)
    )

    # Modify the coordinates for the grid_sample function ->  range [-1, 1]
    coordinates[..., 0] = (coordinates[..., 0] / (0.5 * d - 0.5)) - 1
    coordinates[..., 1] = (coordinates[..., 1] / (0.5 * h - 0.5)) - 1
    coordinates[..., 2] = (coordinates[..., 2] / (0.5 * w - 0.5)) - 1
    coordinates = torch.flip(coordinates, dims=(-1,))

    # Some reshaping and viewing shenanigans to properly sample the real and complex
    # parts of the volumes simultaneously
    # 1. Creating new real/complex channel in volume_rfft
    # 2. Moving channel dimension to the front
    # 3. Expanding the zeroth-dimension to the batch shape which will be sampled
    # NOTE: torch.Tensor.expand does not copy memory
    # 4. Reshaping the coordinates to have correct batch size and dummy (x, y, z) axes
    volume_rfft = torch.view_as_real(volume_rfft)
    volume_rfft = volume_rfft.permute(3, 0, 1, 2)
    volume_rfft = volume_rfft.expand(coordinates.shape[0], -1, -1, -1, -1)
    coordinates = coordinates.view(-1, 1, 1, 1, 3)

    # Do the interpolation on real and imaginary parts
    tmp = F.grid_sample(
        volume_rfft,
        coordinates,
        mode="bilinear",
        # padding_mode="reflection",
        padding_mode="zeros",
        align_corners=True,
    )
    tmp.squeeze_()  # remove dummy axes
    # tmp.index_fill_(0, torch.where(inside_mask)[0], 0)
    # torch.where(inside_mask, tmp, 0, out=tmp)  # zero out the outside values
    # tmp = torch.where(inside_mask, tmp, torch.zeros_like(tmp))
    tmp = tmp.reshape(batch, h, w, 2)

    # Reconstruct the complex tensor
    torch.view_as_complex_copy(tmp.contiguous(), out=outslice)
    # outslice.view(batch * h * w)[inside_mask] = 0.0
    # outslice.view(batch * h * w)[conjugate_mask] = outslice.view(batch * h * w)[conjugate_mask].conj()
    torch.where(
        conjugate_mask.view(-1),
        outslice.view(-1),
        outslice.view(-1).conj(),
        out=outslice.view(-1),
    )
