"""Backend functions related to correlating and refining particle stacks."""


from typing import Literal

import roma
import torch
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.backend.utils import normalize_template_projection
from tt2dtm.utils.cross_correlation import handle_correlation_mode


def core_refine_template(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    defocus_values: torch.Tensor,  # (N,)
    projective_filters: torch.Tensor,  # (N, h, w)
    euler_angle_offsets: torch.Tensor,  # (3, k)
    defocus_offsets: torch.Tensor,  # (l,)
    batch_size: int = 1024,
    # TODO: additional arguments for cc --> z-score scaling
) -> None:
    """Core function to refine orientations and defoci of a set of particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    defocus_values : torch.Tensor
        The defocus values for each particle in the stack. Shape of (N,).
        NOTE: Will likely also need kwargs to pass to the ctf filter function in here
        somewhere and support multiple distinct images.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (3, k).
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.
    """
    # device = particle_stack_dft.device
    # num_particles, H, W = particle_stack_dft.shape
    # d, h, w = template_dft.shape
    # # account for RFFT
    # W = 2 * (W - 1)
    # w = 2 * (w - 1)

    # for i in range(0, num_particles, batch_size):
    #     # extract batch parameters
    #     for delta_df in defocus_offsets:
    #         # Recompute CTF filters for each particle's absolute defocus
    #         for delta_ea in euler_angle_offsets:
    #             # Rotate Euler angles by delta_ea
    #             # call cross_correlate_particle_stack and update best values
    #             pass

    raise NotImplementedError("This function is not yet implemented.")


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    Here, the argument 'particle_stack_dft' is a set of RFFT-ed particle images with
    necessary filtering already applied. The zeroth dimension corresponds to unique
    particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (num_particles, H - h + 1, W - w + 1)
    elif mode == "same":
        output_shape = (num_particles, H, W)

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_euler_angles = euler_angles[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Convert the Euler angles into rotation matrices
        rot_matrix = roma.euler_to_rotmat(
            "ZYZ", batch_euler_angles, degrees=True, device=device
        )

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=rot_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection(projections, (h, w), (H, W))

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation
