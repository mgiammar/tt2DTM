"""Backend functions related to correlating and refining particle stacks."""

from typing import Any, Literal

import roma
import torch
import tqdm
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.backend.core_match_template import _do_bached_orientation_cross_correlate
from tt2dtm.backend.utils import normalize_template_projection
from tt2dtm.utils.cross_correlation import handle_correlation_mode


def core_refine_template(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    euler_angle_offsets: torch.Tensor,  # (3, k)
    defocus_offsets: torch.Tensor,  # (l,)
    defocus_u: torch.Tensor,  # (N,)
    defocus_v: torch.Tensor,  # (N,)
    defocus_angle: torch.Tensor,  # (N,)
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,  # (N, h, w)
    batch_size: int = 64,
    # TODO: additional arguments for cc --> z-score scaling
) -> Any:
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
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (3, k).
    defocus_u : torch.Tensor
        The defocus along the major axis for each particle in the stack. Shape of (N,).
    defocus_v : torch.Tensor
        The defocus along the minor for each particle in the stack. Shape of (N,).
    defocus_angle : torch.Tensor
        The defocus astigmatism angle for each particle in the stack. Shape of (N,).
        Is the same as the defocus for the micrograph the particle came from.
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.
    """
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    # Send other tensors to the same device
    template_dft = template_dft.to(device)
    euler_angles = euler_angles.to(device)
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = defocus_angle.to(device)
    defocus_offsets = torch.tensor(defocus_offsets)
    defocus_offsets = defocus_offsets.to(device)
    projective_filters = projective_filters.to(device)
    euler_angle_offsets = euler_angle_offsets.to(device)

    # Iterate over each particle in the stack
    refined_statistics = []
    for i in range(num_particles):
        particle_image_dft = particle_stack_dft[i]
        particle_index = i

        refined_stats = _core_refine_template_single_thread(
            particle_image_dft=particle_image_dft,
            particle_index=particle_index,
            template_dft=template_dft,
            euler_angles=euler_angles[i, :],
            euler_angle_offsets=euler_angle_offsets,
            defocus_u=defocus_u[i],
            defocus_v=defocus_v[i],
            defocus_angle=defocus_angle[i],
            defocus_offsets=defocus_offsets,
            ctf_kwargs=ctf_kwargs,
            projective_filter=projective_filters[i],
            orientation_batch_size=batch_size,
        )
        refined_statistics.append(refined_stats)

    return refined_statistics


def _core_refine_template_single_thread(
    particle_image_dft: torch.Tensor,
    particle_index: int,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,
    defocus_offsets: torch.Tensor,
    ctf_kwargs: dict,
    projective_filter: torch.Tensor,
    orientation_batch_size: int = 32,
) -> dict[str, float | int]:
    """TODO: docstring."""
    H, W = particle_image_dft.shape
    _, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)
    # valid crop shape
    crop_H = H - h + 1
    crop_W = W - w + 1

    # Output best statistics
    mip = torch.zeros(
        crop_H, crop_W, device=particle_image_dft.device, dtype=torch.float32
    )
    best_phi = torch.zeros(
        crop_H, crop_W, device=particle_image_dft.device, dtype=torch.float32
    )
    best_theta = torch.zeros(
        crop_H, crop_W, device=particle_image_dft.device, dtype=torch.float32
    )
    best_psi = torch.zeros(
        crop_H, crop_W, device=particle_image_dft.device, dtype=torch.float32
    )
    best_defocus = torch.zeros(
        crop_H, crop_W, device=particle_image_dft.device, dtype=torch.float32
    )

    # The "best" Euler angle from the match template program
    default_rot_matrix = roma.euler_to_rotmat(
        "ZYZ", euler_angles, degrees=True, device=particle_image_dft.device
    )
    # default_rot_matrix.to(torch.float32)

    # Calculate the CTF filters with the relative offsets
    defocus = (defocus_u + defocus_v) / 2 + defocus_offsets
    astigmatism = (defocus_u - defocus_v) / 2
    ctf_filters = calculate_ctf_2d(
        defocus=defocus * 1e-4,  # to µm
        astigmatism=astigmatism * 1e-4,  # to µm
        astigmatism_angle=defocus_angle,
        **ctf_kwargs,
    )

    # Combine the single projective filter with the CTF filter
    combined_projective_filter = projective_filter[None, ...] * ctf_filters

    # Setup iterator object with tqdm for progress bar
    num_batches = euler_angle_offsets.shape[0] // orientation_batch_size
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"Refining particle {particle_index}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
    )

    for i in orientation_batch_iterator:
        euler_angle_offsets_batch = euler_angle_offsets[
            i * orientation_batch_size : (i + 1) * orientation_batch_size
        ]
        rot_matrix_batch = roma.euler_to_rotmat(
            "ZYZ",
            euler_angle_offsets_batch,
            degrees=True,
            device=particle_image_dft.device,
        )
        rot_matrix_batch = roma.rotmat_composition(
            (default_rot_matrix, rot_matrix_batch)
        )
        rot_matrix_batch = rot_matrix_batch.to(torch.float32)

        # Cast to float32 for updating statistics
        euler_angle_offsets_batch = euler_angle_offsets_batch.to(torch.float32)

        # Calculate the cross-correlation
        cross_correlation = _do_bached_orientation_cross_correlate(
            image_dft=particle_image_dft,
            template_dft=template_dft,
            rotation_matrices=rot_matrix_batch,
            projective_filters=combined_projective_filter,
        )

        cross_correlation = cross_correlation[..., :crop_H, :crop_W]  # valid crop

        # Update the best refined statistics
        max_values, max_indices = torch.max(
            cross_correlation.view(-1, crop_H, crop_W), dim=0
        )
        max_defocus_idx = max_indices // euler_angle_offsets_batch.shape[0]
        max_orientation_idx = max_indices % euler_angle_offsets_batch.shape[0]

        update_mask = max_values > mip
        torch.where(update_mask, max_values, mip, out=mip)
        torch.where(
            update_mask,
            euler_angle_offsets_batch[max_orientation_idx, 0],
            best_phi,
            out=best_phi,
        )
        torch.where(
            update_mask,
            euler_angle_offsets_batch[max_orientation_idx, 1],
            best_theta,
            out=best_theta,
        )
        torch.where(
            update_mask,
            euler_angle_offsets_batch[max_orientation_idx, 2],
            best_psi,
            out=best_psi,
        )
        torch.where(
            update_mask,
            defocus_offsets[max_defocus_idx],
            best_defocus,
            out=best_defocus,
        )

    # Now find the maximum (x, y) of the MIP and return the best statistics at that pos
    max_idx = torch.argmax(mip)
    pox_y, pos_x = torch.unravel_index(max_idx, mip.shape)
    refined_mip = mip[pox_y, pos_x]
    refined_phi = best_phi[pox_y, pos_x]
    refined_theta = best_theta[pox_y, pos_x]
    refined_psi = best_psi[pox_y, pos_x]
    refined_defocus = best_defocus[pox_y, pos_x]

    return {
        "refined_mip": refined_mip,
        "refined_phi": refined_phi,
        "refined_theta": refined_theta,
        "refined_psi": refined_psi,
        "refined_defocus": refined_defocus,
        "refined_offset_y": pox_y,
        "refined_offset_x": pos_x,
    }


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
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
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
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
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=batch_rotation_matrices,
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
