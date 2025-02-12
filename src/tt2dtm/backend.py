"""Pure PyTorch implementation of whole orientation search backend."""

from multiprocessing import Manager, Process, set_start_method

import numpy as np
import roma
import torch
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d

COMPILE_BACKEND = "inductor"
DEFAULT_STATISTIC_DTYPE = torch.float16
DO_VALID_CROP = True

# Turn off gradient calculations by default
torch.set_grad_enabled(False)

# Set multiprocessing start method to spawn
set_start_method("spawn", force=True)


def construct_multi_gpu_match_template_kwargs(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    projection_batch_size: int,
    devices: list[torch.device],
) -> list[dict[str, torch.Tensor | int]]:
    """Split orientations between requested devices.

    See the `core_match_template` function for further descriptions of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        dft of image
    template_dft : torch.Tensor
        dft of template
    euler_angles : torch.Tensor
        euler angles to search
    projective_filters : torch.Tensor
        filters to apply to each projection
    defocus_values : torch.Tensor
        corresponding defocus values for each filter
    projection_batch_size : int
        number of projections to calculate at once
    devices : list[torch.device]
        list of devices to split the orientations across

    Returns
    -------
    list[dict[str, torch.Tensor | int]]
        List of dictionaries containing the kwargs to call the single-GPU
        function. Each index in the list corresponds to a different device,
        and all tensors in the dictionary have been allocated to that device.
    """
    num_devices = len(devices)
    kwargs_per_device = []

    # Split the euler angles across devices
    euler_angles_split = euler_angles.chunk(num_devices)

    for device, euler_angles_device in zip(devices, euler_angles_split):
        # Allocate all tensors to the device
        image_dft_device = image_dft.to(device)
        template_dft_device = template_dft.to(device)
        euler_angles_device = euler_angles_device.to(device)
        projective_filters_device = projective_filters.to(device)
        defocus_values_device = defocus_values.to(device)

        # Construct the kwargs dictionary
        kwargs = {
            "image_dft": image_dft_device,
            "template_dft": template_dft_device,
            "euler_angles": euler_angles_device,
            "projective_filters": projective_filters_device,
            "defocus_values": defocus_values_device,
            "projection_batch_size": projection_batch_size,
        }

        kwargs_per_device.append(kwargs)

    return kwargs_per_device


######################################################
### Helper functions called at the end of the loop ###
######################################################


def aggregate_distributed_results(
    results: list[dict[str, torch.Tensor | np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Combine the 2DTM results from multiple devices.

    NOTE: This assumes that all tensors have been passed back to the CPU and are in
    the form of numpy arrays.

    Parameters
    ----------
    results : list[dict[str, np.ndarray]]
        List of dictionaries containing the results from each device. Each dictionary
        contains the following keys:
            - "mip": Maximum intensity projection of the cross-correlation values.
            - "best_phi": Best phi angle for each pixel.
            - "best_theta": Best theta angle for each pixel.
            - "best_psi": Best psi angle for each pixel.
            - "best_defocus": Best defocus value for each pixel.
            - "correlation_sum": Sum of cross-correlation values for each pixel.
            - "correlation_squared_sum": Sum of squared cross-correlation values for
              each pixel.
            - "total_projections": Total number of projections calculated.
    """
    # Ensure all the tensors are passed back to CPU as numpy arrays
    # Not sure why cannot sync across devices, but this is a workaround
    results = [
        {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in result.items()
        }
        for result in results
    ]

    mips = np.stack([result["mip"] for result in results], axis=0)
    best_phi = np.stack([result["best_phi"] for result in results], axis=0)
    best_theta = np.stack([result["best_theta"] for result in results], axis=0)
    best_psi = np.stack([result["best_psi"] for result in results], axis=0)
    best_defocus = np.stack([result["best_defocus"] for result in results], axis=0)

    mip_max = mips.max(axis=0)
    mip_argmax = mips.argmax(axis=0)

    best_phi = np.take_along_axis(best_phi, mip_argmax[None, ...], axis=0)
    best_theta = np.take_along_axis(best_theta, mip_argmax[None, ...], axis=0)
    best_psi = np.take_along_axis(best_psi, mip_argmax[None, ...], axis=0)
    best_defocus = np.take_along_axis(best_defocus, mip_argmax[None, ...], axis=0)

    # Sum the sums and squared sums of the cross-correlation values
    correlation_sum = np.stack(
        [result["correlation_sum"] for result in results], axis=0
    ).sum(axis=0)
    correlation_squared_sum = np.stack(
        [result["correlation_squared_sum"] for result in results], axis=0
    ).sum(axis=0)

    # NOTE: Currently only tracking total number of projections for statistics,
    # but could be future case where number of projections calculated on each
    # device is necessary for some statistical computation.
    total_projections = sum(result["total_projections"] for result in results)

    # Cast back to torch tensors on the CPU
    mip_max = torch.from_numpy(mip_max)
    best_phi = torch.from_numpy(best_phi)
    best_theta = torch.from_numpy(best_theta)
    best_psi = torch.from_numpy(best_psi)
    best_defocus = torch.from_numpy(best_defocus)
    correlation_sum = torch.from_numpy(correlation_sum)
    correlation_squared_sum = torch.from_numpy(correlation_squared_sum)

    return {
        "mip": mip_max,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
        "total_projections": total_projections,
    }


def scale_mip(
    mip: torch.Tensor,
    mip_scaled: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    total_correlation_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the MIP to Z-score map by the mean and variance of the correlation values.

    Z-score is accounting for the variation in image intensity and spurious correlations
    by subtracting the mean and dividing by the standard deviation pixel-wise. Since
    cross-correlation values are roughly normally distributed for pure noise, Z-score
    effectively becomes a measure of how unexpected (highly correlated to the reference
    template) a region is in the image. Note that we are looking at maxima of millions
    of Gaussian distributions, so Z-score has to be compared with a generalized extreme
    value distribution (GEV) to determine significance (done elsewhere).

    Parameters
    ----------
    mip : torch.Tensor
        MIP of the correlation values.
    mip_scaled : torch.Tensor
        Scaled MIP of the correlation values.
    correlation_sum : torch.Tensor
        Sum of the correlation values.
    correlation_squared_sum : torch.Tensor
        Sum of the squared correlation values.
    total_correlation_positions : int
        Total number cross-correlograms calculated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the MIP and scaled MIP
    """
    num_pixels = torch.tensor(mip.shape[0] * mip.shape[1])

    # Convert sum and squared sum to mean and variance in-place
    correlation_sum = correlation_sum / total_correlation_positions
    correlation_squared_sum = correlation_squared_sum / total_correlation_positions
    correlation_squared_sum -= correlation_sum**2
    correlation_squared_sum = torch.sqrt(torch.clamp(correlation_squared_sum, min=0))

    # Calculate normalized MIP
    mip_scaled = mip - correlation_sum
    torch.where(
        correlation_squared_sum != 0,  # preventing zero division error, albeit unlikely
        mip_scaled / correlation_squared_sum,
        torch.zeros_like(mip_scaled),
        out=mip_scaled,
    )

    mip = mip * (num_pixels**0.5)

    return mip, mip_scaled


###########################################################################
### Helper functions called during the loop (passed into torch.compile) ###
###########################################################################


def normalize_template_projection(
    projections: torch.Tensor,  # shape (batch, h, w)
    small_shape: tuple[int, int],  # (h, w)
    large_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    """Subtract mean of edge values and set variance to 1 (in large shape).

    This function uses the fact that variance of a sequence, Var(X), is scaled by the
    relative size of the small (unpadded) and large (padded with zeros) space. Some
    negligible error is introduced into the variance (~1e-4) due to this routine.

    Parameters
    ----------
    projections : torch.Tensor
        Real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        Shape of the template.
    large_shape : tuple[int, int]
        Shape of the image (in large space).

    Returns
    -------
    torch.Tensor
        Edge-mean subtracted projections, still in small space, but normalized
        so variance of zero-padded projection is 1.
    """
    # Constants related to scaling the variance
    npix_padded = large_shape[0] * large_shape[1] - small_shape[0] * small_shape[1]
    relative_size = small_shape[0] * small_shape[1] / (large_shape[0] * large_shape[1])

    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., W)
    bottom_edge = projections[..., -1, :]  # shape: (..., W)
    left_edge = projections[..., 1:-1, 0]  # shape: (..., H-2)
    right_edge = projections[..., 1:-1, -1]  # shape: (..., H-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )

    # Subtract the edge pixel mean and calculate variance of small, unpadded projection
    edge_mean = edge_pixels.mean(dim=-1)
    projections -= edge_mean[..., None, None]

    # # Calculate variance like cisTEM (does not match desired results...)
    # variance = (projections**2).sum(dim=(-1, -2), keepdim=True) * relative_size - (
    #     projections.mean(dim=(-1, -2), keepdim=True) * relative_size
    # ) ** 2

    # Fast calculation of mean/var using Torch + appropriate scaling.
    # Scale the variance such that the larger padded space has variance of 1.
    variance, mean = torch.var_mean(projections, dim=(-1, -2), keepdim=True)
    mean += relative_size
    variance *= relative_size
    variance += (1 / npix_padded) * mean**2

    return projections / torch.sqrt(variance)


def do_iteration_statistics_updates(
    cross_correlation: torch.Tensor,
    euler_angles: torch.Tensor,
    defocus_values: torch.Tensor,
    mip: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    H: int,
    W: int,
) -> None:
    """Helper function for updating maxima and tracked statistics.

    NOTE: The batch dimensions are effectively unraveled since taking the
    maximum over a single batch dimensions is much faster than
    multi-dimensional maxima.

    NOTE: Updating the maxima was found to be fastest and least memory
    impactful when using torch.where directly. Other methods tested were
    boolean masking and torch.where with tuples of tensor indexes.

    Parameters
    ----------
    cross_correlation : torch.Tensor
        Cross-correlation values for the current iteration. Has either shape
        (batch, H, W) or (defocus, orientations, H, W).
    euler_angles : torch.Tensor
        Euler angles for the current iteration. Has shape (orientations, 3).
    defocus_values : torch.Tensor
        Defocus values for the current iteration. Has shape (defocus,).
    mip : torch.Tensor
        Maximum intensity projection of the cross-correlation values.
    best_phi : torch.Tensor
        Best phi angle for each pixel.
    best_theta : torch.Tensor
        Best theta angle for each pixel.
    best_psi : torch.Tensor
        Best psi angle for each pixel.
    best_defocus : torch.Tensor
        Best defocus value for each pixel.
    correlation_sum : torch.Tensor
        Sum of cross-correlation values for each pixel.
    correlation_squared_sum : torch.Tensor
        Sum of squared cross-correlation values for each pixel.
    H : int
        Height of the cross-correlation values.
    W : int
        Width of the cross-correlation values.
    """
    max_values, max_indices = torch.max(cross_correlation.view(-1, H, W), dim=0)
    max_defocus_idx = max_indices // euler_angles.shape[0]
    max_orientation_idx = max_indices % euler_angles.shape[0]

    # using torch.where directly
    update_mask = max_values > mip

    torch.where(update_mask, max_values, mip, out=mip)
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 0], best_phi, out=best_phi
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 1], best_theta, out=best_theta
    )
    torch.where(
        update_mask, euler_angles[max_orientation_idx, 2], best_psi, out=best_psi
    )
    torch.where(
        update_mask, defocus_values[max_defocus_idx], best_defocus, out=best_defocus
    )

    correlation_sum += cross_correlation.view(-1, H, W).sum(dim=0)
    correlation_squared_sum += (cross_correlation.view(-1, H, W) ** 2).sum(dim=0)


#################################
### Compiled helper functions ###
#################################

normalize_template_projection_compiled = torch.compile(
    normalize_template_projection, backend=COMPILE_BACKEND
)
do_iteration_statistics_updates_compiled = torch.compile(
    do_iteration_statistics_updates, backend=COMPILE_BACKEND
)


###########################################################
###      Main function for whole orientation search     ###
### (inputs generalize beyond those in pydantic models) ###
###########################################################


def core_match_template(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,  # already fftshifted
    ctf_filters: torch.Tensor,
    whitening_filter_template: torch.Tensor,
    defocus_values: torch.Tensor,
    euler_angles: torch.Tensor,
    device: torch.device | list[torch.device],
    projection_batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Core function for performing the whole-orientation search.

    With the RFFT, the last dimension (fastest dimension) is half the width
    of the input, hence the shape of W // 2 + 1 instead of W for some of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    ctf_filters : torch.Tensor
        Stack of CTF filters at different defocus values to use in the search.
        Has shape (defocus_batch, h, w // 2 + 1).
    whitening_filter_template : torch.Tensor
        Whitening filter for the template volume. Has shape (h, w // 2 + 1).
        Gets multiplied with the ctf filters to create a filter stack.
    euler_angles : torch.Tensor
        Euler angles (in 'zyz' convention) to search over. Has shape
        (orientations, 3).
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    device : torch.device | list[torch.device]
        Device or devices to split computation across.
    projection_batch_size : int, optional
        Number of projections to calculate at once, on each device

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following
            - "mip": Maximum intensity projection of the cross-correlation values across
              orientation and defocus search space.
            - "scaled_mip": Z-score scaled MIP of the cross-correlation values.
            - "best_phi": Best phi angle for each pixel.
            - "best_theta": Best theta angle for each pixel.
            - "best_psi": Best psi angle for each pixel.
            - "best_defocus": Best defocus value for each pixel.
            - "correlation_sum": Sum of cross-correlation values for each pixel.
            - "correlation_squared_sum": Sum of squared cross-correlation values for
              each pixel.
            - "total_projections": Total number of projections calculated.
            - "total_orientations": Total number of orientations searched.
            - "total_defocus": Total number of defocus values searched.
    """
    ##############################################################
    ### Pre-multiply the whitening filter with the CTF filters ###
    ##############################################################

    projective_filters = ctf_filters * whitening_filter_template[None, ...]

    #########################################
    ### Split orientations across devices ###
    #########################################

    if isinstance(device, torch.device):
        device = [device]

    kwargs_per_device = construct_multi_gpu_match_template_kwargs(
        image_dft=image_dft,
        template_dft=template_dft,
        euler_angles=euler_angles,
        projective_filters=projective_filters,
        defocus_values=defocus_values,
        projection_batch_size=projection_batch_size,
        devices=device,
    )

    ##################################################
    ### Initialize and start multiprocessing queue ###
    ##################################################
    manager = Manager()
    result_dict = manager.dict()

    # lists to track processes
    processes = []

    # Start processes
    for i, kwargs in enumerate(kwargs_per_device):
        p = Process(
            target=_core_match_template_single_gpu,
            args=(result_dict, i),
            kwargs=kwargs,
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Get the aggregated results
    partial_results = [result_dict[i] for i in range(len(processes))]
    aggregated_results = aggregate_distributed_results(partial_results)
    mip = aggregated_results["mip"]
    best_phi = aggregated_results["best_phi"]
    best_theta = aggregated_results["best_theta"]
    best_psi = aggregated_results["best_psi"]
    best_defocus = aggregated_results["best_defocus"]
    correlation_sum = aggregated_results["correlation_sum"]
    correlation_squared_sum = aggregated_results["correlation_squared_sum"]
    total_projections = aggregated_results["total_projections"]

    mip_scaled = torch.empty_like(mip)
    mip, mip_scaled = scale_mip(
        mip=mip,
        mip_scaled=mip_scaled,
        correlation_sum=correlation_sum,
        correlation_squared_sum=correlation_squared_sum,
        total_correlation_positions=total_projections,
    )

    return {
        "mip": mip,
        "scaled_mip": mip_scaled,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "correlation_sum": correlation_sum,
        "correlation_squared_sum": correlation_squared_sum,
        "total_projections": total_projections,
        "total_orientations": euler_angles.shape[0],
        "total_defocus": defocus_values.shape[0],
    }


def _core_match_template_single_gpu(
    result_dict: dict,
    device_id: int,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    projection_batch_size: int,
) -> None:
    """Single-GPU call for template matching.

    NOTE: All tensors *must* be allocated on the same device. By calling the
    user-facing `core_match_template` function this is handled automatically.

    NOTE: The result_dict is a shared dictionary between processes and updated in-place
    with this processes's results under the 'device_id' key.

    Parameters
    ----------
    result_dict : dict
        Dictionary to store the results in.
    device_id : int
        ID of the device which computation is running on. Results will be stored
        in the dictionary with this key.
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    euler_angles : torch.Tensor
        Euler angles (in 'zyz' convention) to search over. Has shape
        (orientations // n_devices, 3). This has already been split (e.g.
        4 devices has shape (orientations // 4, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    projection_batch_size : int
        The number of projections to calculate the correlation for at once.

    Returns
    -------
    None
    """
    device = image_dft.device
    H, W = image_dft.shape
    h, w = template_dft.shape[-2:]
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    ################################################
    ### Initialize the tracked output statistics ###
    ################################################

    if DO_VALID_CROP:
        statistic_shape = (H - h + 1, W - w + 1)
    else:
        statistic_shape = (H, W)
    mip = torch.full(
        size=statistic_shape,
        fill_value=-float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_phi = torch.full(
        size=statistic_shape,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_theta = torch.full(
        size=statistic_shape,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_psi = torch.full(
        size=statistic_shape,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_defocus = torch.full(
        size=statistic_shape,
        fill_value=float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    correlation_sum = torch.zeros(
        size=(H, W), dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )
    correlation_squared_sum = torch.zeros(
        size=(H, W), dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )

    ########################################################
    ### Setup iterator object with tqdm for progress bar ###
    ########################################################

    num_batches = euler_angles.shape[0] // projection_batch_size
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"Progress on device: {device.index}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
        position=device.index,
    )

    total_projections = euler_angles.shape[0] * defocus_values.shape[0]

    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        euler_angles_batch = euler_angles[
            i * projection_batch_size : (i + 1) * projection_batch_size
        ]
        rot_matrix = roma.euler_to_rotmat(
            "zyz", euler_angles_batch, degrees=True, device=device
        )

        # Extract central slice(s) from the template volume
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,  # NOTE: requires cubic template
            rotation_matrices=rot_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast

        # Apply the projective filters on a new batch dimension
        fourier_slice = fourier_slice[None, ...] * projective_filters[:, None, ...]

        # NOTE: This is reshaping into a single batch dimension (not used)
        # fourier_slice = fourier_slice.reshape(
        #     -1, fourier_slice.shape[-2], fourier_slice.shape[-1]
        # )

        # Inverse Fourier transform into real space and normalize
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection_compiled(
            projections, (h, w), (H, W)
        )

        # Padded forward Fourier transform for cross-correlation
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))

        ### Cross correlation step by element-wise multiplication ###
        projections_dft = image_dft[None, None, ...] * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))
        cross_correlation = cross_correlation.to(DEFAULT_STATISTIC_DTYPE)

        # Crop the cross-correlation to the valid region
        if DO_VALID_CROP:
            cross_correlation = cross_correlation[
                : statistic_shape[0], : statistic_shape[1]
            ]

        # Update the tracked statistics through compiled function
        do_iteration_statistics_updates_compiled(
            cross_correlation,
            euler_angles_batch,
            defocus_values,
            mip,
            best_phi,
            best_theta,
            best_psi,
            best_defocus,
            correlation_sum,
            correlation_squared_sum,
            statistic_shape[0],
            statistic_shape[1],
        )

    # NOTE: Need to send all tensors back to the CPU as numpy arrays for the shared
    # process dictionary. This is a workaround for now
    result = {
        "mip": mip.cpu().numpy(),
        "best_phi": best_phi.cpu().numpy(),
        "best_theta": best_theta.cpu().numpy(),
        "best_psi": best_psi.cpu().numpy(),
        "best_defocus": best_defocus.cpu().numpy(),
        "correlation_sum": correlation_sum.cpu().numpy(),
        "correlation_squared_sum": correlation_squared_sum.cpu().numpy(),
        "total_projections": total_projections,
    }

    # Place the results in the shared multi-process manager dictionary so accessible
    # by the main process.
    result_dict[device_id] = result

    return None
