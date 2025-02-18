"""Pure PyTorch implementation of whole orientation search backend."""

from multiprocessing import Manager, Process, set_start_method

import roma
import torch
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d

from tt2dtm.backend.process_results import (
    aggregate_distributed_results,
    scale_mip,
)
from tt2dtm.backend.utils import (
    do_iteration_statistics_updates,
    normalize_template_projection,
)

COMPILE_BACKEND = "inductor"
DEFAULT_STATISTIC_DTYPE = torch.float32

# Turn off gradient calculations by default
torch.set_grad_enabled(False)

# Set multiprocessing start method to spawn
set_start_method("spawn", force=True)

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
    orientation_batch_size: int = 1,
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
        Euler angles (in 'ZYZ' convention) to search over. Has shape
        (orientations, 3).
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    device : torch.device | list[torch.device]
        Device or devices to split computation across.
    orientation_batch_size : int, optional
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
        orientation_batch_size=orientation_batch_size,
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


def construct_multi_gpu_match_template_kwargs(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    orientation_batch_size: int,
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
    orientation_batch_size : int
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
            "orientation_batch_size": orientation_batch_size,
        }

        kwargs_per_device.append(kwargs)

    return kwargs_per_device


def _core_match_template_single_gpu(
    result_dict: dict,
    device_id: int,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    orientation_batch_size: int,
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
        Euler angles (in 'ZYZ' convention) to search over. Has shape
        (orientations // n_devices, 3). This has already been split (e.g.
        4 devices has shape (orientations // 4, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    orientation_batch_size : int
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

    mip = torch.full(
        size=(H, W),
        fill_value=-float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_phi = torch.full(
        size=(H, W),
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_theta = torch.full(
        size=(H, W),
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_psi = torch.full(
        size=(H, W),
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_defocus = torch.full(
        size=(H, W),
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

    num_batches = euler_angles.shape[0] // orientation_batch_size
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
            i * orientation_batch_size : (i + 1) * orientation_batch_size
        ]
        rot_matrix = roma.euler_to_rotmat(
            "ZYZ", euler_angles_batch, degrees=True, device=device
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

        # Inverse Fourier transform into real space and normalize
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection_compiled(
            projections, (h, w), (H, W)
        )

        # Padded forward Fourier transform for cross-correlation
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))

        # Cross correlation step by element-wise multiplication
        projections_dft = image_dft[None, None, ...] * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

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
            H,
            W,
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
