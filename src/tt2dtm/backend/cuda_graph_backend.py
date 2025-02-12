import torch

from tt2dtm.backend.branchless_fourier_slice import (
    extract_central_slices_rfft_3d_branchless,
    setup_slice_coordinates,
)
from tt2dtm.backend.statistic_updates import (
    do_update_best_statistics,
    normalize_template_projection,
)

WARMUP_ITERATIONS = 32
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "reduce-overhead"
DEFAULT_STATISTIC_DTYPE = torch.float32

####################################
### Compile the helper functions ###
####################################

extract_central_slices_rfft_3d_branchless_compiled = torch.compile(
    extract_central_slices_rfft_3d_branchless,
    backend=COMPILE_BACKEND,
    mode=COMPILE_MODE,
)
do_update_best_statistics_compiled = torch.compile(
    do_update_best_statistics, backend=COMPILE_BACKEND, mode=COMPILE_MODE
)
normalize_template_projection_compiled = torch.compile(
    normalize_template_projection, backend=COMPILE_BACKEND, mode=COMPILE_MODE
)

################################################################
### Additional helper functions for steps in the 2DTM method ###
################################################################


def apply_projective_filters(
    outslice: torch.Tensor,
    projective_filters: torch.Tensor,
    fourier_slices: torch.Tensor,
    projections: torch.Tensor,
) -> None:
    """Applies projective filters to the extracted slices & transform to real-space."""
    fourier_slices = outslice[None, ...] * projective_filters[:, None, ...]

    projections = torch.fft.irfftn(fourier_slices, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))


def perform_cross_correlation(
    projections: torch.Tensor,
    projections_dft: torch.Tensor,
    image_dft: torch.Tensor,
    cross_correlation: torch.Tensor,
    h: int,
    w: int,
    H: int,
    W: int,
) -> None:
    """Performs the cross-correlation between template projections and image."""
    projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))
    projections_dft = image_dft[None, None, ...] * projections_dft.conj()

    cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))


####################################################
### Single iteration function for match_template ###
####################################################


def match_template_iteration(
    image_dft: torch.Tensor,
    volume_rfft: torch.Tensor,
    coordinates: torch.Tensor,
    euler_angles_batch: torch.Tensor,
    rotation_matrices_batch: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    max_intensity_projection: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    outslice: torch.Tensor,
    conjugate_mask: torch.Tensor,
    inside_mask: torch.Tensor,
    fourier_slices: torch.Tensor,
    projections: torch.Tensor,
    projections_dft: torch.Tensor,
    cross_correlation: torch.Tensor,
    h: int,
    w: int,
    H: int,
    W: int,
):
    """Does all operations for a single iteration of match_template for graph recording.

    Note that all tensors (even those just used as temporary intermediaries) are
    required since the cuda graph needs to record how operations are being applied
    on/between them. Statistics are updated in-place.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    volume_rfft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1). where l is the number of
        slices.
    coordinates : torch.Tensor
        Coordinates to rotate around for a given volume shape. Has shape
        (batch, h, w, 3).
    euler_angles_batch : torch.Tensor
        The current Euler angles for the batch. Has shape (batch, 3).
    rotation_matrices_batch : torch.Tensor
        The current rotation matrices for the batch. Has shape (batch, 3, 3).
    projective_filters : torch.Tensor
        The projective filters to apply to the extracted slices. Has shape
        (defocus_values.shape[0], h, w).
    defocus_values : torch.Tensor
        The defocus values corresponding to the projective filters. Has shape
        (defocus_values.shape[0],).
    max_intensity_projection : torch.Tensor
        Tensor for tracking teh maximum intensity projection. Has shape (H, W).
    best_phi : torch.Tensor
        Tensor for tracking the best phi Euler angle. Has shape (H, W).
    best_theta : torch.Tensor
        Tensor for tracking the best theta Euler angle. Has shape (H, W).
    best_psi : torch.Tensor
        Tensor for tracking the best psi Euler angle. Has shape (H, W).
    best_defocus : torch.Tensor
        Tensor for tracking the best defocus value. Has shape (H, W).
    correlation_sum : torch.Tensor
        Tensor for tracking the sum of the cross-correlation values. Has shape (H, W).
    correlation_squared_sum : torch.Tensor
        Tensor for tracking the sum of the squared cross-correlation values.
        Has shape (H, W).
    outslice : torch.Tensor
        Temporary tensor for storing the extracted slice. Has shape (batch, h, w).
    conjugate_mask : torch.Tensor
        Temporary tensor for storing the conjugate mask. Has shape (batch, h, w).
    inside_mask : torch.Tensor
        Temporary tensor for storing the inside mask. Has shape (batch * h * w).
    fourier_slices : torch.Tensor
        Temporary tensor for storing the Fourier slices (with filters applied). Has
        shape (batch, defocus_values.shape[0], h, w).
    projections : torch.Tensor
        Temporary tensor for storing the template projections. Has shape
        (batch, defocus_values.shape[0], h, w).
    projections_dft : torch.Tensor
        Temporary tensor for storing the DFT of the template projections. Has shape
        (batch, defocus_values.shape[0], H, W).
    cross_correlation : torch.Tensor
        Temporary tensor for storing the cross-correlation values. Has shape
        (batch, H, W).
    projections : torch.Tensor
        Temporary tensor for storing the template projections. Has shape
        (batch, defocus_values.shape[0], h, w).
    h : int
        Height of a projection, in real space.
    w : int
        Width of a projection, in real space.
    H : int
        Height of the image, in real space.
    W : int
        Width of the image, in real space.
    """
    extract_central_slices_rfft_3d_branchless_compiled(
        volume_rfft,
        rotation_matrices_batch,
        coordinates,
        conjugate_mask,
        inside_mask,
        outslice,
    )

    apply_projective_filters(
        outslice,
        projective_filters,
        fourier_slices,
        projections,
    )

    normalize_template_projection_compiled(projections, (h, w), (H, W))

    perform_cross_correlation(
        projections,
        projections_dft,
        image_dft,
        cross_correlation,
        h,
        w,
        H,
        W,
    )
    
    do_update_best_statistics_compiled(
        cross_correlation,
        euler_angles,
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


########################################################
### Helper functions for constructing the CUDA graph ###
########################################################


def warmup(
    image_dft: torch.Tensor,
    volume_rfft: torch.Tensor,
    coordinates: torch.Tensor,
    euler_angles_batch: torch.Tensor,
    rotation_matrices_batch: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    max_intensity_projection: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    outslice: torch.Tensor,
    conjugate_mask: torch.Tensor,
    inside_mask: torch.Tensor,
    fourier_slices: torch.Tensor,
    projections: torch.Tensor,
    projections_dft: torch.Tensor,
    cross_correlation: torch.Tensor,
    h: int,
    w: int,
    H: int,
    W: int,
    immutable_coordinates: torch.Tensor,
) -> None:
    """Runs a few iterations of match_template to "warm up" CUDA objects.

    TODO: Finish docstring.
    """
    batch = rotation_matrices_batch.shape[0]
    device = static_volume_rfft.device

    warmup_euler_angles = torch.randn((WARMUP_ITERATIONS, batch, 3), device=device)
    warmup_rotation_matrices = torch.randn(
        (WARMUP_ITERATIONS, batch, 3, 3), device=device
    )
    warmup_rotation_matrices = torch.linalg.qr(warmup_rotation_matrices)[0]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(WARMUP_ITERATIONS):
            rotation_matrices_batch.copy_(warmup_rotation_matrices[i])
            euler_angles_batch.copy_(warmup_euler_angles[i])
            coordinates.copy_(immutable_coordinates)
            outslice.fill_(0)

            match_template_iteration(
                image_dft,
                volume_rfft,
                coordinates,
                euler_angles_batch,
                rotation_matrices_batch,
                projective_filters,
                defocus_values,
                max_intensity_projection,
                best_phi,
                best_theta,
                best_psi,
                best_defocus,
                correlation_sum,
                correlation_squared_sum,
                outslice,
                conjugate_mask,
                inside_mask,
                fourier_slices,
                projections,
                projections_dft,
                cross_correlation,
                h,
                w,
                H,
                W,
            )

    torch.cuda.current_stream().wait_stream(s)


# fmt: off
def capture_graph(
    image_dft: torch.Tensor,
    volume_rfft: torch.Tensor,
    coordinates: torch.Tensor,
    euler_angles_batch: torch.Tensor,
    rotation_matrices_batch: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    max_intensity_projection: torch.Tensor,
    best_phi: torch.Tensor,
    best_theta: torch.Tensor,
    best_psi: torch.Tensor,
    best_defocus: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
    outslice: torch.Tensor,
    conjugate_mask: torch.Tensor,
    inside_mask: torch.Tensor,
    fourier_slices: torch.Tensor,
    projections: torch.Tensor,
    projections_dft: torch.Tensor,
    cross_correlation: torch.Tensor,
    h: int,
    w: int,
    H: int,
    W: int,
    immutable_coordinates: torch.Tensor,
) -> torch.cuda.CUDAGraph:
    """Captures the CUDA graph for match_template.

    TODO: Finish docstring.
    """
    example_euler_angles = torch.randn_like(euler_angles_batch)
    example_rotation_matrices = torch.randn_like(static_rotation_matrices)
    example_rotation_matrices = torch.linalg.qr(example_rotation_matrices)[0]

    euler_angles_batch.copy_(example_euler_angles)
    rotation_matrices_batch.copy_(example_rotation_matrices)
    coordinates.copy_(immutable_coordinates)
    outslice.fill_(0)
    

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        match_template_iteration(
            image_dft=immutable_image_dft,
            volume_rfft=immutable_volume_rfft,
            coordinates=static_coordinates,
            euler_angles_batch=static_euler_angles_batch,
            rotation_matrices_batch=static_rotation_matrices,
            projective_filters=immutable_projective_filters,
            defocus_values=immutable_defocus_values,
            max_intensity_projection=dynamic_max_intensity_projection,
            best_phi=dynamic_best_phi,
            best_theta=dynamic_best_theta,
            best_psi=dynamic_best_psi,
            best_defocus=dynamic_best_defocus,
            correlation_sum=dynamic_correlation_sum,
            correlation_squared_sum=dynamic_correlation_squared_sum,
            outslice=dynamic_outslice,
            conjugate_mask=static_conjugate_mask,
            inside_mask=static_inside_mask,
            fourier_slices=dynamic_fourier_slices,
            projections=dynamic_projections,
            projections_dft=dynamic_projections_dft,
            cross_correlation=cross_correlation,
            projections=dynamic_projections,
            h=H,
            w=W,
            H=H,
            W=W,
        )
        
    return g


def construct_graph_on_device(
    image_dft: torch.Tensor,
    volume_rfft: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    projection_batch_size: int,
    device: torch.device,
):
    """Creates a CUDA graph object for match_template on the device.

    Here, we use the naming convention 'static_*' for tensors that are inputs to the
    CUDA graph, 'dynamic_*' that are modified in-place *and* whose values are important
    for the next steps, and 'immutable_*' for tensors that are never modified (generally
    copied from at each iteration).
    
    TODO: Finish docstring.
    """
    H, W = image_dft.shape
    d, h, w = volume_rfft.shape
    W = (W - 1) * 2
    w = (w - 1) * 2
    batch = projection_batch_size

    image_dft = image_dft.to(device)
    volume_rfft = volume_rfft.to(device)
    projective_filters = projective_filters.to(device)
    defocus_values = defocus_values.to(device)

    ########################################################################
    ### Immutable tensors (do not change ever, copied from at each iter) ###
    ### into dynamic tensor that changes based on inputs.                ###
    ########################################################################

    immutable_coordinates = setup_slice_coordinates(
        volume_shape=volume_rfft.shape, batch_size=projection_batch_size
    )
    immutable_coordinates = immutable_coordinates.to(device)

    ############################################################################
    ### Static tensors that get operated on / accessed during each iteration ###
    ############################################################################

    coordinates = immutable_coordinates.clone()
    euler_angles_batch = torch.empty((batch, 3), device=device)
    rotation_matrices_batch = torch.empty((batch, 3, 3), device=device)
    conjugate_mask = torch.empty((batch, h, w), device=device, dtype=torch.bool)
    inside_mask = torch.empty((batch * h * w), device=device, dtype=torch.bool)
    outslice = torch.empty((batch, h, w), device=device, dtype=torch.complex64)

    fourier_slices = torch.empty((defocus_values.shape[0], batch, h, w), device=device, dtype=torch.complex64)
    projections = torch.empty((defocus_values.shape[0], batch, h, w), device=device, dtype=torch.float32)
    projections_dft = torch.empty((defocus_values.shape[0], batch, H, W // 2 + 1), device=device, dtype=torch.complex64)
    cross_correlation = torch.empty((batch, H, W), device=device, dtype=torch.float32)
    
    max_intensity_projection = torch.full(
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
    
    warmup(
        image_dft,
        volume_rfft,
        coordinates,
        euler_angles_batch,
        rotation_matrices_batch,
        projective_filters,
        defocus_values,
        max_intensity_projection,
        best_phi,
        best_theta,
        best_psi,
        best_defocus,
        correlation_sum,
        correlation_squared_sum,
        outslice,
        conjugate_mask,
        inside_mask,
        fourier_slices,
        projections,
        projections_dft,
        cross_correlation,
        h,
        w,
        H,
        W,
        immutable_coordinates,
    )
    
    g = capture_graph(
        static_volume_rfft,
        static_coordinates,
        static_rotation_matrices,
        static_conjugate_mask,
        static_inside_mask,
        dynamic_outslice,
        immutable_coordinates,
    )
