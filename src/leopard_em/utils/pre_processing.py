"""Helper functions for pre-processing data for 2DTM."""

import torch
from torch_fourier_filter.ctf import calculate_ctf_2d

from leopard_em.pydantic_models import WhiteningFilterConfig


def calculate_ctf_filter_stack(
    pixel_size: float,
    template_shape: tuple[int, int],
    defocus_u: float,  # in um, *NOT* Angstrom
    defocus_v: float,  # in um, *NOT* Angstrom
    defocus_offsets: list[float],  # in um, *NOT* Angstrom
    astigmatism_angle: float,
    amplitude_contrast_ratio: float = 0.07,
    spherical_aberration: float = 2.7,
    phase_shift: float = 0.0,
    voltage: float = 300.0,
    ctf_B_factor: float = 60.0,
) -> torch.Tensor:
    """Calculate stack (batch) of CTF filters to apply to projections during 2DTM.

    NOTE: While the defocus in the YAML and other Pydantic models is in Angstrom, this
    function expects defocus in um! Beware to convert accordingly.

    Parameters
    ----------
    pixel_size : float
        The pixel size of the images in Angstrom.
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space. Note that RFFT here cause
        resulting filter to *not* be this passed tuple, but (h, w // 2 + 1).
    defocus_u : float
        The defocus in the u direction, in um.
    defocus_v : float
        The defocus in the v direction, in um.
    defocus_offsets : list[float]
        The offsets to apply to the defocus, in um.
    astigmatism_angle : float
        The angle of defocus astigmatism, in degrees.
    amplitude_contrast_ratio : float, optional
        The amplitude contrast ratio, by default 0.07.
    spherical_aberration : float, optional
        The spherical aberration constant, in mm, by default 2.7.
    phase_shift : float, optional
        The phase shift constant, by default 0.0.
    voltage : float, optional
        The voltage of the microscope, in kV, by default 300.0.
    ctf_B_factor : float, optional
        The additional B factor for the CTF, by default 60.0.
    """
    ctf_filters = []

    for delta_df in defocus_offsets:
        defocus = (defocus_u + defocus_v) / 2 + delta_df
        astigmatism = abs(defocus_u - defocus_v) / 2

        ctf = calculate_ctf_2d(
            defocus=defocus,
            astigmatism=astigmatism,
            astigmatism_angle=astigmatism_angle,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast_ratio,
            b_factor=ctf_B_factor,
            phase_shift=phase_shift,
            pixel_size=pixel_size,
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
        )

        ctf_filters.append(ctf)

    return torch.stack(ctf_filters, dim=0).squeeze(1)


def do_image_preprocessing(
    image_rfft: torch.Tensor,
    wf_config: WhiteningFilterConfig,
) -> torch.Tensor:
    """Pre-processes the input image before running the algorithm.

    1. Zero central pixel (0, 0)
    2. Calculate a whitening filter
    3. Do element-wise multiplication with the whitening filter
    4. Zero central pixel again (superfluous, but following cisTEM)
    5. Normalize (x /= sqrt(sum(abs(x)**2)); pixelwise)

    Parameters
    ----------
    image_rfft : torch.Tensor
        The input image, RFFT'd and unshifted.
    wf_config : WhiteningFilterConfig
        The configuration for the whitening filter.

    Returns
    -------
    torch.Tensor
        The pre-processed image.

    """
    H, W = image_rfft.shape
    W = (W - 1) * 2  # Account for RFFT
    npix_real = H * W

    # Zero out the constant term
    image_rfft[0, 0] = 0 + 0j

    wf_image = wf_config.calculate_whitening_filter(
        ref_img_rfft=image_rfft,
        output_shape=image_rfft.shape,
    )
    image_rfft *= wf_image
    image_rfft[0, 0] = 0 + 0j  # superfluous, but following cisTEM

    # NOTE: Extra indexing happening with squared_sum so that Hermitian pairs are
    # counted, but we skip the first column of the RFFT which should not be duplicated.
    squared_image_rfft = torch.abs(image_rfft) ** 2
    squared_sum = squared_image_rfft.sum() + squared_image_rfft[:, 1:].sum()
    image_rfft /= torch.sqrt(squared_sum)

    # # real-space image will now have mean=0 and variance=1
    # image_rfft *= npix_real  # NOTE: This would set the variance to 1 exactly, but...

    # NOTE: We add on extra division by sqrt(num_pixels) so the cross-correlograms
    # are roughly normalized to have mean 0 and variance 1.
    # We do this here since Fourier transform is linear, and we don't have to multiply
    # the cross correlation at each iteration. This *will not* make the image
    # have variance 1.
    image_rfft *= npix_real**0.5

    return image_rfft
