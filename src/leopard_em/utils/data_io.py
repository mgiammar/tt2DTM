"""Utility functions dealing with basic data I/O operations."""

import os
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import torch


def read_mrc_to_numpy(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    """Reads an MRC file and returns the data as a numpy array.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        The MRC data as a numpy array, copied.
    """
    with mrcfile.open(mrc_path) as mrc:
        return mrc.data.copy()


def read_mrc_to_tensor(mrc_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads an MRC file and returns the data as a torch tensor.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC data as a tensor, copied.
    """
    return torch.tensor(read_mrc_to_numpy(mrc_path))


def write_mrc_from_numpy(
    data: np.ndarray,
    mrc_path: str | os.PathLike | Path,
    mrc_header: Optional[dict] = None,
    overwrite: bool = False,
) -> None:
    """Writes a numpy array to an MRC file.

    NOTE: Not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    # TODO: Figure out how to set info in the header
    if mrc_header is not None:
        raise NotImplementedError("Setting header info is not yet implemented.")

    with mrcfile.new(mrc_path, overwrite=overwrite) as mrc:
        mrc.set_data(data)


def write_mrc_from_tensor(
    data: torch.Tensor,
    mrc_path: str | os.PathLike | Path,
    mrc_header: Optional[dict] = None,
    overwrite: bool = False,
) -> None:
    """Writes a tensor array to an MRC file.

    NOTE: Not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    # TODO: Figure out how to set info in the header
    write_mrc_from_numpy(data.numpy(), mrc_path, mrc_header, overwrite)


def load_mrc_image(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an two-dimensional MRC image into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC image as a tensor.

    Raises
    ------
    ValueError
        If the MRC file is not two-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 2D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 2:
        raise ValueError(f"MRC file is not two-dimensional. Got shape: {tensor.shape}")

    return tensor


def load_mrc_volume(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an three-dimensional MRC volume into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC volume as a tensor.

    Raises
    ------
    ValueError
        If the MRC file is not three-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 3D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 3:
        raise ValueError(
            f"MRC file is not three-dimensional. Got shape: {tensor.shape}"
        )

    return tensor


def write_survival_histogram(
    hist_path: str | os.PathLike | Path,
    survival_histogram: torch.Tensor,
    expected_noise: float,
    histogram_data: torch.Tensor,
    expected_survival_hist: torch.Tensor,
    temp_float: float,
    HISTOGRAM_STEP: float,
    HISTOGRAM_NUM_POINTS: int,
) -> None:
    """Write survival histogram to file.

    Parameters
    ----------
    hist_path : str | os.PathLike | Path
        Path to the survival histogram file.
    survival_histogram : torch.Tensor
        Survival histogram.
    expected_noise : float
        Expected noise.
    histogram_data : torch.Tensor
        Histogram data.
    expected_survival_hist : torch.Tensor
        Expected survival histogram.
    temp_float : float
        Temporary float value.
    HISTOGRAM_STEP : float
        Histogram step size.
    HISTOGRAM_NUM_POINTS : int
        Number of histogram points.
    """
    with open(hist_path, "w") as f:
        f.write(f"Expected threshold is {expected_noise}\n")
        f.write("SNR, histogram, survival histogram, random survival histogram\n")
        for i in range(HISTOGRAM_NUM_POINTS):
            f.write(
                f"{temp_float + HISTOGRAM_STEP * i}, "
                f"{histogram_data[i]}, "
                f"{survival_histogram[i]}, "
                f"{expected_survival_hist[i]}\n"
            )
