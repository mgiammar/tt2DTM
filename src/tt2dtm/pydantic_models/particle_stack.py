"""Particle stack Pydantic model for dealing with extracted particle data."""

from typing import Any, ClassVar, Literal

import pandas as pd
import torch
from pydantic import ConfigDict

from tt2dtm.utils.data_io import load_mrc_image
from tt2dtm.utils.particle_stack import get_cropped_image_regions

from .types import BaseModel2DTM, ExcludedTensor

# class Particle(BaseModel2DTM):
#     """Data class for a single particle.

#     TODO: Complete docstring
#     """

#     # Image of the particle
#     image: ExcludedTensor

#     # Location and orientation of particle
#     pos_x: int
#     pos_y: int
#     psi: float
#     theta: float
#     phi: float
#     defocus_u: float
#     defocus_v: float
#     defocus_astigmatism_angle: float

#     # Microscope (CTF) parameters
#     pixel_size: float
#     voltage: float
#     spherical_aberration: float
#     amplitude_contrast: float
#     phase_shift: float
#     ctf_B_factor: float

#     # Where the original image is stored
#     orig_micrograph_path: str

#     @property
#     def pos_x_angstrom(self) -> float:
#         return self.pos_x * self.pixel_size

#     @property
#     def pos_y_angstrom(self) -> float:
#         return self.pos_y * self.pixel_size

#     @property
#     def box_size(self) -> tuple[int, int]:
#         return self.image.shape


class ParticleStack(BaseModel2DTM):
    """Pydantic model for dealing with particle stack data.

    TODO: Complete docstring
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized fields
    df_path: str
    extracted_box_size: tuple[int, int]
    original_template_size: tuple[int, int]

    # Imported tabular data (not serialized)
    _df: pd.DataFrame

    # Cropped out view of the particles from images
    image_stack: ExcludedTensor

    # TODO: Implement methods for accessing the following maps as stacks
    # around the boxes
    # # Template matching related statistic maps (as stacks around the box)
    # _mip_map_stack: ExcludedTensor
    # _scaled_mip_map_stack: ExcludedTensor
    # _psi_map_stack: ExcludedTensor
    # _theta_map_stack: ExcludedTensor
    # _phi_map_stack: ExcludedTensor
    # _defocus_map_stack: ExcludedTensor
    # _corr_mean_map_stack: ExcludedTensor
    # _corr_variance_map_stack: ExcludedTensor

    def __init__(self, skip_df_load: bool = False, **data: dict[str, Any]):
        """Initialize the ParticleStack object.

        Parameters
        ----------
        skip_df_load : bool, optional
            Whether to skip loading the DataFrame, by default False and the dataframe
            is loaded automatically.
        data : dict[str, Any]
            The data to initialize the object with.
        """
        super().__init__(**data)

        if not skip_df_load:
            self.load_df()

    def load_df(self) -> None:
        """Load the DataFrame from the specified path."""
        self._df = pd.read_csv(self.df_path)

    def construct_image_stack(
        self,
        pos_reference: Literal["center", "top-left"] = "center",
        handle_bounds: Literal["pad", "error"] = "pad",
        padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """Construct stack of images from the DataFrame (updates image_stack in-place).

        Parameters
        ----------
        pos_reference : Literal["center", "top-left"], optional
            The reference point for the positions, by default "center". If "center", the
            boxes extracted will be image[y - box_size // 2 : y + box_size // 2, ...].
            If "top-left", the boxes will be image[y : y + box_size, ...].
        handle_bounds : Literal["pad", "clip", "error"], optional
            How to handle the bounds of the image, by default "pad". If "pad", the image
            will be padded with the padding value based on the padding mode. If "error",
            an error will be raised if any region exceeds the image bounds. NOTE:
            clipping is not supported since returned stack may have inhomogeneous sizes.
        padding_mode : Literal["constant", "reflect", "replicate"], optional
            The padding mode to use when padding the image, by default "constant".
            "constant" pads with the value `padding_value`, "reflect" pads with the
            reflection of the image at the edge, and "replicate" pads with the last
            pixel of the image. These match the modes available in
            `torch.nn.functional.pad`.
        padding_value : float, optional
            The value to use for padding when `padding_mode` is "constant", by default
            0.0.

        Returns
        -------
        torch.Tensor
            The stack of images, this is the internal 'image_stack' attribute.
        """
        # Create an empty tensor to store the image stack
        num_particles = len(self._df)
        image_stack = torch.zeros((num_particles, *self.extracted_box_size))

        # Find the indexes in the DataFrame that correspond to each unique image
        image_index_groups = self._df.groupby("reference_micrograph").groups

        # Loop over each unique image and extract the particles
        for img_path, indexes in image_index_groups.items():
            img = load_mrc_image(img_path)

            # with reference to center pixel
            pos_y = self._df.loc[indexes, "img_pos_y"]
            pos_x = self._df.loc[indexes, "img_pos_x"]
            pos_y = torch.tensor(pos_y)
            pos_x = torch.tensor(pos_x)

            cropped_images = get_cropped_image_regions(
                img,
                pos_y,
                pos_x,
                self.extracted_box_size,
                pos_reference=pos_reference,
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
            )
            image_stack[indexes] = cropped_images

        self.image_stack = image_stack

        return image_stack

    @property
    def pos_y(self) -> torch.Tensor:
        """Access the y-coordinate values in the underlying DataFrame."""
        return torch.tensor(self._df["pos_y"].values)

    @property
    def pos_x(self) -> torch.Tensor:
        """Access the x-coordinate values in the underlying DataFrame."""
        return torch.tensor(self._df["pos_x"].values)

    @property
    def mip(self) -> torch.Tensor:
        """Access the MIP values in the underlying DataFrame."""
        return torch.tensor(self._df["mip"].values)

    @property
    def scaled_mip(self) -> torch.Tensor:
        """Access the scaled MIP values in the underlying DataFrame."""
        return torch.tensor(self._df["scaled_mip"].values)

    @property
    def psi(self) -> torch.Tensor:
        """Access the psi values in the underlying DataFrame."""
        return torch.tensor(self._df["psi"].values)

    @property
    def theta(self) -> torch.Tensor:
        """Access the theta values in the underlying DataFrame."""
        return torch.tensor(self._df["theta"].values)

    @property
    def phi(self) -> torch.Tensor:
        """Access the phi values in the underlying DataFrame."""
        return torch.tensor(self._df["phi"].values)

    @property
    def defocus(self) -> torch.Tensor:
        """Access the defocus values in the underlying DataFrame."""
        return torch.tensor(self._df["defocus"].values)

    @property
    def corr_average(self) -> torch.Tensor:
        """Access the average correlation values in the underlying DataFrame."""
        return torch.tensor(self._df["corr_average"].values)

    @property
    def corr_variance(self) -> torch.Tensor:
        """Access the variance of correlation values in the underlying DataFrame."""
        return torch.tensor(self._df["corr_variance"].values)

    @property
    def corr_total(self) -> torch.Tensor:
        """Access the total correlation values in the underlying DataFrame."""
        return torch.tensor(self._df["corr_total"].values)

    @property
    def img_pos_y(self) -> torch.Tensor:
        """Access the y-coordinate values (for image) in the underlying DataFrame."""
        return torch.tensor(self._df["img_pos_y"].values)

    @property
    def img_pos_x(self) -> torch.Tensor:
        """Access the x-coordinate values (for image) in the underlying DataFrame."""
        return torch.tensor(self._df["img_pos_x"].values)

    @property
    def img_pos_y_angstrom(self) -> torch.Tensor:
        """Access the y-coordinate values (for image) in the underlying DataFrame."""
        return torch.tensor(self._df["img_pos_y_angstrom"].values)

    @property
    def img_pos_x_angstrom(self) -> torch.Tensor:
        """Access the x-coordinate values (for image) in the underlying DataFrame."""
        return torch.tensor(self._df["img_pos_x_angstrom"].values)

    @property
    def defocus_u(self) -> torch.Tensor:
        """Access the defocus_u values in the underlying DataFrame."""
        return torch.tensor(self._df["defocus_u"].values)

    @property
    def defocus_v(self) -> torch.Tensor:
        """Access the defocus_v values in the underlying DataFrame."""
        return torch.tensor(self._df["defocus_v"].values)

    @property
    def defocus_astigmatism_angle(self) -> torch.Tensor:
        """Access the defocus_astigmatism_angle values in the underlying DataFrame."""
        return torch.tensor(self._df["defocus_astigmatism_angle"].values)

    @property
    def pixel_size(self) -> torch.Tensor:
        """Access the pixel_size values in the underlying DataFrame."""
        return torch.tensor(self._df["pixel_size"].values)

    @property
    def voltage(self) -> torch.Tensor:
        """Access the voltage values in the underlying DataFrame."""
        return torch.tensor(self._df["voltage"].values)

    @property
    def spherical_aberration(self) -> torch.Tensor:
        """Access the spherical_aberration values in the underlying DataFrame."""
        return torch.tensor(self._df["spherical_aberration"].values)

    @property
    def amplitude_contrast_ratio(self) -> torch.Tensor:
        """Access the amplitude_contrast_ratio values in the underlying DataFrame."""
        return torch.tensor(self._df["amplitude_contrast_ratio"].values)

    @property
    def phase_shift(self) -> torch.Tensor:
        """Access the phase_shift values in the underlying DataFrame."""
        return torch.tensor(self._df["phase_shift"].values)

    @property
    def ctf_B_factor(self) -> torch.Tensor:
        """Access the ctf_B_factor values in the underlying DataFrame."""
        return torch.tensor(self._df["ctf_B_factor"].values)
