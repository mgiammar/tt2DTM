"""Pydantic model for running the refine template program."""

from typing import Any, ClassVar

import pandas as pd
import torch
from pydantic import ConfigDict

from tt2dtm.backend.core_refine_template import core_refine_template
from tt2dtm.pydantic_models.computational_config import ComputationalConfig
from tt2dtm.pydantic_models.correlation_filters import PreprocessingFilters
from tt2dtm.pydantic_models.defocus_search import DefocusSearchConfig
from tt2dtm.pydantic_models.match_template_manager import MatchTemplateManager
from tt2dtm.pydantic_models.orientation_search import RefineOrientationConfig
from tt2dtm.pydantic_models.particle_stack import ParticleStack
from tt2dtm.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_volume


class RefineTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    template_volume_path : str
        Path to the template volume MRC file.
    particle_stack : ParticleStack
        Particle stack object containing particle data.
    defocus_refinement_config : DefocusSearchConfig
        Configuration for defocus refinement.
    orientation_refinement_config : RefineOrientationConfig
        Configuration for orientation refinement.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    computational_config : ComputationalConfig
        What computational resources to allocate for the program.
    template_volume : ExcludedTensor
        The template volume tensor (excluded from serialization).

    Methods
    -------
    TODO serialization/import methods
    __init__(self, skip_mrc_preloads: bool = False, **data: Any)
        Initialize the refine template manager.
    make_backend_core_function_kwargs(self) -> dict[str, Any]
        Create the kwargs for the backend refine_template core function.
    run_refine_template(self, particle_batch_size: int = 64) -> None
        Run the refine template program.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    template_volume_path: str  # In df per-particle, but ensure only one reference
    particle_stack: ParticleStack
    defocus_refinement_config: DefocusSearchConfig
    orientation_refinement_config: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    # refine_template_result: RefineTemplateResult
    computational_config: ComputationalConfig

    # Excluded tensors
    template_volume: ExcludedTensor

    @classmethod
    def from_results_csv(cls, results_csv_path: str) -> "RefineTemplateManager":
        """Take tabular data (from csv) and create a RefineTemplateManager object.

        Parameters
        ----------
        results_csv_path : str
            Path to the CSV file containing the per-particle data.

        Returns
        -------
        RefineTemplateManager
        """
        df = pd.read_csv(results_csv_path)

        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "RefineTemplateManager":
        """Tabular data (from DataFrame) and create a RefineTemplateManager object."""
        # # Ensure there is only one template volume path
        # template_volume_path = dataframe["template_volume_path"].unique()
        # if len(template_volume_path) != 1:
        #     raise ValueError("Multiple template volume paths found in dataframe.")
        # template_volume_path = template_volume_path[0]

        # # Tensor data
        # pos_x = dataframe["pos_x"].values
        # pos_y = dataframe["pos_y"].values
        # mip = dataframe["mip"].values
        # scaled_mip = dataframe["scaled_mip"].values
        # psi = dataframe["psi"].values
        # theta = dataframe["theta"].values
        # phi = dataframe["phi"].values
        # defocus = dataframe["defocus"].values
        # corr_avg = dataframe["corr_average"].values
        # corr_var = dataframe["corr_variance"].values

        # micrograph_paths = dataframe["micrograph_path"].values
        # micrograph_paths_unique = micrograph_paths  # sorted
        # template_volume_path = dataframe["template_volume_path"].values

        raise NotImplementedError("Method not implemented yet.")

    @classmethod
    def from_match_template_manager(
        cls, mt_manager: MatchTemplateManager
    ) -> "RefineTemplateManager":
        """Creates a RefineTemplateManager object from a MatchTemplateManager object."""
        raise NotImplementedError("Method not implemented yet.")

    def __init__(self, skip_mrc_preloads: bool = False, **data: Any):
        super().__init__(**data)

        # Load the data from the MRC files
        if not skip_mrc_preloads:
            self.template_volume = load_mrc_volume(self.template_volume_path)

    def make_backend_core_function_kwargs(self) -> dict[str, Any]:
        """Create the kwargs for the backend refine_template core function."""
        device = self.computational_config.gpu_devices()
        if len(device) > 1:
            raise ValueError("Only single-device execution is currently supported.")
        device = device[0]

        if self.template_volume is None:
            self.template_volume = load_mrc_volume(self.template_volume_path)
        if not isinstance(self.template_volume, torch.Tensor):
            template = torch.from_numpy(self.template_volume)
        else:
            template = self.template_volume

        template_shape = template.shape[-2:]

        particle_images = self.particle_stack.construct_image_stack(
            pos_reference="center",
            padding_value=0.0,
            handle_bounds="pad",
            padding_mode="constant",
        )
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Calculate and apply the filters for the particle image stack
        filter_stack = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters, output_shape=particle_images_dft.shape[-2:]
        )
        particle_images_dft *= filter_stack

        # Normalize each particle image to mean zero variance 1
        squared_image_dft = torch.abs(particle_images_dft) ** 2
        squared_sum = torch.sum(squared_image_dft, dim=(-2, -1), keepdim=True)
        particle_images_dft /= torch.sqrt(squared_sum)

        # Normalize by the effective number of pixels in the particle images
        # (sum of the bandpass filter). See comments in 'match_template_manager.py'.
        bp_config = self.preprocessing_filters.bandpass_filter
        bp_filter_image = bp_config.calculate_bandpass_filter(
            particle_images_dft.shape[-2:]
        )
        dimensionality = bp_filter_image.sum() + bp_filter_image[:, 1:-1].sum()
        particle_images_dft /= dimensionality**0.5

        # Calculate the filters applied to each template (besides CTF)
        projective_filters = self.particle_stack.construct_filter_stack(
            self.preprocessing_filters,
            output_shape=(template_shape[-2], template_shape[-1] // 2 + 1),
        )

        # Calculate the DFT of the template to take Fourier slices from
        # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
        # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
        # structure factors in the Fourier domain. This just requires an extra
        # IFFTshift after converting a slice back to real-space (handled already).
        template_dft = torch.fft.fftshift(template, dim=(-3, -2, -1))
        template_dft = torch.fft.rfftn(template_dft, dim=(-3, -2, -1))
        template_dft = torch.fft.fftshift(template_dft, dim=(-3, -2))  # skip rfft dim

        # The set of "best" euler angles from match template search
        euler_angles = torch.stack(
            (
                self.particle_stack.psi,
                self.particle_stack.theta,
                self.particle_stack.phi,
            ),
            dim=-1,
        )

        # The relative Euler angle offsets to search over
        euler_angle_offsets = self.orientation_refinement_config.euler_angles_offsets

        # The best defocus values for each particle (+ astigmatism)
        defocus_u = self.particle_stack.defocus_u + self.particle_stack.defocus
        defocus_v = self.particle_stack.defocus_v + self.particle_stack.defocus
        defocus_angle = self.particle_stack.defocus_astigmatism_angle

        # The relative defocus values to search over
        defocus_offsets = self.defocus_refinement_config.defocus_values

        # Keyword arguments for the CTF filter calculation call
        # NOTE: We currently enforce the parameters (other than the defocus values) are
        # all the same. This could be updated in the future...
        part_stk = self.particle_stack
        assert part_stk.pixel_size.unique().shape[0] == 1
        assert part_stk.voltage.unique().shape[0] == 1
        assert part_stk.spherical_aberration.unique().shape[0] == 1
        assert part_stk.amplitude_contrast_ratio.unique().shape[0] == 1
        assert part_stk.phase_shift.unique().shape[0] == 1
        assert part_stk.ctf_B_factor.unique().shape[0] == 1

        ctf_kwargs = {
            "voltage": part_stk.voltage[0].item(),
            "spherical_aberration": part_stk.spherical_aberration[0].item(),
            "amplitude_contrast": part_stk.amplitude_contrast_ratio[0].item(),
            "b_factor": part_stk.ctf_B_factor[0].item(),
            "phase_shift": part_stk.phase_shift[0].item(),
            "pixel_size": part_stk.pixel_size[0].item(),
            "image_shape": template_shape,
            "rfft": True,
            "fftshift": False,
        }

        return {
            "particle_stack_dft": particle_images_dft.to(device),
            "template_dft": template_dft,
            "euler_angles": euler_angles,
            "euler_angle_offsets": euler_angle_offsets,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "defocus_angle": defocus_angle,
            "defocus_offsets": defocus_offsets,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": projective_filters,
        }

    def run_refine_template(self, particle_batch_size: int = 64) -> None:
        """Run the refine template program."""
        backend_kwargs = self.make_backend_core_function_kwargs()

        core_refine_template(batch_size=particle_batch_size, **backend_kwargs)

    # def particles_to_dataframe(self) -> pd.DataFrame:
    #     """Export refined particles as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def results_to_dataframe(self) -> pd.DataFrame:
    #     """Export full results as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def particle_stack_to_dataframe(self) -> pd.DataFrame:
    #     """Export particle stack information as dataframe."""
    #     raise NotImplementedError("Method not implemented yet.")

    # def save_particle_stack(self, save_dir: str) -> None:
    #     """Save the particle stack to disk."""
    #     raise NotImplementedError("Method not implemented yet.")
