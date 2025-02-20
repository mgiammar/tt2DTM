"""Pydantic model for running the refine template program."""

from typing import Annotated, Any, ClassVar, Literal

import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from tt2dtm.pydantic_models.computational_config import ComputationalConfig
from tt2dtm.pydantic_models.correlation_filters import PreprocessingFilters
from tt2dtm.pydantic_models.defocus_search import DefocusSearchConfig
from tt2dtm.pydantic_models.match_template_manager import MatchTemplateManager
from tt2dtm.pydantic_models.optics_group import OpticsGroup
from tt2dtm.pydantic_models.orientation_search import RefineOrientationConfig
from tt2dtm.pydantic_models.types import BaseModel2DTM, ExcludedTensor
from tt2dtm.utils.data_io import load_mrc_volume


class ExtractParticleStackConfig(BaseModel2DTM):
    """Configuration parameters defining how to extract particles from micrographs.

    Attributes of this class are passed directly to the
    `tt2dtm.utils.get_cropped_image_regions` function when extracting particles from
    micrographs.

    Attributes
    ----------
    box_size: int
        The size of the box to extract around each particle. Must be greater than zero.
        Box size is the same in both the x and y dimensions.
    pos_reference: Literal["top_left", "center"]
        Reference point for the position of the particle. If "top_left", the position
        is the top-left corner of the box. If "center", the position is the center of
        the box.
    handle_bounds: Literal["pad", "error"]
        How to handle particles that are too close to the edge of the micrograph. If
        "pad", pad the particle with zeros. If "error", raise an error.
    padding_mode: Literal["constant", "reflect", "replicate", "circular"]
        Padding mode to use when padding particles. See the PyTorch documentation on
        the `torch.nn.functional.pad` function for more information.
    padding_value: float
        Value to use when padding particles. Only used if `padding_mode` is "constant".
    """

    box_size: Annotated[int, Field(gt=0)]
    pos_reference: Literal["top_left", "center"] = "center"
    handle_bounds: Literal["pad", "error"] = "pad"
    padding_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant"
    padding_value: float = 0.0


class RefineTemplateManager(BaseModel2DTM):
    """Model holding parameters necessary for running the refine template program.

    Attributes
    ----------
    TODO: Fill in attributes.

    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)

    # Serialized configuration attributes
    micrograph_paths: str | list[str]
    template_volume_path: str
    extract_particle_stack_config: ExtractParticleStackConfig
    optics_group: OpticsGroup
    defocus_search_config: DefocusSearchConfig
    refine_orientation: RefineOrientationConfig
    preprocessing_filters: PreprocessingFilters
    # refine_template_result: RefineTemplateResult
    computational_config: ComputationalConfig

    micrograph_index: list[int]
    optics_group_index: list[int]
    particle_stack_index: list[int]

    # Excluded tensors
    particle_stack: ExcludedTensor
    template_volume: ExcludedTensor

    pos_x: ExcludedTensor
    pos_y: ExcludedTensor
    mip: ExcludedTensor
    scaled_mip: ExcludedTensor
    psi: ExcludedTensor
    theta: ExcludedTensor
    phi: ExcludedTensor
    defocus: ExcludedTensor
    corr_avg: ExcludedTensor
    corr_var: ExcludedTensor

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

    def load_particle_stack(self) -> None:
        """Extracts particles from known positions in micrographs."""
        raise NotImplementedError("Method not implemented yet.")

    @field_validator("micrograph_paths")  # type: ignore
    def validate_micrograph_paths(cls, value: str | list[str]) -> list[str]:
        """Validate the micrograph paths."""
        if isinstance(value, str):
            return [value]
        return value

    def particles_to_dataframe(self) -> pd.DataFrame:
        """Export refined particles as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def results_to_dataframe(self) -> pd.DataFrame:
        """Export full results as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def particle_stack_to_dataframe(self) -> pd.DataFrame:
        """Export particle stack information as dataframe."""
        raise NotImplementedError("Method not implemented yet.")

    def save_particle_stack(self, save_dir: str) -> None:
        """Save the particle stack to disk."""
        raise NotImplementedError("Method not implemented yet.")
