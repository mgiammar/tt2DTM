"""Serialization and validation of defocus search parameters for 2DTM."""

from typing import Annotated

import numpy as np
from pydantic import Field

from tt2dtm.pydantic_models.types import BaseModel2DTM


class DefocusSearchConfig(BaseModel2DTM):
    """Serialization and validation of defocus search parameters for 2DTM.

    Attributes
    ----------
    enabled : bool
        Whether to enable defocus search. Default is True.
    defocus_min : float
        Minimum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_max : float
        Maximum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_step : float
        Step size for defocus search in units of Angstroms.

    Properties
    ----------
    defocus_values : list[float]
        List of relative defocus values to search over based on held params.
    """

    enabled: bool = True
    defocus_min: float
    defocus_max: float
    defocus_step: Annotated[float, Field(..., gt=0.0)]

    @property
    def defocus_values(self) -> list[float]:
        """Relative defocus values to search over based on held params."""
        # Return a relative defocus of 0.0 if search is disabled.
        if not self.enabled:
            return [0.0]

        vals = np.arange(
            self.defocus_min,
            self.defocus_max + self.defocus_step,
            self.defocus_step,
        )
        vals = vals.tolist()

        return vals  # type: ignore
