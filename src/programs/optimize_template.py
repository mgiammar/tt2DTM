"""Script is used to optimize the template for a given pdb file."""

from typing import Optional, Union

import numpy as np
import torch
from ttsim3d.models import Simulator, SimulatorConfig

from leopard_em.pydantic_models import RefineTemplateManager

YAML_NO_SEARCH_CONFIG_PATH = "refine_config_no-search.yaml"
YAML_SEARCH_CONFIG_PATH = "refine_config_search.yaml"
PDB_PATH = "parsed_6Q8Y_whole_LSU_match3.pdb"
ORIENTATION_BATCH_SIZE = 16
GPU_IDS = None


def run_ttsim3d(
    px_value: float, gpu_ids: Optional[Union[int, list[int]]] = None
) -> torch.Tensor:
    """
    Run ttsim3d simulation with given pixel size.

    Parameters
    ----------
    px_value : float
        Pixel size value in Angstroms
    gpu_ids : Optional[Union[int, list[int]]], optional
        GPU device IDs to use for simulation, by default None

    Returns
    -------
    torch.Tensor
        Simulated template volume
    """
    sim_conf = SimulatorConfig(
        voltage=300.0,
        apply_dose_weighting=True,
        dose_start=0.0,
        dose_end=50.0,
        dose_filter_modify_signal="rel_diff",
        upsampling=-1,
        mtf_reference="/home/jdickerson/git/teamtomo/ttsim3d/src/data/mtf_falcon4EC_300kV.star",
    )

    sim = Simulator(
        pdb_filepath=PDB_PATH,
        pixel_spacing=px_value,
        volume_shape=(512, 512, 512),
        b_factor_scaling=0.5,
        additional_b_factor=0,
        simulator_config=sim_conf,
    )
    return sim.run(gpu_ids=gpu_ids)


def evaluate_peaks(rtm: RefineTemplateManager, backend_kwargs: dict) -> float:
    """
    Evaluate peaks using either match_template or refine_template.

    Parameters
    ----------
    rtm : RefineTemplateManager
        Manager object containing template matching parameters and methods
    backend_kwargs : dict
        Additional keyword arguments for the backend processing

    Returns
    -------
    float
        Mean SNR value from peak detection
    """
    result = rtm.get_refine_result(
        backend_kwargs, orientation_batch_size=ORIENTATION_BATCH_SIZE
    )
    df_refined = rtm.particle_stack._df.copy()
    refined_mip = result["refined_cross_correlation"]
    refined_scaled_mip = refined_mip - df_refined["correlation_mean"]
    refined_scaled_mip = refined_scaled_mip / np.sqrt(
        df_refined["correlation_variance"]
    )
    mean_snr = float(refined_scaled_mip.mean())
    print(f"max snr: {refined_scaled_mip.max()}, min snr: {refined_scaled_mip.min()}")
    return mean_snr


def evaluate_template_px(px_value: float, rtm: RefineTemplateManager) -> float:
    """
    Evaluate a single template pixel size and return the mean SNR.

    Parameters
    ----------
    px_value : float
        Template pixel size to evaluate
    rtm : RefineTemplateManager
        Manager object containing template matching parameters and methods

    Returns
    -------
    float
        Mean SNR value for the given pixel size
    """
    template_volume = run_ttsim3d(px_value=px_value, gpu_ids=GPU_IDS)
    rtm.template_volume = template_volume
    backend_kwargs = rtm.make_backend_core_function_kwargs()
    return evaluate_peaks(rtm, backend_kwargs)


def optimize_pixel_size_grid(
    rtm: RefineTemplateManager,
    initial_px: float,
    do_fine_search: bool = True,
    coarse_range: float = 0.005,
    coarse_step: float = 0.001,
    fine_range: float = 0.005,
    fine_step: float = 0.001,
) -> float:
    """
    Two-stage template pixel size optimization using grid search.

    Parameters
    ----------
    rtm : RefineTemplateManager
        Manager object containing template matching parameters and methods
    initial_px : float
        Initial pixel size guess
    do_fine_search : bool, optional
        Whether to perform a fine search, by default True
    coarse_range : float, optional
        Range for coarse search, by default 0.05
    coarse_step : float, optional
        Step size for coarse search, by default 0.01
    fine_range : float, optional
        Range for fine search, by default 0.005
    fine_step : float, optional
        Step size for fine search, by default 0.001

    Returns
    -------
    float
        Optimal pixel size found
    """
    coarse_px_values = torch.arange(
        initial_px - coarse_range, initial_px + coarse_range + 1e-10, coarse_step
    )

    best_snr = float("-inf")
    best_px = initial_px

    print("Starting coarse search...")
    for px in coarse_px_values:
        snr = evaluate_template_px(px.item(), rtm)
        print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")
        if snr > best_snr:
            best_snr = snr
            best_px = px.item()

    if do_fine_search:
        fine_px_values = torch.arange(
            best_px - fine_range, best_px + fine_range + 1e-10, fine_step
        )

        print("\nStarting fine search...")
        for px in fine_px_values:
            snr = evaluate_template_px(px.item(), rtm)
            print(f"Pixel size: {px:.3f}, SNR: {snr:.3f}")
            if snr > best_snr:
                best_snr = snr
                best_px = px.item()

    print(f"\nOptimal pixel size: {best_px:.3f} Å with SNR: {best_snr:.3f}")
    return best_px


def optimize_b_grid(rtm: RefineTemplateManager) -> float:
    """
    Optimize the CTF B-factor using grid search.

    Parameters
    ----------
    rtm : RefineTemplateManager
        Manager object containing template matching parameters and methods

    Returns
    -------
    float
        Optimal B-factor found
    """
    ctf_b = torch.arange(0.0, 401.0, 10.0)

    best_snr = float("-inf")
    best_b = 0.0
    for b in ctf_b:
        rtm.particle_stack._df["ctf_B_factor"] = b.item()
        backend_kwargs = rtm.make_backend_core_function_kwargs()
        snr = evaluate_peaks(rtm, backend_kwargs)
        if snr > best_snr:
            best_snr = snr
            best_b = b
        print(f"B factor: {b:.3f}, SNR: {snr:.3f}")

    return best_b


def main_grid() -> None:
    """Main function for grid search optimization."""
    rtm = RefineTemplateManager.from_yaml(YAML_NO_SEARCH_CONFIG_PATH)
    # Optimize template pixel size
    part_stk = rtm.particle_stack
    initial_template_px = part_stk["refined_pixel_size"].mean().item()
    print(f"Initial template px: {initial_template_px:.3f} Å")
    optimal_template_px = optimize_pixel_size_grid(
        rtm,
        initial_template_px,
        do_fine_search=False,
        coarse_range=0.001,
        coarse_step=0.001,
        fine_range=0.005,
        fine_step=0.001,
    )

    # Generate model with optimal px
    template_volume = run_ttsim3d(px_value=optimal_template_px, gpu_ids=GPU_IDS)
    # run refine tm on mtm results with diff b's
    rtm = RefineTemplateManager.from_yaml(YAML_NO_SEARCH_CONFIG_PATH)
    rtm.template_volume = template_volume
    # This time we'll do the angle and defoc search
    best_b = optimize_b_grid(rtm)

    # print template px and ctf b recommnendations
    print(f"Optimal template px: {optimal_template_px:.3f} Å")
    print(f"Optimal ctf b: {best_b:.3f}")


def main_gradient() -> None:
    """Main function for gradient search optimization."""
    pass


if __name__ == "__main__":
    # Choose which optimization method to use
    # main_gradient()  # or main_grid()
    main_grid()
