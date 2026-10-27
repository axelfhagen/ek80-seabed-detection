"""Statistical validation and uncertainty quantification for seabed segmentation."""

from validation.uncertainty import (
    mc_dropout_predict,
    prediction_entropy,
    bottom_uncertainty_band,
)
from validation.metrics import (
    threshold_sensitivity,
    calibration_curve,
)
from validation.ensemble import (
    load_ensemble,
    ensemble_predict_file,
    ensemble_evaluate_file,
    ensemble_evaluate_files,
    plot_ensemble_uncertainty,
)
from validation.dataset_shift import (
    collect_sv_samples,
    compute_pairwise_stats,
    plot_kde_overlay,
    plot_heatmap,
)
from validation.conformal import (
    conformal_quantile,
    calibrate_depth,
    predict_depth_band,
    evaluate_coverage,
    coverage_curve,
    calibrate_profile,
    predict_profile_band,
    evaluate_profile_coverage,
    calibrate_locally_weighted,
    predict_locally_weighted_band,
    compute_transition_width,
    plot_sigma_vs_error,
    calibrate_pixel,
    predict_pixel_set,
    extract_bottom_depths,
    plot_conformal_band,
    plot_coverage_curve,
)

__all__ = [
    "mc_dropout_predict",
    "prediction_entropy",
    "bottom_uncertainty_band",
    "threshold_sensitivity",
    "calibration_curve",
    "load_ensemble",
    "ensemble_predict_file",
    "ensemble_evaluate_file",
    "ensemble_evaluate_files",
    "plot_ensemble_uncertainty",
    "collect_sv_samples",
    "compute_pairwise_stats",
    "plot_kde_overlay",
    "plot_heatmap",
    "conformal_quantile",
    "calibrate_depth",
    "predict_depth_band",
    "evaluate_coverage",
    "coverage_curve",
    "calibrate_profile",
    "predict_profile_band",
    "evaluate_profile_coverage",
    "calibrate_locally_weighted",
    "predict_locally_weighted_band",
    "compute_transition_width",
    "plot_sigma_vs_error",
    "calibrate_pixel",
    "predict_pixel_set",
    "extract_bottom_depths",
    "plot_conformal_band",
    "plot_coverage_curve",
]
