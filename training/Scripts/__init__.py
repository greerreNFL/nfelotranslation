'''
Training Scripts — entry points for model training and validation.

Pipeline dependency order:
  1. Recalibrator  — no upstream dependencies
  2. SpreadMapper  — depends on Recalibrator (uses ml_wp_cal)
  3. KeyModel      — depends on Recalibrator + SpreadMapper (uses ml_wp_cal, spread coords)
  4. Distribution validation — system-level check after all components are trained

DataLoader is reset between steps so downstream fitters see upstream configs.
A shared pipeline_id links all steps, enabling downstream fitters to verify
their upstream dependencies came from the same training run.
'''

## built-ins ##
from typing import Any, Dict, List, Optional, Tuple

## local ##
from nfelotranslation.Utilities.JsonIo import generate_pipeline_id
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from training.Base.Fitter import Fitter
from training.Calibration.RecalibratorFitter import RecalibratorFitter
from training.Data.DataLoader import DataLoader
from training.Distribution.Key.KeyModelFitter import KeyModelFitter
from training.SpreadMap.SpreadMapperFitter import SpreadMapperFitter
from training.Validation import MarginDistributionValidator


## pipeline order — each step depends on the ones above it ##
_PIPELINE_ORDER = [
    RecalibratorFitter,
    SpreadMapperFitter,
    KeyModelFitter,
]

## seasonal fitters get seasons= passed to fit() ##
_SEASONAL = {SpreadMapperFitter, KeyModelFitter}


def train_all(
    seasons: Optional[List[int]] = None,
    validate: bool = True,
) -> List[Tuple[Fitter, Any, Optional[ValidationReport]]]:
    '''
    Train all fitters in dependency order with a shared pipeline_id.

    Resets the DataLoader between steps so downstream fitters pick up
    upstream configs (e.g., recalibrated win probs, spread mapper).

    Parameters:
    * seasons: optional list of seasons for seasonal fitters (default: all available)
    * validate: whether to run validation after fitting (default: True)

    Returns:
    * list of (fitter, fit_result, validation_report) tuples
    '''
    pipeline_id = generate_pipeline_id()
    print(f'Pipeline ID: {pipeline_id}')
    print()
    results = []
    for fitter_cls in _PIPELINE_ORDER:
        ## reset DataLoader so this step sees upstream configs ##
        DataLoader.reset()
        fitter = fitter_cls(pipeline_id=pipeline_id)
        print(f'{"=" * 60}')
        print(f'Training {fitter.model_name}...')
        print(f'{"=" * 60}')
        ## fit ##
        if fitter_cls in _SEASONAL:
            fit_result = fitter.fit(seasons=seasons)
        else:
            fit_result = fitter.fit()
        ## diagnostics ##
        if hasattr(fit_result, 'summary'):
            print(fit_result.summary())
        else:
            diag = fitter.diagnostics()
            print(diag.summary())
        ## validate ##
        report = None
        if validate:
            report = fitter.validate()
            print(report.summary())
            filepath = fitter.save_validation(report)
            print(f'Saved: {filepath}')
        results.append((fitter, fit_result, report))
        print()
    ## -------- distribution-level validation (system integration check) -------- ##
    if validate:
        DataLoader.reset()
        dist_validator = MarginDistributionValidator(games=DataLoader.get().games)
        print(f'{"=" * 60}')
        print(f'Validating margin_distribution...')
        print(f'{"=" * 60}')
        dist_report = dist_validator.validate()
        print(dist_report.summary())
        dist_path = dist_validator.save(dist_report)
        print(f'Saved: {dist_path}')
        print()
    return results
