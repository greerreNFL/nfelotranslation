'''
Validator — base ABC for all model validators.

Produces ValidationReports from fitted models and data.  Validators
are intentionally decoupled from Fitters: they take explicit inputs
in their constructors rather than holding a fitter reference.
'''

## built-ins ##
from abc import ABC, abstractmethod

## local ##
from nfelotranslation.Utilities.ValidationTypes import ValidationReport


class Validator(ABC):
    '''
    Base class for model validators.

    Subclasses must implement:
    * model_name (property) — string identifier matching the fitter
    * validate() — run checks and return a ValidationReport
    '''

    @property
    @abstractmethod
    def model_name(self) -> str:
        '''Identifier for this model (matches the fitter's model_name).'''

    @abstractmethod
    def validate(self) -> ValidationReport:
        '''
        Run validation checks and tracked metrics.

        Returns:
        * ValidationReport with gated checks and tracked metrics
        '''
