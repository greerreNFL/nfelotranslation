'''
Train the KeyModel (credibility-weighted trackers) across all available seasons.

Usage (from repository root, with PYTHONPATH including ``.`` and ``src``):
    python -m training.Scripts.train_key_model
'''

## local ##
from training.Distribution.Key.KeyModelFitter import KeyModelFitter


def main():
    fitter = KeyModelFitter()
    result = fitter.fit()
    print(result.summary())


if __name__ == '__main__':
    main()
