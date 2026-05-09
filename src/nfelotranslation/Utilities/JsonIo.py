'''
JSON config I/O shared by inference and training.

* Envelope format: ``{ "metadata": {...}, "params": {...} }`` plus legacy flat JSON.
* Seasonal file resolution for ``name_YYYY.json`` with fallback to the latest prior year.

Training code writes envelopes; inference reads them. Both import this module only
(no dependency on the optional ``training/`` package).
'''

## built-ins ##
import datetime
import json
import os
import pathlib
import re
import secrets
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ConfigMetadata:
    '''
    Metadata attached to every fitted model config file.

    Enables cross-fitter traceability: when a training orchestrator runs
    multiple fitters in sequence, a shared pipeline_id lets downstream fitters
    verify upstream dependencies came from the same run.
    '''
    pipeline_id: Optional[str] = None
    generated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_id': self.pipeline_id,
            'generated_at': self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ConfigMetadata':
        if not data:
            return cls()
        return cls(
            pipeline_id=data.get('pipeline_id'),
            generated_at=data.get('generated_at'),
        )

    @classmethod
    def new(cls, pipeline_id: Optional[str] = None) -> 'ConfigMetadata':
        '''Stamp a fresh metadata record with the current UTC timestamp.'''
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        return cls(pipeline_id=pipeline_id, generated_at=now)


def generate_pipeline_id() -> str:
    '''Generate a short random identifier for a training run.'''
    return secrets.token_hex(4)


def write_config_envelope(
    filepath: str,
    params: Dict[str, Any],
    metadata: ConfigMetadata,
) -> None:
    '''Write a config file with the standard {metadata, params} envelope.'''
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        'metadata': metadata.to_dict(),
        'params': params,
    }
    with open(filepath, 'w') as f:
        json.dump(envelope, f, indent=4)


def read_config_envelope(filepath: str) -> Tuple[Dict[str, Any], ConfigMetadata]:
    '''
    Read a config file and return (params, metadata).

    Supports enveloped format and legacy flat payload at the top level.
    '''
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'params' in data and 'metadata' in data:
        return data['params'], ConfigMetadata.from_dict(data.get('metadata'))
    return data, ConfigMetadata()


def find_config_path(
    model_name: str,
    season: int,
    config_dir: str,
    extension: str = '.json',
) -> Optional[str]:
    '''
    Resolve a seasonal config file with fallback to the most recent prior file.

    Naming convention:
        ``{model_name}_{season}.json`` = trained through season-1, valid for season
        (e.g. key_model_2024.json trained through 2023, valid for 2024).
    '''
    if not os.path.isdir(config_dir):
        return None
    pattern = re.compile(
        rf'^{re.escape(model_name)}_(\d{{4}}){re.escape(extension)}$'
    )
    candidates = {}
    for filename in os.listdir(config_dir):
        match = pattern.match(filename)
        if match:
            file_season = int(match.group(1))
            candidates[file_season] = os.path.join(config_dir, filename)
    if not candidates:
        return None
    if season in candidates:
        return candidates[season]
    valid = {s: p for s, p in candidates.items() if s <= season}
    if not valid:
        return None
    fallback_season = max(valid.keys())
    warnings.warn(
        f'{model_name}: no config for season {season}, '
        f'falling back to {model_name}_{fallback_season}{extension}',
        stacklevel=2,
    )
    return valid[fallback_season]
