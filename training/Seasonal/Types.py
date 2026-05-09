'''
Seasonal training data structures.
'''

## built-ins ##
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SeasonalDiagnostics:
    '''
    Diagnostics for a single season's out-of-sample evaluation.

    Parameters:
    * season: the season evaluated
    * metrics: fitter-specific metric names → values
    * metadata: optional extra information
    '''
    season: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'season': self.season,
            'metrics': self.metrics,
            'metadata': self.metadata,
        }

    def to_file(self, filepath: str) -> None:
        '''Save to JSON file, creating parent directories as needed.'''
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeasonalDiagnostics':
        return cls(
            season=int(data['season']),
            metrics=data['metrics'],
            metadata=data.get('metadata', {}),
        )


@dataclass
class SeasonalResult:
    '''
    Aggregate result from a full seasonal training run.

    Parameters:
    * model_name: identifier for the model trained
    * seasons_trained: list of seasons processed
    * per_season: per-season diagnostics (out-of-sample)
    * aggregate_metrics: mean of each metric across seasons
    * output_dir: directory where config files were saved
    '''
    model_name: str
    seasons_trained: List[int]
    per_season: List[SeasonalDiagnostics]
    aggregate_metrics: Dict[str, float]
    output_dir: str

    def summary(self) -> str:
        '''Human-readable summary table of per-season and aggregate metrics.'''
        if not self.per_season:
            return f'{self.model_name}: no seasons trained'
        ## collect all metric names ##
        metric_names = list(self.per_season[0].metrics.keys())
        ## header ##
        col_width = max(12, *(len(m) for m in metric_names))
        header = f'{"season":>8}'
        for name in metric_names:
            header += f'  {name:>{col_width}}'
        lines = [
            f'{self.model_name} — Seasonal Training Results',
            '=' * len(header),
            header,
            '-' * len(header),
        ]
        ## per-season rows ##
        for diag in self.per_season:
            row = f'{diag.season:>8}'
            for name in metric_names:
                val = diag.metrics.get(name, float('nan'))
                row += f'  {val:>{col_width}.6f}'
            lines.append(row)
        ## aggregate row ##
        lines.append('-' * len(header))
        agg_row = f'{"mean":>8}'
        for name in metric_names:
            val = self.aggregate_metrics.get(name, float('nan'))
            agg_row += f'  {val:>{col_width}.6f}'
        lines.append(agg_row)
        lines.append(f'\nOutput: {self.output_dir}')
        return '\n'.join(lines)
