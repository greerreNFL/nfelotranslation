'''
Structured types for training validation reports (JSON-serializable).

Used by ``training/`` validators; not required for ``Translator`` inference.
'''

## built-ins ##
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationCheck:
    '''A gated validation check with a pass/fail threshold.'''
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'threshold': self.threshold,
            'passed': self.passed,
            'detail': self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = '') -> 'ValidationCheck':
        return cls(
            name=data.get('name', name),
            value=float(data['value']),
            threshold=float(data['threshold']),
            passed=bool(data['passed']),
            detail=data.get('detail', ''),
        )


@dataclass
class TrackedMetric:
    '''An informational metric for human review — no pass/fail gate.'''
    name: str
    value: float
    detail: str = ''
    per_season: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'value': self.value,
            'detail': self.detail,
        }
        if self.per_season:
            result['per_season'] = {str(k): v for k, v in self.per_season.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: str = '') -> 'TrackedMetric':
        metric_name = data.get('name', name)
        per_season_raw = data.get('per_season', {})
        per_season = {int(k): float(v) for k, v in per_season_raw.items()}
        return cls(
            name=metric_name,
            value=float(data['value']),
            detail=data.get('detail', ''),
            per_season=per_season,
        )


@dataclass
class ValidationReport:
    '''Complete validation output from a training run.'''
    model_name: str
    data_through: int
    checks: List[ValidationCheck] = field(default_factory=list)
    metrics: List[TrackedMetric] = field(default_factory=list)
    tables: Dict[str, str] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        status = 'PASSED' if self.all_passed else 'FAILED'
        n_passed = sum(1 for c in self.checks if c.passed)
        header = f'{self.model_name} Validation (data through {self.data_through})'
        lines = [header, '=' * len(header)]
        if self.checks:
            lines.append('Checks:')
            for c in self.checks:
                mark = 'PASS' if c.passed else 'FAIL'
                line = f'  [{mark}] {c.name}: {c.value:.6f} (threshold: {c.threshold:.6f})'
                if c.detail:
                    line += f'  — {c.detail}'
                lines.append(line)
        if self.metrics:
            lines.append('Metrics:')
            for m in self.metrics:
                line = f'  {m.name}: {m.value:.6f}'
                if m.detail:
                    line += f'  — {m.detail}'
                lines.append(line)
                if m.per_season:
                    seasons_str = ', '.join(
                        f'{s}: {v:.6f}' for s, v in sorted(m.per_season.items())
                    )
                    lines.append(f'    per_season: {seasons_str}')
        for table_name, table_content in self.tables.items():
            lines.append(f'{table_name}:')
            for table_line in table_content.split('\n'):
                lines.append(f'  {table_line}')
        lines.append(f'{status} ({n_passed}/{len(self.checks)} checks)')
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'data_through': self.data_through,
            'checks': {c.name: c.to_dict() for c in self.checks},
            'metrics': {m.name: m.to_dict() for m in self.metrics},
            'tables': self.tables,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationReport':
        raw_checks = data.get('checks', {})
        raw_metrics = data.get('metrics', {})
        if isinstance(raw_checks, list):
            checks = [ValidationCheck.from_dict(c) for c in raw_checks]
        else:
            checks = [ValidationCheck.from_dict(v, name=k) for k, v in raw_checks.items()]
        if isinstance(raw_metrics, list):
            metrics = [TrackedMetric.from_dict(m) for m in raw_metrics]
        else:
            metrics = [TrackedMetric.from_dict(v, name=k) for k, v in raw_metrics.items()]
        return cls(
            model_name=data['model_name'],
            data_through=int(data['data_through']),
            checks=checks,
            metrics=metrics,
            tables=data.get('tables', {}),
        )

    def to_file(self, filepath: str) -> None:
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(cls, filepath: str) -> 'ValidationReport':
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
