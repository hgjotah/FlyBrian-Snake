"""Parameters used both by experiments and by the desktop application."""
from dataclasses import dataclass, field, replace
import math


@dataclass(frozen=True)
class LIF:
    dt: float = 0.1
    rest: float = -52.0
    reset: float = -52.0
    threshold: float = -45.0
    membrane_tau: float = 20.0
    synapse_tau: float = 5.0
    refractory: float = 2.2
    delay: float = 1.8
    unit_weight: float = 0.275
    stimulus_jump: float = 68.75

    def __post_init__(self):
        if not (all(math.isfinite(v) for v in vars(self).values())
                and 0 < self.dt <= self.delay and self.membrane_tau > 0 and self.synapse_tau > 0
                and self.reset < self.threshold and self.rest < self.threshold
                and self.refractory >= 0 and self.unit_weight > 0 and self.stimulus_jump > 0):
            raise ValueError("Parámetros LIF inválidos")
        for value in (self.delay, self.refractory):
            if not math.isclose(value / self.dt, round(value / self.dt), abs_tol=1e-8):
                raise ValueError("El retraso y el refractario deben ser múltiplos de dt")


@dataclass(frozen=True)
class Plasticity:
    tau_ms: float = 350.0
    post_tau_ms: float = 100.0
    learning_rate: float = 0.075
    min_scale: float = 0.20
    post_required: bool = True

    def __post_init__(self):
        if not (type(self.post_required) is bool
                and all(math.isfinite(v) for v in (self.tau_ms, self.post_tau_ms, self.learning_rate, self.min_scale))
                and self.tau_ms > 0 and self.post_tau_ms > 0 and 0 < self.learning_rate <= 1
                and 0 <= self.min_scale < 1):
            raise ValueError("Parámetros de plasticidad inválidos")


@dataclass(frozen=True)
class Rewards:
    apple: float = 30.0
    collision: float = -60.0
    distance: float = 0.25
    wall: float = 12.0
    idle: float = 0.0
    profile: str = "edge_target"
    route: float = 2.0
    completion: float = 150.0
    avoidance: float = 18.0

    def __post_init__(self):
        if not (self.apple > self.wall >= 0 and -self.collision > self.wall
                and min(self.distance, self.idle, self.route, self.completion, self.avoidance) >= 0
                and self.profile in ("edge_target", "edge_aware", "graded", "legacy")
                and all(math.isfinite(v) for v in (self.apple, self.collision, self.distance, self.wall, self.idle,
                                               self.route, self.completion, self.avoidance))):
            raise ValueError("Magnitudes de recompensa inválidas")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 1
    size: int = 7
    stage: int = 4
    max_steps: int = 100
    food_partition: str = "all"
    association_valence: str = "appetitive"
    decision_ms: float = 60.0
    outcome_ms: float = 30.0
    sensory_hz: float = 90.0
    reinforcement_hz: float = 120.0
    plasticity: bool = True
    dopamine: bool = True
    compact_trace: bool = False
    practice: str = "snake"
    wall_memory_hz: float = 0.0
    wall_preview_cells: int = 1
    synaptic_signal: str = "curated"
    save_every: int = 10
    eval_every: int = 0
    rewards: Rewards = field(default_factory=Rewards)
    lif: LIF = field(default_factory=LIF)
    learning: Plasticity = field(default_factory=Plasticity)

    def __post_init__(self):
        integers = (self.seed, self.size, self.stage, self.max_steps, self.save_every, self.eval_every,
                    self.wall_preview_cells)
        durations_and_rates = (self.decision_ms, self.outcome_ms, self.sensory_hz, self.reinforcement_hz, self.wall_memory_hz)
        if not (all(type(v) is int for v in integers) and all(math.isfinite(v) for v in durations_and_rates)
                and all(type(v) is bool for v in (self.plasticity, self.dopamine, self.compact_trace))
                and 0 <= self.seed < 2**63 and 4 <= self.size <= 100
                and self.stage in range(5) and self.max_steps >= 0
                and self.practice in ("snake", "routes", "walls", "edge_reset") and self.wall_memory_hz >= 0
                and 0 <= self.wall_preview_cells <= 3
                and self.synaptic_signal in ("curated", "contact_probabilities")
                and self.food_partition in ("all", "train", "heldout")
                and self.association_valence in ("appetitive", "aversive") and self.save_every >= 1
                and self.eval_every >= 0 and self.sensory_hz >= 0 and self.reinforcement_hz >= 0
                and self.decision_ms > 0 and self.outcome_ms > 0):
            raise ValueError("Configuración de entrenamiento inválida")
        if self.practice != "snake" and self.stage != 4:
            raise ValueError("La práctica requiere la etapa 4")
        if self.practice in ("routes", "walls") and self.max_steps == 0:
            raise ValueError("Los ejercicios cortos requieren un límite de pasos")
        if self.stage == 2 and self.max_steps == 0:
            raise ValueError("El toroide de etapa 2 requiere un límite de pasos")
        for value in (self.decision_ms, self.outcome_ms):
            if not math.isclose(value / self.lif.dt, round(value / self.lif.dt), abs_tol=1e-8):
                raise ValueError("Las ventanas deben ser múltiplos de dt")


def burst_config(config: RunConfig) -> RunConfig:
    """Short stage-4 episodes for rapid practice on the same full connectome."""
    dt = config.lif.dt
    return replace(config, stage=4, max_steps=12, save_every=50, eval_every=0, compact_trace=True,
                   decision_ms=max(1, round(36 / dt)) * dt, outcome_ms=max(1, round(18 / dt)) * dt)
