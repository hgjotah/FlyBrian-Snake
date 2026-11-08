"""Fixed sensory and motor interfaces. Only the motor readout chooses an action."""
from dataclasses import dataclass
import math

import numpy as np

from .game import Direction, Observation, Reward, VECTORS
from .simulator import Stimulus


def sensory(observation: Observation, groups, gain: float, *, profile: str = "edge_target",
            wall_memory_hz: float = 0.0, wall_preview_cells: int = 0) -> tuple[Stimulus, ...]:
    obs = observation
    signals = {}
    edge_sensors = profile in ("edge_target", "edge_aware")
    frontal_wall = obs.walls and obs.wall_distance(obs.direction) == 0
    visible_flank = 0
    frontal_target = False
    preview_events = []
    if obs.apple is not None:
        dx = obs.apple[0] - obs.head[0]
        dy = obs.apple[1] - obs.head[1]
        if not obs.walls:
            half = obs.size // 2
            dx = (dx + half) % obs.size - half
            dy = (dy + half) % obs.size - half
        fx, fy = VECTORS[obs.direction]
        rx, ry = VECTORS[obs.direction.turn(1)]
        front, lateral = dx * fx + dy * fy, dx * rx + dy * ry
        frontal_target = front > 0 and lateral == 0
        if profile == "edge_target" and frontal_wall and front == 0 and lateral:
            visible_flank = -1 if lateral < 0 else 1
        for signal, value, tag in (("food_front", front, "forward"), ("food_back", -front, "back"),
                                   ("food_left", -lateral, "left"), ("food_right", lateral, "right")):
            if value > 0:
                intensity = gain * value / (obs.size - 1)
                if edge_sensors and signal in ("food_left", "food_right"):
                    intensity = gain * (.30 + .70 * value / (obs.size - 1))
                if not edge_sensors or signal != "food_back":
                    signals[signal] = intensity
                signals["tag_" + tag] = intensity * 0.8
        if profile == "edge_target" and frontal_target:
            signals["food_front"] *= .5
    if obs.walls:
        for relative, name in ((-1, "left"), (0, "front"), (1, "right")):
            intensity = gain / (1 + obs.wall_distance(obs.direction.turn(relative))) ** 2
            signals["tag_wall_" + name] = intensity * 0.8
            if relative:
                lateral_flow = .08 if profile == "edge_target" and frontal_target and not frontal_wall else 1.0
                signals["wall_" + name] = intensity * lateral_flow * (.2 if relative == visible_flank else 1)
        preview = min(wall_preview_cells, (obs.size - 2) // 2)
        frontal_distance = obs.wall_distance(obs.direction)
        if (edge_sensors or wall_preview_cells > 0) and (frontal_wall or 0 < frontal_distance <= preview):
            warning = .15 * (1 - frontal_distance / (preview + 1)) ** 2 if frontal_distance else 1.0
            if frontal_target and not frontal_wall:
                warning *= .15  # A visible fruit lies before the boundary.
            left = obs.wall_distance(obs.direction.turn(-1))
            right = obs.wall_distance(obs.direction.turn(1))
            for side, clearance, relative in (("left", left, -1), ("right", right, 1)):
                # A fruit on the lateral sightline lies before the barrier, not inside it.
                proximity = 0.0 if relative == visible_flank else 1 - clearance / (obs.size - 1)
                intensity = gain * warning * (.35 + 1.55 * proximity**2)
                if frontal_wall:
                    signals["loom_" + side] = intensity
                else:
                    preview_events.append(Stimulus("boundary_" + side, groups["loom_" + side], intensity,
                                                   stream="boundary"))
            # A calibrated sensory asymmetry reduces bilateral cancellation at mid-edge.
            if frontal_wall and left == right and not visible_flank:
                signals["loom_left"] *= 1.5
    signals["heading_" + obs.direction.name.lower()] = gain * 0.25
    events = [Stimulus(name, groups[name], intensity) for name, intensity in signals.items()]
    if obs.walls and wall_memory_hz > 0:
        proximity = max(0.0, (4 - obs.wall_distance(obs.direction)) / 4)
        if proximity > 0:
            events.append(Stimulus("wall_memory", groups["conditioning_A"], wall_memory_hz * proximity))
    return tuple(events + preview_events)


def reinforcement(reward: Reward, groups, intensity: float, *, profile: str = "edge_target") -> tuple[Stimulus, ...]:
    if profile in ("edge_target", "edge_aware"):
        net = (reward.apple + reward.collision + reward.distance + reward.wall + reward.idle
               + reward.route + reward.avoidance + reward.practice_failure + reward.edge_contact)
        positive, negative = max(0.0, net), max(0.0, -net)
    else:
        positive = reward.apple + reward.avoidance + max(0.0, reward.distance) + max(0.0, reward.route)
        negative = (-reward.collision - reward.wall - reward.idle - reward.practice_failure - reward.edge_contact
                    + max(0.0, -reward.distance) + max(0.0, -reward.route))
    appetite_scale, aversion_scale = (2.0, 4.0) if profile == "legacy" else (8.0, 18.0)
    appetitive = -math.expm1(-positive / appetite_scale)
    if positive > 0 and reward.apple > 0 and reward.avoidance > 0:
        # Make the compound task prize distinguishable from an ordinary apple.
        appetitive -= .75 * math.expm1(-reward.avoidance / 18.0)
    # An ordinary apple already nearly saturates the base pulse.
    appetitive -= 2 * math.expm1(-reward.completion / 60.0)
    aversive = -math.expm1(-negative / aversion_scale)
    if reward.edge_contact < 0:
        aversive -= math.expm1(reward.edge_contact / 60.0)
    events = []
    for channel, fraction in (("appetitive", appetitive), ("aversive", aversive)):
        if intensity > 0 and fraction > 0:
            events.append(Stimulus(channel, groups[channel], intensity * fraction))
    return tuple(events)


@dataclass(frozen=True)
class Decision:
    turn: int
    left_spikes: int
    right_spikes: int
    provenance: str

    def absolute(self, heading: Direction):
        return heading.turn(self.turn)


def motor(counts: np.ndarray, groups) -> Decision:
    left, right = groups["motor_left"], groups["motor_right"]
    l_count, r_count = int(counts[left].sum()), int(counts[right].sum())
    difference = r_count * len(left) - l_count * len(right)
    turn = int(difference > 0) - int(difference < 0)
    return Decision(turn, l_count, r_count, "spikes descendentes" if turn else "continuidad por empate")
