"""
Predicate Ontology -- Single Source of Truth

All 20 predicates from the RuleGround paper (Table 7, Appendix B).
Every other module imports predicate definitions from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence


class PredicateType(Enum):
    """Temporal type of a predicate."""

    STATE = auto()    # Holds over an interval (e.g., defender_set)
    INSTANT = auto()  # Point event (e.g., contact_occurred)
    SPATIAL = auto()  # Spatial relation (e.g., restricted_area)


class Sport(Enum):
    """Sport domains."""

    SHARED = auto()      # Cross-sport predicates
    BASKETBALL = auto()
    FOOTBALL = auto()
    SOCCER = auto()


@dataclass(frozen=True)
class Predicate:
    """A single predicate in the ontology."""

    name: str
    ptype: PredicateType
    sport: Sport
    description: str
    example: str = ""


# ---------------------------------------------------------------------------
# Complete Predicate Ontology (20 predicates)
# Paper Table 7, Appendix B
# ---------------------------------------------------------------------------

PREDICATE_ONTOLOGY: List[Predicate] = [
    # ---- Shared (cross-sport) ----
    Predicate(
        name="ball_in_play",
        ptype=PredicateType.STATE,
        sport=Sport.SHARED,
        description="Ball/puck is active and live",
        example="Live vs. dead ball",
    ),
    Predicate(
        name="contact_occurred",
        ptype=PredicateType.INSTANT,
        sport=Sport.SHARED,
        description="Physical touch between entities",
        example="Hand-on-arm contact",
    ),
    Predicate(
        name="contact_before_arrival",
        ptype=PredicateType.INSTANT,
        sport=Sport.SHARED,
        description="Contact before ball/puck arrives at target",
        example="Pass interference timing",
    ),
    Predicate(
        name="incidental_contact",
        ptype=PredicateType.STATE,
        sport=Sport.SHARED,
        description="Contact not materially affecting play",
        example="Exception predicate",
    ),
    # ---- Basketball ----
    Predicate(
        name="defender_set",
        ptype=PredicateType.STATE,
        sport=Sport.BASKETBALL,
        description="Defender in legal guarding position",
        example="Charge/blocking call",
    ),
    Predicate(
        name="restricted_area",
        ptype=PredicateType.SPATIAL,
        sport=Sport.BASKETBALL,
        description="Contact in restricted area arc",
        example="Blocking exception zone",
    ),
    Predicate(
        name="pivot_foot_lifted",
        ptype=PredicateType.INSTANT,
        sport=Sport.BASKETBALL,
        description="Lift pivot foot before dribble",
        example="Travel initiation",
    ),
    Predicate(
        name="ball_released",
        ptype=PredicateType.INSTANT,
        sport=Sport.BASKETBALL,
        description="Ball left the player's hands",
        example="Shot release, pass release",
    ),
    Predicate(
        name="shooting_motion",
        ptype=PredicateType.STATE,
        sport=Sport.BASKETBALL,
        description="Player is in the act of shooting",
        example="Shooting foul determination",
    ),
    Predicate(
        name="verticality_maintained",
        ptype=PredicateType.STATE,
        sport=Sport.BASKETBALL,
        description="Defender stayed vertical during contest",
        example="Legal vertical defense",
    ),
    # ---- Football (American) ----
    Predicate(
        name="ball_catchable",
        ptype=PredicateType.STATE,
        sport=Sport.FOOTBALL,
        description="Thrown ball is catchable by the receiver",
        example="DPI applicability",
    ),
    Predicate(
        name="ball_in_air",
        ptype=PredicateType.STATE,
        sport=Sport.FOOTBALL,
        description="Forward pass is airborne",
        example="Pass interference window",
    ),
    Predicate(
        name="offensive_push_off",
        ptype=PredicateType.INSTANT,
        sport=Sport.FOOTBALL,
        description="Offensive player initiates illegal push",
        example="OPI initiation",
    ),
    Predicate(
        name="within_five_yards",
        ptype=PredicateType.SPATIAL,
        sport=Sport.FOOTBALL,
        description="Contact within 5-yard legal contact zone",
        example="Legal jam zone",
    ),
    # ---- Soccer ----
    Predicate(
        name="offside_position",
        ptype=PredicateType.SPATIAL,
        sport=Sport.SOCCER,
        description="Attacker beyond second-to-last defender",
        example="Attacker behind offside line",
    ),
    Predicate(
        name="involved_in_play",
        ptype=PredicateType.STATE,
        sport=Sport.SOCCER,
        description="Player actively involved in play from offside position",
        example="Offside active participation",
    ),
    Predicate(
        name="ball_contact_arm",
        ptype=PredicateType.INSTANT,
        sport=Sport.SOCCER,
        description="Ball touches arm below shoulder",
        example="Handball infraction",
    ),
    Predicate(
        name="arm_natural_position",
        ptype=PredicateType.STATE,
        sport=Sport.SOCCER,
        description="Arm is in a natural position relative to body",
        example="Handball exception",
    ),
    Predicate(
        name="played_by_opponent",
        ptype=PredicateType.INSTANT,
        sport=Sport.SOCCER,
        description="Ball was deliberately played by an opponent",
        example="Offside exception / deflection",
    ),
    Predicate(
        name="denying_goal",
        ptype=PredicateType.STATE,
        sport=Sport.SOCCER,
        description="Denying an obvious goal-scoring opportunity (DOGSO)",
        example="Red card / penalty determination",
    ),
]

# ---------------------------------------------------------------------------
# Derived constants and helpers
# ---------------------------------------------------------------------------

# Canonical ordered list of all predicate names
ALL_PREDICATE_NAMES: List[str] = [p.name for p in PREDICATE_ONTOLOGY]

# Total count
NUM_PREDICATES: int = len(PREDICATE_ONTOLOGY)

# Name -> index mapping
PREDICATE_NAME_TO_IDX: Dict[str, int] = {
    p.name: i for i, p in enumerate(PREDICATE_ONTOLOGY)
}

# Name -> Predicate mapping
PREDICATE_REGISTRY: Dict[str, Predicate] = {p.name: p for p in PREDICATE_ONTOLOGY}

# Sport -> set of predicate names (includes SHARED for every sport)
_SPORT_PREDICATES: Dict[Sport, FrozenSet[str]] = {}
for _sport in (Sport.BASKETBALL, Sport.FOOTBALL, Sport.SOCCER):
    _names = frozenset(
        p.name
        for p in PREDICATE_ONTOLOGY
        if p.sport == _sport or p.sport == Sport.SHARED
    )
    _SPORT_PREDICATES[_sport] = _names

# Type -> set of predicate names
INSTANT_PREDICATES: FrozenSet[str] = frozenset(
    p.name for p in PREDICATE_ONTOLOGY if p.ptype == PredicateType.INSTANT
)
STATE_PREDICATES: FrozenSet[str] = frozenset(
    p.name for p in PREDICATE_ONTOLOGY if p.ptype == PredicateType.STATE
)
SPATIAL_PREDICATES: FrozenSet[str] = frozenset(
    p.name for p in PREDICATE_ONTOLOGY if p.ptype == PredicateType.SPATIAL
)

# Sport string -> Sport enum
SPORT_FROM_STR: Dict[str, Sport] = {
    "basketball": Sport.BASKETBALL,
    "football": Sport.FOOTBALL,
    "soccer": Sport.SOCCER,
}

# Sport enum -> integer id (for tensor encoding)
SPORT_TO_ID: Dict[Sport, int] = {
    Sport.BASKETBALL: 0,
    Sport.FOOTBALL: 1,
    Sport.SOCCER: 2,
}
ID_TO_SPORT: Dict[int, Sport] = {v: k for k, v in SPORT_TO_ID.items()}


def get_predicates_for_sport(sport: Sport | str) -> List[Predicate]:
    """Return predicates relevant to a sport (sport-specific + shared)."""
    if isinstance(sport, str):
        sport = SPORT_FROM_STR[sport.lower()]
    return [
        p for p in PREDICATE_ONTOLOGY
        if p.sport == sport or p.sport == Sport.SHARED
    ]


def get_predicate_names_for_sport(sport: Sport | str) -> List[str]:
    """Return predicate names relevant to a sport."""
    return [p.name for p in get_predicates_for_sport(sport)]


def get_predicate_indices_for_sport(sport: Sport | str) -> List[int]:
    """Return indices (into ALL_PREDICATE_NAMES) for a sport's predicates."""
    names = get_predicate_names_for_sport(sport)
    return [PREDICATE_NAME_TO_IDX[n] for n in names]


def get_sport_mask(sport: Sport | str) -> List[bool]:
    """Return a boolean mask over all predicates: True if relevant to this sport."""
    if isinstance(sport, str):
        sport = SPORT_FROM_STR[sport.lower()]
    relevant = _SPORT_PREDICATES[sport]
    return [p.name in relevant for p in PREDICATE_ONTOLOGY]


def get_instant_predicate_indices() -> List[int]:
    """Return indices of instant-type predicates."""
    return [i for i, p in enumerate(PREDICATE_ONTOLOGY) if p.ptype == PredicateType.INSTANT]


def get_state_predicate_indices() -> List[int]:
    """Return indices of state-type predicates."""
    return [
        i for i, p in enumerate(PREDICATE_ONTOLOGY)
        if p.ptype in (PredicateType.STATE, PredicateType.SPATIAL)
    ]
