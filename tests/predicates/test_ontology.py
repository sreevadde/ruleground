"""Tests for the predicate ontology (single source of truth)."""

from ruleground.predicates.ontology import (
    PREDICATE_ONTOLOGY,
    ALL_PREDICATE_NAMES,
    NUM_PREDICATES,
    PREDICATE_NAME_TO_IDX,
    PREDICATE_REGISTRY,
    INSTANT_PREDICATES,
    STATE_PREDICATES,
    SPATIAL_PREDICATES,
    Sport,
    PredicateType,
    get_predicates_for_sport,
    get_predicate_names_for_sport,
    get_predicate_indices_for_sport,
    get_sport_mask,
    get_instant_predicate_indices,
    get_state_predicate_indices,
)


class TestPredicateOntology:
    """Verify ontology completeness and consistency."""

    def test_total_predicate_count(self):
        """Paper Table 7 defines exactly 20 predicates."""
        assert NUM_PREDICATES == 20
        assert len(PREDICATE_ONTOLOGY) == 20
        assert len(ALL_PREDICATE_NAMES) == 20

    def test_all_names_unique(self):
        assert len(set(ALL_PREDICATE_NAMES)) == NUM_PREDICATES

    def test_name_to_idx_consistent(self):
        for i, name in enumerate(ALL_PREDICATE_NAMES):
            assert PREDICATE_NAME_TO_IDX[name] == i

    def test_registry_complete(self):
        for p in PREDICATE_ONTOLOGY:
            assert p.name in PREDICATE_REGISTRY
            assert PREDICATE_REGISTRY[p.name] is p

    def test_shared_predicates(self):
        shared = [p for p in PREDICATE_ONTOLOGY if p.sport == Sport.SHARED]
        assert len(shared) == 4
        shared_names = {p.name for p in shared}
        assert shared_names == {
            "ball_in_play", "contact_occurred",
            "contact_before_arrival", "incidental_contact",
        }

    def test_basketball_predicates(self):
        bball = [p for p in PREDICATE_ONTOLOGY if p.sport == Sport.BASKETBALL]
        assert len(bball) == 6
        names = {p.name for p in bball}
        assert "defender_set" in names
        assert "shooting_motion" in names
        assert "verticality_maintained" in names

    def test_football_predicates(self):
        fb = [p for p in PREDICATE_ONTOLOGY if p.sport == Sport.FOOTBALL]
        assert len(fb) == 4
        names = {p.name for p in fb}
        assert "ball_catchable" in names
        assert "within_five_yards" in names

    def test_soccer_predicates(self):
        soccer = [p for p in PREDICATE_ONTOLOGY if p.sport == Sport.SOCCER]
        assert len(soccer) == 6
        names = {p.name for p in soccer}
        assert "involved_in_play" in names
        assert "played_by_opponent" in names
        assert "denying_goal" in names

    def test_predicate_types_coverage(self):
        all_typed = INSTANT_PREDICATES | STATE_PREDICATES | SPATIAL_PREDICATES
        assert all_typed == set(ALL_PREDICATE_NAMES)

    def test_instant_predicates(self):
        expected_instant = {
            "contact_occurred", "contact_before_arrival",
            "pivot_foot_lifted", "ball_released",
            "offensive_push_off", "ball_contact_arm", "played_by_opponent",
        }
        assert INSTANT_PREDICATES == expected_instant

    def test_get_predicates_for_sport_includes_shared(self):
        for sport in ("basketball", "football", "soccer"):
            preds = get_predicates_for_sport(sport)
            names = {p.name for p in preds}
            assert "ball_in_play" in names, f"shared pred missing for {sport}"
            assert "contact_occurred" in names

    def test_basketball_sport_mask(self):
        mask = get_sport_mask("basketball")
        assert len(mask) == 20
        # Basketball predicates and shared should be True
        assert mask[PREDICATE_NAME_TO_IDX["defender_set"]] is True
        assert mask[PREDICATE_NAME_TO_IDX["ball_in_play"]] is True
        # Soccer-only should be False
        assert mask[PREDICATE_NAME_TO_IDX["offside_position"]] is False

    def test_sport_predicate_counts(self):
        # Basketball: 4 shared + 6 sport-specific = 10
        assert len(get_predicate_names_for_sport("basketball")) == 10
        # Football: 4 shared + 4 sport-specific = 8
        assert len(get_predicate_names_for_sport("football")) == 8
        # Soccer: 4 shared + 6 sport-specific = 10
        assert len(get_predicate_names_for_sport("soccer")) == 10

    def test_instant_and_state_indices_partition(self):
        instant_idx = set(get_instant_predicate_indices())
        state_idx = set(get_state_predicate_indices())
        assert instant_idx & state_idx == set(), "Overlap between instant and state"
        assert instant_idx | state_idx == set(range(NUM_PREDICATES))
