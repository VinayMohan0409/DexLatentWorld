"""Tests of the planner gates: the contact check, the hover goal and its release, the secure stages, their streak and caps, and the weights of each stage."""
import sim_settings as cfg
import numpy as np
import pytest

from lawm.sim.dexwm.planning import plan_costs as C
from lawm.sim.dexwm.planning import phase_gates as S


def grip(per_side=(15, 15), gap=0.01):
    """Object points on two clusters at x = +-0.02 and hand points gap away from each cluster on the outer side; returns (coords, segment)."""
    obj = np.array([[0.02, 0, 0]] * 20 + [[-0.02, 0, 0]] * 20)
    hand = np.array([[0.02 + gap, 0, 0]] * per_side[0] + [[-0.02 - gap, 0, 0]] * per_side[1])
    table = np.array([[1.0, 1.0, 0.0]] * 3)
    coords = np.concatenate([hand, obj, table])
    segment = np.concatenate([np.full(len(hand), 0), np.full(len(obj), 1), np.full(3, 2)])
    return coords, segment


def test_contact_counts_points_within_two_centimetres_and_finds_the_opposed_pair():
    """Hand points 1 cm from the two sides of the object are in contact and face each other; 3 cm away they are not."""
    coords, segment = grip()
    contact = S.grasp_contact(coords[segment == 0], coords[segment == 1], cfg.secure())
    assert contact.n_contact == 30 and contact.has_antipodal_pair and contact.best_opposition == pytest.approx(1.0, abs=1e-6)
    far = S.grasp_contact(*(lambda c, s: (c[s == 0], c[s == 1]))(*grip(gap=0.03)), cfg.secure())
    assert far.n_contact == 0 and far.best_opposition is None and not far.has_antipodal_pair


def test_contacts_on_one_side_are_not_opposed_and_the_angle_limit_is_30_degrees():
    """Two contacts looking the same way are not a pair; contacts 160 degrees apart pass and 120 degrees apart fail (the limit is strictly more than cos 30 degrees)."""
    obj = np.zeros((10, 3))
    hand_same = np.array([[0.01, 0.001, 0], [0.01, -0.001, 0]])
    assert not S.grasp_contact(hand_same, obj, cfg.secure()).has_antipodal_pair
    for degrees, passes in ((160, True), (120, False)):
        angle = np.radians(degrees)
        hand = np.array([[0.01, 0, 0], [0.01 * np.cos(angle), 0.01 * np.sin(angle), 0]])
        assert S.grasp_contact(hand, obj, cfg.secure()).has_antipodal_pair is passes


def test_fewer_than_two_contacts_give_no_opposition_value():
    """One touching point is a contact but has no opposition to report."""
    contact = S.grasp_contact(np.array([[0.01, 0, 0], [0.5, 0, 0]]), np.zeros((5, 3)), cfg.secure())
    assert contact.n_contact == 1 and contact.best_opposition is None and not contact.has_antipodal_pair


def hover_scene():
    """A goal of 4 hand points at z = 0.5, 2 object points and 2 table points, and a current cloud with the same layout."""
    segment = np.array([0] * 4 + [1] * 2 + [2] * 2)
    goal = np.zeros((8, 3))
    goal[:4] = [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5], [0.1, 0.1, 0.5]]
    goal[4:] = [[0.3, 0.3, 0.9], [0.3, 0.4, 0.9], [1, 1, 0], [1, 2, 0]]
    return goal, segment


def current_at(goal, segment, dz_mm):
    """The goal cloud with the hand points moved dz_mm above the HOVER goal (hover goal = goal + 0.2 m in z)."""
    now = goal.copy()
    now[segment == 0, 2] += 0.2 + dz_mm / 1000.0
    return now


def test_hover_goal_raises_only_the_hand_points_by_the_margin():
    """The hover goal equals the real goal except for z += 0.2 m on the hand rows; the real goal is kept untouched."""
    goal, segment = hover_scene()
    gate = S.HoverGate(goal, segment, cfg.hover())
    expected = goal.copy()
    expected[:4, 2] += 0.2
    assert np.array_equal(gate.goal, expected) and np.array_equal(gate.real_goal, goal)


def test_hover_releases_at_fifty_millimetres_and_not_before():
    """49.9 mm from the hover goal switches to the real goal; 50.1 mm does not."""
    goal, segment = hover_scene()
    for dz, switches in ((50.1, False), (49.9, True), (10.0, True)):
        gate = S.HoverGate(goal, segment, cfg.hover())
        assert gate.update(current_at(goal, segment, dz), segment, replan=3) is switches
        assert gate.active is (not switches)
        assert np.array_equal(gate.goal, goal if switches else gate.hover_goal)


def test_hover_releases_at_the_replan_cap_even_when_far_and_reports_the_switch_once():
    """A hand 300 mm away still ends the hover at replan 40, and later updates do nothing."""
    goal, segment = hover_scene()
    gate = S.HoverGate(goal, segment, cfg.hover())
    far = current_at(goal, segment, 300.0)
    assert gate.update(far, segment, replan=39) is False
    assert gate.update(far, segment, replan=40) is True
    assert gate.update(far, segment, replan=41) is False and np.array_equal(gate.goal, goal)


def test_hover_distance_uses_the_hand_centre_only():
    """Moving the object and table rows of the current cloud does not change the measured distance."""
    goal, segment = hover_scene()
    gate = S.HoverGate(goal, segment, cfg.hover())
    now = current_at(goal, segment, 80.0)
    gate.update(now, segment, replan=0)
    first = gate.distance_mm
    now[4:] += 5.0
    gate.update(now, segment, replan=0)
    assert first == pytest.approx(80.0) and gate.distance_mm == first


def test_place_weights_of_the_two_stages():
    """Stage 1: hand off, grasp quality 2; stage 2 adds rigid attachment 1, object goal 2 and grasp tight 2; nothing else is on."""
    one, two = cfg.weights("place")[0], cfg.weights("place")[1]
    assert (one.hand_scale, one.grasp_quality, one.rigid_attachment, one.object_goal, one.grasp_tight, one.object_still) == (0.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    assert (two.hand_scale, two.grasp_quality, two.rigid_attachment, two.object_goal, two.grasp_tight, two.object_still) == (0.0, 2.0, 1.0, 2.0, 2.0, 0.0)
    assert isinstance(one, C.CostWeights)


def test_secure_needs_two_replans_in_a_row_with_thirty_contacts_and_an_opposed_pair():
    """One secure replan is not enough; a failing replan resets the count; two in a row start stage 2."""
    gate = S.SecureGate(cfg.secure(), cfg.weights("place"))
    good, bad = grip(), grip(per_side=(15, 0))
    assert gate.weights.object_goal == 0.0
    assert gate.update(*good, replan=0) is False and gate.streak == 1
    assert gate.update(*bad, replan=1) is False and gate.streak == 0
    assert gate.update(*good, replan=2) is False
    assert gate.update(*good, replan=3) is True and gate.stage == 2
    assert gate.weights.object_goal == 2.0 and gate.weights.rigid_attachment == 1.0


def test_secure_contact_count_boundary_is_thirty():
    """29 contact points never count, 30 do."""
    for per_side, secure in (((15, 14), False), ((15, 15), True)):
        gate = S.SecureGate(cfg.secure(), cfg.weights("place"))
        gate.update(*grip(per_side), replan=0)
        gate.update(*grip(per_side), replan=1)
        assert (gate.stage == 2) is secure


def test_secure_ends_at_the_replan_cap_without_a_grip_and_reports_once():
    """With no contact at all stage 2 still starts at replan 40, once."""
    gate = S.SecureGate(cfg.secure(), cfg.weights("place"))
    nothing = grip(gap=0.5)
    assert gate.update(*nothing, replan=39) is False and gate.stage == 1
    assert gate.update(*nothing, replan=40) is True and gate.stage == 2
    assert gate.update(*nothing, replan=41) is False


def test_the_gates_do_not_share_state_between_instances():
    """A second gate starts fresh after the first has finished (the original kept such values in module globals)."""
    first = S.SecureGate(cfg.secure(), cfg.weights("place"))
    first.update(*grip(), replan=0)
    first.update(*grip(), replan=1)
    second = S.SecureGate(cfg.secure(), cfg.weights("place"))
    assert first.stage == 2 and second.stage == 1 and second.streak == 0
