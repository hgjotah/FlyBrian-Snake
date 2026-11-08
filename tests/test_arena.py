from flybrain_snake.arena import Duel
from flybrain_snake.game import Direction


def test_duel_shares_seed_but_only_player_uses_keyboard(laboratory):
    duel = Duel(laboratory, 53, 30)
    before = duel.snapshot()
    assert before["brian"]["body"] == before["human"]["body"]
    assert not laboratory.game.allow_edge_food and duel.human.allow_edge_food
    duel.advance(Direction.UP)
    first = duel.snapshot()
    assert first["brian"]["steps"] == first["human"]["steps"] == 1
    assert first["human"]["body"][0] == (3, 2)
    assert first["brian"]["body"][0] == (4, 3)
    assert first["brian"]["direction"] == Direction.RIGHT
    assert laboratory.last_step["source"] == "Brian"
    assert laboratory.last_step["plasticity_enabled"]
    assert before["human"]["body"][0] == (3, 3)

    duel.advance(Direction.UP)
    duel.advance(Direction.UP)
    duel.advance(Direction.RIGHT)  # Brian crashes and the duel ends on this turn.
    result = duel.snapshot()
    assert result["winner"] == "Tú"
    assert result["brian"]["steps"] == 4
    assert result["human"]["steps"] == 4
    assert result["result"] == "choque"
    assert set(result["reinforcement_hz"]) == {"aversive"}
    duel.advance(Direction.RIGHT)
    assert duel.snapshot()["turn"] == 4


def test_duel_goal_ends_a_match_without_a_turn_limit(laboratory):
    duel = Duel(laboratory, 53, 2)
    assert laboratory.config.max_steps == 6
    laboratory.game.apple = (0, 0)
    laboratory.game.steps = duel.human.steps = 10_000
    duel.human.apples = 1
    duel.human.apple = (3, 2)
    duel.advance(Direction.UP)
    result = duel.snapshot()
    assert result["winner"] == "Tú"
    assert result["result"] == "meta"
    assert result["apple_goal"] == result["human"]["apples"] == 2
    assert result["brian"]["steps"] == result["human"]["steps"] == 10_001


def test_duel_ties_on_simultaneous_results(laboratory):
    duel = Duel(laboratory, 53, 1)
    laboratory.game.apple = duel.human.apple = (4, 3)
    duel.advance(Direction.RIGHT)
    assert duel.snapshot()["winner"] == "Empate"
    assert duel.snapshot()["result"] == "meta"

    duel = Duel(laboratory, 53, 1)
    laboratory.game.body = [(6, 3), (5, 3)]
    duel.human.body = laboratory.game.body.copy()
    duel.advance(Direction.RIGHT)
    assert duel.snapshot()["winner"] == "Empate"
    assert duel.snapshot()["result"] == "choque"
