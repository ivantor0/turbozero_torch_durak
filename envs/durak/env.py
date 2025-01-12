# envs/durak/env.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from core.env import Env, EnvConfig
from core.utils.utils import rand_argmax_2d

# The same definitions from your durak.py
_NUM_PLAYERS = 2
_NUM_CARDS = 36
_CARDS_PER_PLAYER = 6

# Additional actions:
TAKE_CARDS       = _NUM_CARDS       # 36
FINISH_ATTACK    = _NUM_CARDS + 1   # 37
FINISH_DEFENSE   = _NUM_CARDS + 2   # 38

# Phases
CHANCE     = 0
ATTACK     = 1
DEFENSE    = 2
ADDITIONAL = 3

# Utility for suits & ranks
def suit_of(card: int) -> int:
    return card // 9

def rank_of(card: int) -> int:
    return card % 9

@dataclass
class DurakEnvConfig(EnvConfig):
    # E.g. no extra parameters, we just store env_type and maybe debug
    pass

class DurakEnv(Env):
    """
    A vectorized Durak environment that implements the same logic as your durak.py.
    Each environment in parallel has:
      - a shuffled deck
      - hands for each of the 2 players
      - a set of table cards
      - a discard
      - a trump card/suit
      - a phase
      - an attacker/defender
      - a game_over boolean
    The environment's step() interprets the chosen action, updates state, possibly changes phases, and assigns rewards if terminal.
    """

    def __init__(
        self,
        parallel_envs: int,
        config: DurakEnvConfig,
        device: torch.device,
        debug: bool = False
    ):
        # We'll define input shape in a minimal form, e.g. (channels=1, height=1, width=1).
        # Typically you'd create a more robust encoding for your neural net.
        # For demonstration, let's just do shape (1,1,1).
        # The policy shape is 39 => 36 card-plays + 3 extras.
        # Value shape is (1,).
        state_shape = torch.Size([1, 1, 1])
        policy_shape = torch.Size([_NUM_CARDS + 3])  # 39
        value_shape  = torch.Size([1])

        super().__init__(
            parallel_envs=parallel_envs,
            config=config,
            device=device,
            num_players=_NUM_PLAYERS,
            state_shape=state_shape,
            policy_shape=policy_shape,
            value_shape=value_shape,
            debug=debug
        )

        # We'll store key aspects of the game as GPU tensors:
        # decks: shape [N, 36], each row is a permutation
        self._decks = torch.zeros((parallel_envs, _NUM_CARDS), dtype=torch.long, device=device)
        # For each environment i, _deck_pos[i] is how many cards have been "dealt" so far
        self._deck_pos = torch.zeros(parallel_envs, dtype=torch.long, device=device)
        # For dealing the initial 6+6 = 12 cards
        self._cards_dealt = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        # We track which cards each of the 2 players has: shape [N, 2, 36]. True means card is in that player's hand
        self._hands = torch.zeros((parallel_envs, _NUM_PLAYERS, _NUM_CARDS), dtype=torch.bool, device=device)

        # Table: shape [N, 6, 2], since at most 6 attacking cards can be placed.
        # -1 means empty. table[i, k, 0] = attacking card, table[i, k, 1] = defending card
        self._table_cards = torch.full((parallel_envs, 6, 2), -1, dtype=torch.long, device=device)

        # We'll store a discard mask: shape [N, 36]. True => card is in the discard pile.
        # (In your single-state logic you stored a list, but here we do a boolean mask.)
        self._discard = torch.zeros((parallel_envs, _NUM_CARDS), dtype=torch.bool, device=device)

        # Trump card/suit: shape [N], -1 means unknown
        self._trump_card = torch.full((parallel_envs,), -1, dtype=torch.long, device=device)
        self._trump_suit = torch.full((parallel_envs,), -1, dtype=torch.long, device=device)

        # Who is attacker/defender? shape [N], can be 0 or 1
        self._attacker = torch.zeros(parallel_envs, dtype=torch.long, device=device)
        self._defender = torch.ones(parallel_envs, dtype=torch.long, device=device)

        # Phase: shape [N]. 0=CHANCE, 1=ATTACK, 2=DEFENSE, 3=ADDITIONAL
        self._phase = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        # Who started the round as attacker, used for a special rule if both players run out
        self._round_starter = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        # Is the game over? shape [N], bool
        self._game_over = torch.zeros(parallel_envs, dtype=torch.bool, device=device)

        self.reset()

    def reset(self, seed: Optional[int] = None) -> int:
        """
        Initialize each environment i with a freshly shuffled deck, empty hands, CHANCE phase, etc.
        """
        if seed is not None:
            # for reproducibility, but note that setting the same seed for each environment might
            # cause them to have identical deck permutations unless you offset the seed.
            torch.manual_seed(seed)

        # Shuffle decks individually for each env
        # We'll do this by sampling permutations of size 36 for each environment:
        # A quick approach is to do a random matrix shape [N,36] from e.g. torch.rand, then sort
        rand_vals = torch.rand(self.parallel_envs, _NUM_CARDS, device=self.device)
        sort_idx = torch.argsort(rand_vals, dim=1)
        # sort_idx[i] is the random permutation for environment i
        self._decks = sort_idx.long()

        self._deck_pos.zero_()
        self._cards_dealt.zero_()
        self._hands.zero_()
        self._table_cards.fill_(-1)
        self._discard.zero_()
        self._trump_card.fill_(-1)
        self._trump_suit.fill_(-1)
        self._attacker.zero_()
        self._defender.fill_(1)
        self._phase.fill_(CHANCE)
        self._round_starter.zero_()
        self._game_over.zero_()

        # self.states is from the parent class, used as an input to the neural net if desired:
        self.states.zero_()
        self.terminated.zero_()
        self.cur_players.zero_()  # start with player=0, but we override in current_player() anyway

        return 0 if seed is None else seed

    def is_terminal(self) -> torch.Tensor:
        # Return self._game_over as a boolean
        return self._game_over

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized step. actions is shape [N], each an integer in [0..38].
        We handle each environment i's logic:
         - If the game is over, do nothing.
         - If phase=CHANCE, we proceed to deal or reveal trump.
         - Else interpret the action (playing a card / TAKE_CARDS / FINISH_ATTACK / FINISH_DEFENSE).
         - Then we check if the game is over.
        Return self.terminated (bool [N]).
        """
        # Find active environments
        active = ~self._game_over

        # Process all chance steps first
        self._process_all_chance_steps(active)

        # Now, find environments that are not in CHANCE and are active
        mask_non_chance = (self._phase != CHANCE) & active

        # Find indices
        idxs = torch.nonzero(mask_non_chance, as_tuple=False).flatten()
        if len(idxs) > 0:
            selected_actions = actions[idxs]
            self._apply_actions(idxs, selected_actions)

        # Check if any environment ended up game_over after the move
        self._check_game_over(mask_non_chance)

        # Update termination
        self.terminated = self._game_over

        # Update current players
        self.cur_players = self._current_player_ids()

        return self.terminated

    def _process_all_chance_steps(self, mask: torch.Tensor):
        """
        Repeatedly apply chance logic until no more environments are in CHANCE.
        """
        max_iters = 20  # Safeguard against infinite loops
        for _ in range(max_iters):
            chance_mask = mask & (self._phase == CHANCE) & (~self._game_over)
            if not chance_mask.any():
                break
            self._apply_chance_logic(chance_mask)
        else:
            if self.debug:
                print("Max iterations reached in _process_all_chance_steps")

    def _apply_chance_logic(self, mask: torch.Tensor):
        """
        For each environment i in mask where _phase[i] == CHANCE and not game_over,
        we attempt to do the next dealing step, or reveal trump, per your code.
        If we finish dealing + revealing, we set phase=ATTACK.
        """
        idxs = torch.nonzero(mask & (self._phase == CHANCE), as_tuple=False).flatten()
        if len(idxs) == 0:
            return

        for i in idxs:
            if self._cards_dealt[i] < _CARDS_PER_PLAYER * _NUM_PLAYERS:
                # deal top card to next player
                next_card = self._decks[i, self._deck_pos[i]]
                player_idx = self._cards_dealt[i] % _NUM_PLAYERS
                self._hands[i, player_idx, next_card] = True
                self._deck_pos[i] += 1
                self._cards_dealt[i] += 1
            else:
                # reveal the last card as trump, if not done
                if self._trump_card[i] < 0:
                    # the last card:
                    self._trump_card[i] = self._decks[i, _NUM_CARDS-1]
                    self._trump_suit[i] = suit_of(self._trump_card[i].item())
                    # decide first attacker
                    self._decide_first_attacker(i)
                    self._phase[i] = ATTACK
                    self._round_starter[i] = self._attacker[i]

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return final rewards if game_over, else intermediate 0.0.
        Mimics your returns() logic:
          - If exactly one player has cards, that player gets -1, the other +1.
          - If both have cards or none have cards, handle accordingly.
          - If none have cards and deck is empty, attacker wins.
          - If none have cards but deck isn't empty, treat as 0.0.
        """
        if player_ids is None:
            player_ids = self.cur_players

        # Initialize rewards to 0.0
        rew = torch.zeros((self.parallel_envs,), device=self.device, dtype=torch.float32)

        done = self._game_over
        if not done.any():
            # If none are done => return all zeros
            return rew

        # For envs that are done, replicate your returns() logic:
        # We'll figure out how many cards each player has
        hands_count = self._hands.sum(dim=2)  # shape [N, 2], number of cards for each player
        # players_with_cards is the set of players with >0 cards
        # we can't store a "set," so we'll do a boolean.
        has_cards = (hands_count > 0)  # shape [N,2]
        count_players_with_cards = has_cards.sum(dim=1) # shape [N], 0..2

        # We'll produce a final result for each environment as a pair [r0, r1]. Then we'll pick r = r[player_id].
        # We'll store them in a Nx2 array:
        final_results = torch.zeros((self.parallel_envs, 2), device=self.device)

        # Case 1: exactly 1 player has cards => that is the loser => -1, other => +1
        mask_1 = (count_players_with_cards == 1) & done
        if mask_1.any():
            idxs = torch.nonzero(mask_1, as_tuple=False).flatten()
            for i in idxs:
                # figure out who has cards
                if has_cards[i,0] and not has_cards[i,1]:
                    final_results[i,0] = -1.0
                    final_results[i,1] =  1.0
                elif has_cards[i,1] and not has_cards[i,0]:
                    final_results[i,1] = -1.0
                    final_results[i,0] =  1.0
                else:
                    # theoretically shouldn't happen, but just in case
                    pass

        # Case 2: both have cards => [0,0]
        mask_2 = (count_players_with_cards == 2) & done
        final_results[mask_2, 0] = 0.0
        final_results[mask_2, 1] = 0.0

        # Case 3: none have cards => check if deck is empty => attacker is winner, else [0,0]
        mask_0 = (count_players_with_cards == 0) & done
        if mask_0.any():
            idxs = torch.nonzero(mask_0, as_tuple=False).flatten()
            for i in idxs:
                if self._deck_pos[i] >= _NUM_CARDS:
                    atk = self._attacker[i].item()
                    final_results[i, atk] = 1.0
                    final_results[i, 1 - atk] = -1.0
                else:
                    # 0,0
                    pass

        # Now we pick the perspective of player_ids[i]:
        for i in range(self.parallel_envs):
            if done[i]:
                p = player_ids[i].item()
                rew[i] = final_results[i, p]
            else:
                # not done => 0.0
                rew[i] = 0.0

        return rew

    def get_legal_actions(self) -> torch.Tensor:
        """
        Return a bool [N,39] indicating for each environment which actions are available.
        Replicates your _legal_actions from durak.py, for the current player in each env.
        If game is over or it's chance, then no actions are available.
        """
        legal = torch.zeros((self.parallel_envs, _NUM_CARDS + 3), dtype=torch.bool, device=self.device)

        # If env is done or phase=CHANCE => no legal moves
        done_or_chance = (self._game_over | (self._phase == CHANCE))
        # So we only fill legal actions for the others
        idxs = torch.nonzero(~done_or_chance, as_tuple=False).flatten()

        for i in idxs:
            cur_p = self._current_player_id(i)
            ph    = self._phase[i].item()

            # Gather that player's hand
            # This is a boolean [36], we want to see which card indices are True
            hand_mask = self._hands[i, cur_p]
            hand_cards = torch.nonzero(hand_mask, as_tuple=False).flatten().tolist()

            if ph in (ATTACK, ADDITIONAL) and cur_p == self._attacker[i].item():
                # 1) If table_cards empty => can place any card from hand
                table_occ = self._table_cards[i]
                # table_occ is shape (6,2)
                # how many are used?
                n_placed = self._num_cards_on_table(table_occ)
                if n_placed == 0:
                    # can place any card from hand
                    for c in hand_cards:
                        legal[i, c] = True
                else:
                    # can only place cards if rank matches anything on table
                    # also ensure we haven't exceeded _CARDS_PER_PLAYER or defender’s hand size
                    if n_placed < _CARDS_PER_PLAYER and self._count_hand(i, self._defender[i].item()) > 0:
                        ranks_on_table = set()
                        for pair in table_occ:
                            ac, dc = pair[0].item(), pair[1].item()
                            if ac >= 0:
                                ranks_on_table.add(rank_of(ac))
                            if dc >= 0:
                                ranks_on_table.add(rank_of(dc))
                        for c in hand_cards:
                            if rank_of(c) in ranks_on_table:
                                legal[i, c] = True

                # 2) If there's at least one card on the table => allow FINISH_ATTACK
                if n_placed > 0:
                    legal[i, FINISH_ATTACK] = True

            elif ph == DEFENSE and cur_p == self._defender[i].item():
                # The defender can TAKE_CARDS always
                legal[i, TAKE_CARDS] = True
                # If everything is covered => can FINISH_DEFENSE
                if self._all_covered(self._table_cards[i]):
                    legal[i, FINISH_DEFENSE] = True
                else:
                    # The earliest uncovered card => see if we can cover it
                    row, att_card = self._find_earliest_uncovered(self._table_cards[i])
                    if row != -1 and att_card >= 0:
                        # For each c in hand, check if can defend
                        for c in hand_cards:
                            if self._can_defend_card(i, c, att_card):
                                legal[i, c] = True

        return legal

    def next_turn(self):
        """
        Called by MCTS expansions. We do nothing because our step method already handles switching phases, etc.
        """
        pass

    def save_node(self):
        """
        Return enough data so we can restore it exactly in load_node.
        We'll store all the big fields.
        For performance, we often do partial copying only for those envs that we want to preserve,
        but for clarity we just store them all as .clone().
        """
        return (
            self._decks.clone(),
            self._deck_pos.clone(),
            self._cards_dealt.clone(),
            self._hands.clone(),
            self._table_cards.clone(),
            self._discard.clone(),
            self._trump_card.clone(),
            self._trump_suit.clone(),
            self._attacker.clone(),
            self._defender.clone(),
            self._phase.clone(),
            self._round_starter.clone(),
            self._game_over.clone()
        )

    def load_node(self, load_envs: torch.Tensor, saved):
        """
        load_envs is [N] bool, indicating which envs to restore. We restore from the saved data.
        """
        (d, dp, cd, h, tc, di, trc, trs, att, defn, ph, rs, go) = saved
        idxs = torch.nonzero(load_envs, as_tuple=False).flatten()
        for i in idxs:
            self._decks[i]        = d[i]
            self._deck_pos[i]     = dp[i]
            self._cards_dealt[i]  = cd[i]
            self._hands[i]        = h[i]
            self._table_cards[i]  = tc[i]
            self._discard[i]      = di[i]
            self._trump_card[i]   = trc[i]
            self._trump_suit[i]   = trs[i]
            self._attacker[i]     = att[i]
            self._defender[i]     = defn[i]
            self._phase[i]        = ph[i]
            self._round_starter[i]= rs[i]
            self._game_over[i]    = go[i]

        self.update_terminated()

    def reset_terminated_states(self, seed: Optional[int] = None):
        """
        For each environment i where game is over, we re-init that environment if needed.
        """
        if seed is not None:
            torch.manual_seed(seed)
        done = self._game_over
        if not done.any():
            return  # nothing to do

        # re-init those envs
        idxs = torch.nonzero(done, as_tuple=False).flatten()
        for i in idxs:
            # shuffle deck
            perm = torch.argsort(torch.rand(_NUM_CARDS, device=self.device))
            self._decks[i] = perm
            self._deck_pos[i] = 0
            self._cards_dealt[i] = 0
            self._hands[i].fill_(False)
            self._table_cards[i].fill_(-1)
            self._discard[i].fill_(False)
            self._trump_card[i] = -1
            self._trump_suit[i] = -1
            self._attacker[i] = 0
            self._defender[i] = 1
            self._phase[i] = CHANCE
            self._round_starter[i] = 0
            self._game_over[i] = False

        self.update_terminated()

    def print_state(self, last_action: Optional[int] = None):
        """
        For debugging a single environment i=0
        """
        i = 0
        print(f"--- [DurakEnv] environment {i} ---")
        print(f"phase={self._phase[i].item()}, attacker={self._attacker[i].item()}, defender={self._defender[i].item()}")
        print(f"deck_pos={self._deck_pos[i].item()}, cards_dealt={self._cards_dealt[i].item()}")
        print(f"trump_card={self._trump_card[i].item()} (suit={self._trump_suit[i].item()})")
        # list out the hand for each player:
        for p in range(_NUM_PLAYERS):
            cidxs = torch.nonzero(self._hands[i,p], as_tuple=False).flatten().tolist()
            print(f"  Player {p} hand = {cidxs}")
        # table
        tab = self._table_cards[i].cpu().numpy().tolist()
        print(f"  Table: {tab}")
        # discard
        disc_cards = torch.nonzero(self._discard[i], as_tuple=False).flatten().tolist()
        print(f"  Discard: {disc_cards}")
        if last_action is not None:
            print(f"  last_action={last_action}")

    # --------------------------------------------------------------------------
    # Internal logic

    def _apply_actions(self, idxs: torch.Tensor, actions: torch.Tensor):
        """
        For each environment i in idxs, interpret the chosen action.
        """
        for k, i in enumerate(idxs):
            act = actions[k].item()
            if self._game_over[i]:
                continue
            p = self._current_player_id(i)
            ph = self._phase[i].item()

            if act >= _NUM_CARDS:
                # extra action
                if act == TAKE_CARDS:
                    self._defender_takes_cards(i)
                elif act == FINISH_ATTACK:
                    self._attacker_finishes_attack(i)
                elif act == FINISH_DEFENSE:
                    self._defender_finishes_defense(i)
            else:
                # normal card
                # check if in player's hand
                if self._hands[i, p, act]:
                    if ph in [ATTACK, ADDITIONAL] and p == self._attacker[i].item():
                        # place attacking card
                        self._hands[i, p, act] = False
                        # put on first free row in table_cards if we are continuing ATTACK
                        row = self._find_first_empty_attack_row(i)
                        if row != -1:
                            self._table_cards[i, row, 0] = act
                        # remain in ATTACK phase
                        self._phase[i] = ATTACK
                    elif ph == DEFENSE and p == self._defender[i].item():
                        # defend earliest uncovered
                        row, att_card = self._find_earliest_uncovered(self._table_cards[i])
                        if row != -1 and att_card >= 0:
                            # check if can defend
                            if self._can_defend_card(i, act, att_card):
                                self._hands[i, p, act] = False
                                self._table_cards[i, row, 1] = act
                                # if all covered => ADDITIONAL
                                if self._all_covered(self._table_cards[i]):
                                    self._phase[i] = ADDITIONAL

    def _check_game_over(self, mask: torch.Tensor):
        """
        For each environment i in mask, replicate your _check_game_over from durak.py.
        """
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        for i in idxs:
            if self._game_over[i]:
                continue
            # 1) if a player has no cards + deck is empty => game over
            # or if both have no cards + deck is empty => game over
            c0 = self._count_hand(i, 0)
            c1 = self._count_hand(i, 1)
            decksize = _NUM_CARDS - self._deck_pos[i].item()
            if (c0 == 0 or c1 == 0) and decksize <= 0:
                self._game_over[i] = True
                continue
            # 2) if both players have no cards
            if c0 == 0 and c1 == 0:
                # if deck is empty => game over, else refill
                if decksize <= 0:
                    self._game_over[i] = True
                else:
                    # refill
                    self._refill_hands(i)

    # Mirroring your logic from durak.py:

    def _defender_takes_cards(self, i: int):
        """
        If the defender picks up all table cards, same attacker remains, table cleared, refill.
        """
        if self._game_over[i]:
            return
        def_p = self._defender[i].item()
        table = self._table_cards[i]  # shape (6,2)
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0:
                self._hands[i, def_p, ac] = True
            if dc >= 0:
                self._hands[i, def_p, dc] = True
        self._table_cards[i].fill_(-1)
        self._phase[i] = ATTACK
        self._refill_hands(i)

    def _attacker_finishes_attack(self, i: int):
        """
        If no table cards => do nothing. Else phase=DEFENSE
        """
        table = self._table_cards[i]
        if self._num_cards_on_table(table) == 0:
            return
        self._phase[i] = DEFENSE

    def _defender_finishes_defense(self, i: int):
        """
        If uncovered => TAKE_CARDS. Else discard them all, swap roles, refill, phase=ATTACK.
        """
        if not self._all_covered(self._table_cards[i]):
            self._defender_takes_cards(i)
        else:
            # discard
            table = self._table_cards[i]
            for row in range(6):
                ac, dc = table[row,0].item(), table[row,1].item()
                if ac >= 0:
                    self._discard[i, ac] = True
                if dc >= 0:
                    self._discard[i, dc] = True
            self._table_cards[i].fill_(-1)
            # swap roles
            old_att = self._attacker[i].item()
            old_def = self._defender[i].item()
            self._attacker[i] = old_def
            self._defender[i] = old_att
            self._refill_hands(i)
            self._phase[i] = ATTACK

    def _refill_hands(self, i: int):
        """
        Refill attacker first, then defender, up to 6 cards each.
        The last card in the deck is the trump. We keep it in place. If deck_pos < 35 we can still draw.
        """
        if self._game_over[i]:
            return
        order = [self._attacker[i].item(), self._defender[i].item()]

        while self._deck_pos[i] < _NUM_CARDS -1:
            for p in order:
                if self._count_hand(i, p) < _CARDS_PER_PLAYER and self._deck_pos[i] < _NUM_CARDS -1:
                    # draw from top (except the very last card is the trump face up)
                    c = self._decks[i, self._deck_pos[i]].item()
                    self._deck_pos[i] += 1
                    self._hands[i, p, c] = True
            # if both full, break
            if self._count_hand(i, order[0]) >= _CARDS_PER_PLAYER and self._count_hand(i, order[1]) >= _CARDS_PER_PLAYER:
                break

    def _decide_first_attacker(self, i: int):
        """
        Find the lowest trump card among each player's 6.
        That player => attacker.
        """
        if self._trump_card[i] < 0:
            return
        tsuit = self._trump_suit[i].item()
        best = 999
        who = 0
        for p in range(_NUM_PLAYERS):
            # find min rank of trump in p's hand
            hand_cards = torch.nonzero(self._hands[i,p], as_tuple=False).flatten().tolist()
            for c in hand_cards:
                if suit_of(c) == tsuit:
                    r = rank_of(c)
                    if r < best:
                        best = r
                        who = p
        self._attacker[i] = who
        self._defender[i] = 1 - who

    # Some small helpers:
    def _current_player_ids(self) -> torch.Tensor:
        """
        Return an [N] of current player IDs based on self._phase.
        If phase in [ATTACK, ADDITIONAL], current=attacker, else if DEFENSE => defender, else CHANCE => 0.
        """
        cp = torch.full((self.parallel_envs,), 0, dtype=torch.long, device=self.device)  # default to 0 for CHANCE
        mask_attack = (self._phase == ATTACK) | (self._phase == ADDITIONAL)
        cp[mask_attack] = self._attacker[mask_attack]
        mask_def = (self._phase == DEFENSE)
        cp[mask_def] = self._defender[mask_def]
        return cp

    def _current_player_id(self, i: int) -> int:
        """
        Single environment version.
        """
        ph = self._phase[i].item()
        if ph in (ATTACK, ADDITIONAL):
            return self._attacker[i].item()
        elif ph == DEFENSE:
            return self._defender[i].item()
        else:
            # CHANCE => pick 0
            return 0

    def _can_defend_card(self, i: int, defense_card: int, attack_card: int) -> bool:
        """
        Mirroring your _can_defend_card from durak.py
        """
        if self._trump_suit[i] < 0:
            return False
        att_s, att_r = suit_of(attack_card), rank_of(attack_card)
        def_s, def_r = suit_of(defense_card), rank_of(defense_card)
        tsuit = self._trump_suit[i].item()

        # same suit, higher rank
        if att_s == def_s and def_r > att_r:
            return True
        # defend with trump if attack is not trump
        if def_s == tsuit and att_s != tsuit:
            return True
        # if both trump, rank must be higher
        if att_s == tsuit and def_s == tsuit and def_r > att_r:
            return True
        return False

    def _find_earliest_uncovered(self, table: torch.Tensor) -> Tuple[int,int]:
        """
        table is shape (6,2). Return (row, attacking_card).
        If no uncovered, return (-1, -1).
        The earliest uncovered is the first row where table[row,1] == -1 and table[row,0] != -1
        """
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0 and dc < 0:
                return (row, ac)
        return (-1, -1)

    def _num_cards_on_table(self, table: torch.Tensor) -> int:
        """
        Return how many attacking cards have been placed.
        That is how many rows have table[row,0] >= 0
        """
        return (table[:,0] >= 0).sum().item()

    def _all_covered(self, table: torch.Tensor) -> bool:
        """
        Return True if every row that has an attacking card also has a defending card.
        """
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0 and dc < 0:
                return False
        return True

    def _find_first_empty_attack_row(self, i: int) -> int:
        """
        Return the first row index in table_cards[i] that has table[row,0] == -1
        """
        table = self._table_cards[i]
        for row in range(6):
            if table[row,0] < 0:
                return row
        return -1

    def _count_hand(self, i: int, p: int) -> int:
        """
        Return how many cards player p has in environment i
        """
        return int(self._hands[i,p].sum().item())
