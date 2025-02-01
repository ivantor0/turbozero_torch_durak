# envs/durak/env.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from core.env import Env, EnvConfig

# Card definitions
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

# A fallback to avoid infinite random loops. If environment takes more steps than this, we forcibly end the game.
_MAX_STEPS = 300

# Suits/ranks for printing:
_suit_symbols = ["♠", "♣", "♦", "♥"]
_rank_symbols = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]

def suit_of(card: int) -> int:
    return card // 9

def rank_of(card: int) -> int:
    return card % 9

def card_to_string(card_idx: int) -> str:
    """
    Convert card index [0..35] into a string like 6♠, 7♣, 10♦, J♥, etc.
    """
    if card_idx < 0 or card_idx >= _NUM_CARDS:
        return "???"
    s = suit_of(card_idx)
    r = rank_of(card_idx)
    return f"{_rank_symbols[r]}{_suit_symbols[s]}"

@dataclass
class DurakEnvConfig(EnvConfig):
    pass

class DurakEnv(Env):
    """
    A vectorized Durak environment.
    """

    def __init__(
        self,
        parallel_envs: int,
        config: DurakEnvConfig,
        device: torch.device,
        debug: bool = False
    ):
        # Instead of a minimal (1,1,1) input, we now build a full state representation that is 93-dimensional.
        # The state vector consists of:
        #  - current player's hand (36)
        #  - table cards (flattened 6x2 = 12)
        #  - discard mask (36)
        #  - trump suit as one-hot (4)
        #  - phase as one-hot (4)
        #  - attacker indicator (1)
        # Total = 36+12+36+4+4+1 = 93.
        state_shape  = torch.Size([93, 1, 1])
        policy_shape = torch.Size([_NUM_CARDS + 3])  # 39 possible actions
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

        # Deck permutations: [N,36]
        self._decks = torch.zeros((parallel_envs, _NUM_CARDS), dtype=torch.long, device=device)

        # deck_pos: how many cards have effectively been drawn from the top
        self._deck_pos = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        # For dealing the initial 12 cards
        self._cards_dealt = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        # Hands: shape [N,2,36]
        # True => that player holds that card
        self._hands = torch.zeros((parallel_envs, _NUM_PLAYERS, _NUM_CARDS), dtype=torch.bool, device=device)

        # Table: shape [N,6,2]. Each row is (att_card, def_card), or -1 if unused
        self._table_cards = torch.full((parallel_envs, 6, 2), -1, dtype=torch.long, device=device)

        # Discard mask: [N,36], True => that card is in the discard
        self._discard = torch.zeros((parallel_envs, _NUM_CARDS), dtype=torch.bool, device=device)

        # Trump
        self._trump_card = torch.full((parallel_envs,), -1, dtype=torch.long, device=device)
        self._trump_suit = torch.full((parallel_envs,), -1, dtype=torch.long, device=device)

        # Roles
        self._attacker = torch.zeros(parallel_envs, dtype=torch.long, device=device)
        self._defender = torch.ones(parallel_envs, dtype=torch.long, device=device)

        # Phase
        self._phase = torch.zeros(parallel_envs, dtype=torch.long, device=device)
        # Round starter
        self._round_starter = torch.zeros(parallel_envs, dtype=torch.long, device=device)
        # Game over
        self._game_over = torch.zeros(parallel_envs, dtype=torch.bool, device=device)

        # Step counters to avoid infinite loops:
        self._step_count = torch.zeros(parallel_envs, dtype=torch.long, device=device)

        self.reset()

    def get_nn_input(self) -> torch.Tensor:
        """
        Construct the neural network input from the internal (private) state.
        Only the current player's hand is visible; opponent's hand is masked out.
        We also include public information: table cards, discard pile, trump suit,
        current phase, and an indicator of whether the current player is the attacker.
        The output is a tensor of shape (parallel_envs, 93) which is then reshaped to (93,1,1).
        """
        # Determine current player for each environment (0 or 1)
        cp = self.cur_players  # shape (parallel_envs,)
        idx = torch.arange(self.parallel_envs, device=self.device)
        # current player's hand (as float)
        hands = self._hands[idx, cp].float()  # shape (parallel_envs, 36)
        # table cards (all public, convert -1 to 0 for "empty" then leave card indices as is normalized by _NUM_CARDS)
        table = self._table_cards.float().view(self.parallel_envs, -1)  # shape (parallel_envs, 12)
        # discard mask (binary vector)
        discard = self._discard.float()  # shape (parallel_envs, 36)
        # trump suit as one-hot vector (if trump not revealed, all zeros)
        trump = torch.zeros((self.parallel_envs, 4), device=self.device)
        valid_trump = self._trump_suit >= 0
        indices = self._trump_suit.clone()
        indices[~valid_trump] = 0
        trump[idx, indices] = valid_trump.float()
        # phase as one-hot (4)
        phase = torch.zeros((self.parallel_envs, 4), device=self.device)
        phase[idx, self._phase] = 1.0
        # Attacker indicator: 1 if current player is the attacker, 0 otherwise.
        attacker_indicator = (self._attacker == cp).float().unsqueeze(1)  # shape (parallel_envs, 1)
        # Concatenate all features along dimension 1 (total length = 36+12+36+4+4+1 = 93)
        state_vec = torch.cat([hands, table, discard, trump, phase, attacker_indicator], dim=1)
        # Reshape to (parallel_envs, 93, 1, 1) to match state_shape.
        return state_vec.view(self.parallel_envs, 93, 1, 1)


    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed(seed)

        # Shuffle decks
        rand_vals = torch.rand(self.parallel_envs, _NUM_CARDS, device=self.device)
        sort_idx  = torch.argsort(rand_vals, dim=1)
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
        self._step_count.zero_()

        # Clear parent tracking
        self.states.zero_()
        self.terminated.zero_()
        self.cur_players.zero_()

        # Immediately handle dealing and trump reveal
        self._process_all_chance_steps(~self._game_over)

        return 0 if seed is None else seed

    def is_terminal(self) -> torch.Tensor:
        return self._game_over

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        # Mark step counts
        self._step_count += 1

        # Force-terminate any environment that hits _MAX_STEPS
        overlimit = (self._step_count >= _MAX_STEPS) & (~self._game_over)
        self._game_over |= overlimit

        active = ~self._game_over
        mask_non_chance = (self._phase != CHANCE) & active

        idxs = torch.nonzero(mask_non_chance, as_tuple=False).flatten()
        if len(idxs) > 0:
            selected_actions = actions[idxs]
            self._apply_actions(idxs, selected_actions)

        self._check_game_over(mask_non_chance)

        # Update termination flags
        self.terminated = self._game_over
        self.cur_players = self._current_player_ids()

        return self.terminated

    # -------------------- CHANCE HANDLING --------------------
    def _process_all_chance_steps(self, mask: torch.Tensor):
        """
        Repeatedly apply dealing logic until no environment is in CHANCE.
        """
        for _ in range(40):  # safeguard
            chance_mask = mask & (self._phase == CHANCE) & (~self._game_over)
            if not chance_mask.any():
                break
            self._apply_chance_logic(chance_mask)

    def _apply_chance_logic(self, mask: torch.Tensor):
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        for i in idxs:
            if self._cards_dealt[i] < _CARDS_PER_PLAYER * _NUM_PLAYERS:
                # Deal next card
                next_card = self._decks[i, self._deck_pos[i]]
                player_idx = self._cards_dealt[i] % _NUM_PLAYERS
                self._hands[i, player_idx, next_card] = True
                self._deck_pos[i] += 1
                self._cards_dealt[i] += 1
            else:
                # Reveal the last card as trump if not done
                if self._trump_card[i] < 0:
                    self._trump_card[i] = self._decks[i, _NUM_CARDS - 1]
                    tsuit = suit_of(self._trump_card[i].item())
                    self._trump_suit[i] = tsuit
                    self._decide_first_attacker(i)
                    self._round_starter[i] = self._attacker[i]
                    self._phase[i] = ATTACK

    # -------------------- REWARD LOGIC --------------------
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return final +1/-1 from the perspective of each environment's current player if game is done with a winner/loser,
        else 0 for ongoing or draws/timeouts.
        """
        if player_ids is None:
            player_ids = self.cur_players

        rew = torch.zeros((self.parallel_envs,), device=self.device, dtype=torch.float32)
        done = self._game_over
        if not done.any():
            return rew  # no environment ended, so 0 reward

        # For each done environment, decide final result from vantage of player_ids[i].
        # We'll replicate the single-state logic: if exactly one player has cards => that player is loser => -1.
        # else if a single player is out => that player is winner => +1, etc.
        # If we forced a time-out or a scenario with 2 players with cards => 0.
        hands_count = self._hands.sum(dim=2)  # [N,2], how many cards each player holds
        has_cards = (hands_count > 0)
        count_with_cards = has_cards.sum(dim=1)

        # We'll build final_results Nx2: final_results[i, p] is that player's payoff in environment i.
        final_results = torch.zeros((self.parallel_envs, 2), device=self.device)

        idxs = torch.nonzero(done, as_tuple=False).flatten()
        for i in idxs:
            c0 = hands_count[i,0].item()
            c1 = hands_count[i,1].item()
            deck_rem = _NUM_CARDS - self._deck_pos[i].item()
            # Check forced timeout
            if self._step_count[i].item() >= _MAX_STEPS:
                # We will penalize stalling
                final_results[i, :] = -1.0
                continue

            # Evaluate normal end
            if (c0 == 0 and c1 == 0):
                # both out => if deck empty => attacker=winner => +1, else draw=0
                if deck_rem <= 0:
                    atk = self._attacker[i].item()
                    final_results[i, atk] =  1.0
                    final_results[i, 1 - atk] = -1.0
                else:
                    # draw
                    final_results[i, :] = 0.0
            elif c0 == 0 and c1 > 0:
                # Player0 is out => p0 is winner => +1, p1 = -1
                final_results[i,0] =  1.0
                final_results[i,1] = -1.0
            elif c1 == 0 and c0 > 0:
                # Player1 is out => p1 is winner => +1, p0 = -1
                final_results[i,1] =  1.0
                final_results[i,0] = -1.0
            elif (c0 > 0 and c1 > 0 and deck_rem <= 0):
                # both have cards and no deck => scenario might happen => treat as 0
                final_results[i, :] = 0.0
            else:
                # Any other done scenario => treat as draw => 0
                final_results[i, :] = 0.0

        # Now pick the perspective of player_ids[i]
        for i in range(self.parallel_envs):
            if done[i]:
                p = player_ids[i].item()
                # If the environment ended while it was chance or after the final move,
                # we fallback to environment's "current" perspective (which is attacker or defender).
                # If p == -1 or 0 in chance, or we do a safe clamp:
                p = max(0, min(1, p))
                rew[i] = final_results[i, p]
            else:
                rew[i] = 0.0

        return rew

    # -------------------- LEGAL ACTIONS --------------------
    def get_legal_actions(self) -> torch.Tensor:
        """
        Return a [N,39] bool mask of valid actions
        """
        legal = torch.zeros((self.parallel_envs, _NUM_CARDS+3), dtype=torch.bool, device=self.device)

        done_or_chance = (self._game_over | (self._phase == CHANCE))
        idxs = torch.nonzero(~done_or_chance, as_tuple=False).flatten()

        for i in idxs:
            ph = self._phase[i].item()
            cur_p = self._current_player_id(i)
            hand_mask = self._hands[i, cur_p]  # shape [36], True if in hand
            hand_cards = torch.nonzero(hand_mask, as_tuple=False).flatten().tolist()

            if ph in (ATTACK, ADDITIONAL) and (cur_p == self._attacker[i].item()):
                table_occ = self._table_cards[i]
                n_placed = self._num_cards_on_table(table_occ)
                if n_placed == 0:
                    for c in hand_cards:
                        legal[i, c] = True
                else:
                    # can place only if rank in table, and not exceeding _CARDS_PER_PLAYER or defender's hand size
                    if n_placed < _CARDS_PER_PLAYER and self._count_hand(i, self._defender[i].item()) > 0:
                        ranks_on_table = set()
                        for row in range(6):
                            ac, dc = table_occ[row,0].item(), table_occ[row,1].item()
                            if ac >= 0:
                                ranks_on_table.add(rank_of(ac))
                            if dc >= 0:
                                ranks_on_table.add(rank_of(dc))
                        for c in hand_cards:
                            if rank_of(c) in ranks_on_table:
                                legal[i, c] = True
                if n_placed > 0:
                    legal[i, FINISH_ATTACK] = True

            elif ph == DEFENSE and (cur_p == self._defender[i].item()):
                if self._all_covered(self._table_cards[i]):
                    legal[i, FINISH_DEFENSE] = True
                else:
                    legal[i, TAKE_CARDS] = True
                    row, attc = self._find_earliest_uncovered(self._table_cards[i])
                    if row != -1 and attc >= 0:
                        for c in hand_cards:
                            if self._can_defend_card(i, c, attc):
                                legal[i, c] = True

        return legal

    # -------------------- STEP LOGIC --------------------
    def _apply_actions(self, idxs: torch.Tensor, actions: torch.Tensor):
        for k, i in enumerate(idxs):
            if self._game_over[i]:
                continue
            ph = self._phase[i].item()
            p  = self._current_player_id(i)
            act= actions[k].item()

            if act >= _NUM_CARDS:
                # Special action
                if act == TAKE_CARDS:
                    self._defender_takes_cards(i)
                elif act == FINISH_ATTACK:
                    self._attacker_finishes_attack(i)
                elif act == FINISH_DEFENSE:
                    self._defender_finishes_defense(i)
            else:
                # playing a card from hand
                if self._hands[i, p, act]:
                    if ph in (ATTACK, ADDITIONAL) and p == self._attacker[i].item():
                        # Attacker places a card
                        self._hands[i, p, act] = False
                        row = self._find_first_empty_attack_row(i)
                        if row != -1:
                            self._table_cards[i, row, 0] = act
                        self._phase[i] = ATTACK

                    elif ph == DEFENSE and p == self._defender[i].item():
                        row, att_card = self._find_earliest_uncovered(self._table_cards[i])
                        if row != -1 and att_card >= 0:
                            if self._can_defend_card(i, act, att_card):
                                self._hands[i, p, act] = False
                                self._table_cards[i, row, 1] = act
                                if self._all_covered(self._table_cards[i]):
                                    self._phase[i] = ADDITIONAL


    def _check_game_over(self, mask: torch.Tensor):
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        for i in idxs:
            if self._game_over[i]:
                continue

            # 1) If either has 0 cards AND deck is empty => done
            c0 = self._count_hand(i, 0)
            c1 = self._count_hand(i, 1)
            deck_rem = _NUM_CARDS - self._deck_pos[i].item()

            if (c0 == 0 or c1 == 0) and deck_rem == 0:
                self._game_over[i] = True
                continue

            # 2) if both have 0 => if deck empty => done, else refill
            if c0 == 0 and c1 == 0:
                if deck_rem == 0:
                    self._game_over[i] = True
                else:
                    self._refill_hands(i)

    # -------------- Implementation of sub-logic --------------
    def _defender_takes_cards(self, i: int):
        if self._game_over[i]:
            return
        def_p = self._defender[i].item()
        table = self._table_cards[i]
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0:
                self._hands[i, def_p, ac] = True
            if dc >= 0:
                self._hands[i, def_p, dc] = True
        table.fill_(-1)
        self._phase[i] = ATTACK
        self._refill_hands(i)

    def _attacker_finishes_attack(self, i: int):
        table = self._table_cards[i]
        if self._num_cards_on_table(table) == 0:
            return
        self._phase[i] = DEFENSE

    def _defender_finishes_defense(self, i: int):
        table = self._table_cards[i]
        if not self._all_covered(table):
            self._defender_takes_cards(i)
        else:
            # discard covered cards
            for row in range(6):
                ac, dc = table[row,0].item(), table[row,1].item()
                if ac >= 0:
                    self._discard[i, ac] = True
                if dc >= 0:
                    self._discard[i, dc] = True
            table.fill_(-1)
            # swap roles
            old_att = self._attacker[i].item()
            old_def = self._defender[i].item()
            self._attacker[i] = old_def
            self._defender[i] = old_att
            self._refill_hands(i)
            self._phase[i] = ATTACK

    def _refill_hands(self, i: int):
        if self._game_over[i]:
            return
        order = [self._attacker[i].item(), self._defender[i].item()]
        while self._deck_pos[i] < _NUM_CARDS:
            for p in order:
                if self._count_hand(i, p) < _CARDS_PER_PLAYER and (self._deck_pos[i] < _NUM_CARDS):
                    c = self._decks[i, self._deck_pos[i]].item()
                    self._deck_pos[i] += 1
                    self._hands[i, p, c] = True
            # If both are full, break
            if (self._count_hand(i, order[0]) >= _CARDS_PER_PLAYER and
                self._count_hand(i, order[1]) >= _CARDS_PER_PLAYER):
                break

    def _decide_first_attacker(self, i: int):
        if self._trump_card[i] < 0:
            return
        tsuit = self._trump_suit[i].item()
        best = 999
        who = 0
        for p in range(_NUM_PLAYERS):
            hand_cards = torch.nonzero(self._hands[i,p], as_tuple=False).flatten().tolist()
            for c in hand_cards:
                if suit_of(c) == tsuit:
                    r = rank_of(c)
                    if r < best:
                        best = r
                        who = p
        self._attacker[i] = who
        self._defender[i] = 1 - who

    # -------------- Helpers --------------
    def _current_player_ids(self) -> torch.Tensor:
        cp = torch.zeros((self.parallel_envs,), dtype=torch.long, device=self.device)
        mask_att = (self._phase == ATTACK) | (self._phase == ADDITIONAL)
        cp[mask_att] = self._attacker[mask_att]
        mask_def = (self._phase == DEFENSE)
        cp[mask_def] = self._defender[mask_def]
        # if CHANCE or game_over => 0 by default
        return cp

    def _current_player_id(self, i: int) -> int:
        ph = self._phase[i].item()
        if ph in (ATTACK, ADDITIONAL):
            return self._attacker[i].item()
        elif ph == DEFENSE:
            return self._defender[i].item()
        else:
            return 0  # CHANCE or done => default to 0

    def _can_defend_card(self, i: int, defense_card: int, attack_card: int) -> bool:
        if self._trump_suit[i] < 0:
            return False
        att_s, att_r = suit_of(attack_card), rank_of(attack_card)
        def_s, def_r = suit_of(defense_card), rank_of(defense_card)
        ts = self._trump_suit[i].item()
        # same suit, higher rank
        if att_s == def_s and def_r > att_r:
            return True
        # if attacker not trump, but defender is trump
        if (att_s != ts) and (def_s == ts):
            return True
        # if both trump, rank must be higher
        if (att_s == ts) and (def_s == ts) and (def_r > att_r):
            return True
        return False

    def _find_earliest_uncovered(self, table: torch.Tensor) -> Tuple[int,int]:
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0 and dc < 0:
                return (row, ac)
        return (-1, -1)

    def _num_cards_on_table(self, table: torch.Tensor) -> int:
        return (table[:,0] >= 0).sum().item()

    def _all_covered(self, table: torch.Tensor) -> bool:
        for row in range(6):
            ac, dc = table[row,0].item(), table[row,1].item()
            if ac >= 0 and dc < 0:
                return False
        return True

    def _find_first_empty_attack_row(self, i: int) -> int:
        t = self._table_cards[i]
        for row in range(6):
            if t[row,0] < 0:
                return row
        return -1

    def _count_hand(self, i: int, p: int) -> int:
        return int(self._hands[i,p].sum().item())

    # -------------------- SAVE/LOAD NODES --------------------
    def save_node(self):
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
            self._game_over.clone(),
            self._step_count.clone()
        )

    def load_node(self, load_envs: torch.Tensor, saved):
        (d, dp, cd, h, tc, disc, trc, trs, att, defn, ph, rs, go, st) = saved
        idxs = torch.nonzero(load_envs, as_tuple=False).flatten()
        for i in idxs:
            self._decks[i]         = d[i]
            self._deck_pos[i]      = dp[i]
            self._cards_dealt[i]   = cd[i]
            self._hands[i]         = h[i]
            self._table_cards[i]   = tc[i]
            self._discard[i]       = disc[i]
            self._trump_card[i]    = trc[i]
            self._trump_suit[i]    = trs[i]
            self._attacker[i]      = att[i]
            self._defender[i]      = defn[i]
            self._phase[i]         = ph[i]
            self._round_starter[i] = rs[i]
            self._game_over[i]     = go[i]
            self._step_count[i]    = st[i]
        self.update_terminated()

    def reset_terminated_states(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        done = self._game_over
        if not done.any():
            return
        idxs = torch.nonzero(done, as_tuple=False).flatten()
        for i in idxs:
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
            self._step_count[i] = 0

        self.update_terminated()

    # -------------------- Debug Printing --------------------
    def print_state(self, last_action: Optional[int] = None) -> None:
        """
        For debugging environment #0
        """
        i = 0
        ph    = self._phase[i].item()
        atk   = self._attacker[i].item()
        dfd   = self._defender[i].item()
        tcard = self._trump_card[i].item()
        tsuit = self._trump_suit[i].item()
        sc    = self._step_count[i].item()

        print(f"--- [DurakEnv] environment {i} step_count={sc} ---")
        print(f"phase={ph}, attacker={atk+1}, defender={dfd+1}")
        print(f"deck_pos={self._deck_pos[i].item()}, cards_dealt={self._cards_dealt[i].item()}")
        if tcard >= 0:
            print(f"trump_card={card_to_string(tcard)} (suit={tsuit})")
        else:
            print("trump_card=???")
        # Hands
        for p in range(_NUM_PLAYERS):
            card_idxs = torch.nonzero(self._hands[i, p], as_tuple=False).flatten().tolist()
            card_strs = [card_to_string(c) for c in card_idxs]
            print(f"  Player {p+1} hand = {card_strs}")
        # Table
        trows = self._table_cards[i].cpu().numpy()
        table_strs = []
        for row in trows:
            ac, dc = row[0], row[1]
            if ac < 0:
                table_strs.append("[-,-]")
            else:
                a_str = card_to_string(ac)
                if dc < 0:
                    table_strs.append(f"[{a_str},?]")
                else:
                    d_str = card_to_string(dc)
                    table_strs.append(f"[{a_str}, {d_str}]")
        print("  Table: " + ", ".join(table_strs))

        # Discard
        disc_idxs = torch.nonzero(self._discard[i], as_tuple=False).flatten().tolist()
        disc_strs = [card_to_string(c) for c in disc_idxs]
        print(f"  Discard: {disc_strs}")

        if last_action is not None:
            if last_action < _NUM_CARDS:
                la_str = card_to_string(last_action)
            elif last_action == TAKE_CARDS:
                la_str = "TAKE_CARDS"
            elif last_action == FINISH_ATTACK:
                la_str = "FINISH_ATTACK"
            elif last_action == FINISH_DEFENSE:
                la_str = "FINISH_DEFENSE"
            else:
                la_str = f"??? ({last_action})"
            print(f"  last_action={la_str}\n")
