# envs/durak/demo.py

import random
import torch
from core.demo.demo import TwoPlayerDemo
from IPython.display import clear_output
import os

class DurakDemo(TwoPlayerDemo):
    def __init__(self,
        evaluator,
        evaluator2,
        manual_step: bool = False
    ) -> None:
        super().__init__(evaluator, evaluator2, manual_step)
        # No additional assertions needed for durak

    def run(self, print_evaluation: bool = False, print_state: bool = True, interactive: bool = True):
        seed = random.randint(0, 2**32 - 1)
        
        self.evaluator.reset(seed)
        self.evaluator2.reset(seed)
        # Randomly choose which evaluator starts; note that internally players are 0 and 1,
        # but we display them as 1 and 2.
        p1_turn = random.choice([True, False])
        cur_player = self.evaluator.env.cur_players.item()
        # p1_player_id will be shown as 1 or 2
        p1_player_id = (cur_player if p1_turn else 1 - cur_player) + 1
        p1_evaluation = 0.5
        p2_evaluation = 0.5
        actions = None
        while True:
            active_evaluator = self.evaluator if p1_turn else self.evaluator2
            other_evaluator = self.evaluator2 if p1_turn else self.evaluator
            if print_state:
                print(f'Player 1: {self.evaluator.__class__.__name__ if p1_player_id == 1 else self.evaluator2.__class__.__name__}')
                print(f'Player 2: {self.evaluator.__class__.__name__ if p1_player_id == 2 else self.evaluator2.__class__.__name__}')
                active_evaluator.env.print_state(actions.item() if actions is not None else None)
            if self.manual_step:
                input('Press any key to continue...')
            _, _, value, actions, terminated = active_evaluator.step()
            if p1_turn:
                p1_evaluation = value[0] if value is not None else None
            else:
                p2_evaluation = value[0] if value is not None else None
            other_evaluator.step_evaluator(actions, terminated)
            if interactive:
                clear_output(wait=True)
            else:
                os.system('clear')
            if print_evaluation and value is not None:
                print(f'Evaluation: {value[0]}')
            if terminated:
                print('Game over!')
                print('Final state:')
                active_evaluator.env.print_state(actions.item())
                self.print_rewards(p1_player_id)
                break
                
            p1_turn = not p1_turn

    def print_rewards(self, p1_player_id):
        # Get reward from the perspective of the current player (p1_player_id is 1 or 2)
        reward = self.evaluator.env.get_rewards(torch.tensor([p1_player_id - 1]))[0]
        if reward == 1:
            print(f'Player 1 ({self.evaluator.__class__.__name__}) won!')
        elif reward == -1:
            print(f'Player 2 ({self.evaluator2.__class__.__name__}) won!')
        else:
            print('Draw!')
