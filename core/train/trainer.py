## File: core/train/trainer.py

import torch
import logging
from core.test.tester import TesterConfig, Tester
from core.train.collector import Collector
from core.utils.history import TrainingMetrics, Metric

class Trainer:
    """
    Trainer specialized for an environment.
    """
    def __init__(self,
        config,
        collector: Collector,
        tester: Tester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'model',
        debug: bool = False
    ):
        self.config = config
        self.collector = collector
        self.tester = tester
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.raw_train_config = raw_train_config
        self.raw_env_config = raw_env_config
        self.history = history
        self.log_results = log_results
        self.interactive = interactive
        self.run_tag = run_tag
        self.debug = debug

    def policy_transform(self, policy):
        return policy

    def value_transform(self, value):
        return value

    def training_step(self):
        inputs, target_policy, target_value, legal_actions = zip(*self.collector.episode_memory.sample(self.config.minibatch_size))
        inputs = torch.stack(inputs).to(device=self.device)

        target_policy = torch.stack(target_policy).to(device=self.device)
        target_policy = self.policy_transform(target_policy)

        target_value = torch.stack(target_value).to(device=self.device)
        target_value = self.value_transform(target_value)

        legal_actions = torch.stack(legal_actions).to(device=self.device)

        self.optimizer.zero_grad()
        policy_logits, values = self.model(inputs)
        # multiply policy logits by legal actions mask, set illegal actions to very small negative numbers
        policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
        policy_loss = self.config.policy_factor * torch.nn.functional.cross_entropy(policy_logits, target_policy)
        # FIX: Remove the factor *2 so that the value loss is computed directly on the [-1, 1] outputs
        value_loss = torch.nn.functional.mse_loss(values.flatten(), target_value)
        loss = policy_loss + value_loss

        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy_logits, dim=1)).float().mean()

        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()

    def train_minibatch(self):
        self.model.train()
        memory_size = len(self.collector.episode_memory.memory)
        if memory_size >= self.config.replay_memory_min_size:
            policy_loss, value_loss, policy_accuracy, loss = self.training_step()
            self.history.add_training_data({
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'policy_accuracy': policy_accuracy,
                'loss': loss
            }, log=self.log_results)
        else:
            logging.info(f'Replay memory size ({memory_size}) is less than minimum required ({self.config.replay_memory_min_size}), skipping training step')

    def add_collection_metrics(self, episodes):
        # episodes is a list of finished episodes for all parallel environments
        for ep in episodes:
            final_reward = ep[-1][2].item()  # (inputs, visits, reward, legal_actions)
            self.history.add_episode_data({'episode_reward': final_reward}, log=self.log_results)
            # For win rate, win is defined as final_reward > 0
            win_val = 1.0 if final_reward > 0 else 0.0
            self.history.add_episode_data({'episode_win_rate': win_val}, log=self.log_results)

    def add_epoch_metrics(self):
        # Not strictly required. We can gather, for example, an average from the stored episode data.
        pass

    def fill_replay_memory(self):
        # Fill the replay memory until it has enough samples.
        while len(self.collector.episode_memory.memory) < self.config.replay_memory_min_size:
            finished_episodes, _ = self.collector.collect()
            for episode in finished_episodes:
                self.collector.episode_memory.insert(episode)

    def training_loop(self, epochs: int = None):
        epoch = 0
        if epochs is None:
            epochs = float('inf')
        # Optionally run initial test batch with untrained model
        if self.tester.config.episodes_per_epoch > 0:
            self.tester.collect_test_batch()
        while epoch < epochs:
            self.history.cur_epoch = epoch
            # Populate replay memory if necessary
            self.fill_replay_memory()
            # Train an epoch
            finished_episodes, _ = self.collector.collect()
            self.add_collection_metrics(finished_episodes)
            self.train_minibatch()
            if self.tester.config.episodes_per_epoch > 0:
                self.tester.collect_test_batch()
            self.add_epoch_metrics()
            epoch += 1
