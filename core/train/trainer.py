from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
import torch
import logging
from pathlib import Path
from core.algorithms.evaluator import EvaluatorConfig
from core.resnet import TurboZeroResnet
from core.test.tester import TesterConfig, Tester
from core.train.collector import Collector
from core.utils.history import Metric, TrainingMetrics

from core.utils.memory import GameReplayMemory, ReplayMemory


def init_history(log_results: bool = True):
    return TrainingMetrics(
        train_metrics=[
            Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=log_results),
            Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=log_results, proper_name='Value Loss'),
            Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=log_results, proper_name='Policy Loss'),
            Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=log_results, proper_name='Policy Accuracy'),
        ],
        episode_metrics=[],
        eval_metrics=[],
        epoch_metrics=[]
    )


@dataclass
class TrainerConfig:
    algo_config: EvaluatorConfig
    episodes_per_epoch: int
    learning_rate: float
    lr_decay_gamma: float
    minibatch_size: int
    minibatches_per_update: int
    parallel_envs: int
    policy_factor: float
    replay_memory_min_size: int
    replay_memory_max_size: int
    test_config: TesterConfig
    replay_memory_sample_games: bool = True


class Trainer:
    def __init__(self,
        config: TrainerConfig,
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
    ):
        self.collector = collector
        self.tester = tester
        self.parallel_envs = collector.evaluator.env.parallel_envs
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.log_results = log_results
        self.interactive = interactive
        self.run_tag = run_tag
        self.raw_train_config = raw_train_config
        self.raw_env_config = raw_env_config
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay_gamma)
        

        self.history = history 

        if config.replay_memory_sample_games:
            self.replay_memory = GameReplayMemory(
                config.replay_memory_max_size
            )
        else:
            self.replay_memory = ReplayMemory(
                config.replay_memory_max_size,
            )
    
    def add_collection_metrics(self, episodes):
        raise NotImplementedError()
    
    def add_train_metrics(self, policy_loss, value_loss, policy_accuracy, loss):
        self.history.add_training_data({
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': policy_accuracy,
            'loss': loss
        }, log=self.log_results)

    def add_epoch_metrics(self):
        raise NotImplementedError()
    
    def training_steps(self, num_steps: int):
        if num_steps > 0:
            self.model.train()
            memory_size = self.replay_memory.size()
            if memory_size >= self.config.replay_memory_min_size:
                for _ in range(num_steps):
                    minibatch_value_loss = 0.0
                    minibatch_policy_loss = 0.0
                    minibatch_policy_accuracy = 0.0
                    minibatch_loss = 0.0
                    
                    for _ in range(self.config.minibatches_per_update):
                        policy_loss, value_loss, polcy_accuracy, loss = self.training_step()
                        minibatch_value_loss += value_loss
                        minibatch_policy_loss += policy_loss
                        minibatch_policy_accuracy += polcy_accuracy
                        minibatch_loss += loss

                    minibatch_value_loss /= self.config.minibatches_per_update
                    minibatch_policy_loss /= self.config.minibatches_per_update
                    minibatch_policy_accuracy /= self.config.minibatches_per_update
                    minibatch_loss /= self.config.minibatches_per_update
                    self.add_train_metrics(minibatch_policy_loss, minibatch_value_loss, minibatch_policy_accuracy, minibatch_loss)
            else:
                logging.info(f'Replay memory samples ({memory_size}) <= min samples ({self.config.replay_memory_min_size}), skipping training steps')
    
    def training_step(self):
        inputs, target_policy, target_value = zip(*self.replay_memory.sample(self.config.minibatch_size))
        inputs = torch.stack(inputs).to(device=self.device)

        target_policy = torch.stack(target_policy).to(device=self.device)
        target_policy = self.policy_transform(target_policy)

        target_value = torch.stack(target_value).to(device=self.device)
        target_value = self.value_transform(target_value)

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)

        policy_loss = self.config.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value.flatten(), target_value)
        loss = policy_loss + value_loss

        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).float().mean()

        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()
    
    def policy_transform(self, policy):
        return policy.div(policy.sum(dim=1, keepdim=True))
    
    def value_transform(self, value):
        return value
    
    def selfplay_step(self):
        finished_episodes, _ = self.collector.collect()
        if finished_episodes:
            for episode in finished_episodes:
                episode = self.collector.postprocess(episode)
                self.replay_memory.insert(episode)
        self.add_collection_metrics(finished_episodes)
        
        num_train_steps = len(finished_episodes)
        self.training_steps(num_train_steps)

    def training_loop(self, epochs: Optional[int] = None):
        while self.history.cur_epoch < epochs if epochs is not None else True:
            while self.history.cur_train_step < self.config.episodes_per_epoch * (self.history.cur_epoch+1):
                self.selfplay_step()
            if self.tester.config.episodes_per_epoch > 0:
                self.tester.collect_test_batch()
            self.add_epoch_metrics()

            if self.interactive:
                self.history.generate_plots()
            self.scheduler.step()
            self.history.start_new_epoch()
            self.save_checkpoint()
            
    def save_checkpoint(self, custom_name: Optional[str] = None) -> None:
        directory = f'./checkpoints/{self.run_tag}/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = custom_name if custom_name is not None else str(self.history.cur_epoch)
        filepath = directory + f'{filename}.pt'
        torch.save({
            'model_arch_params': self.model.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'run_tag': self.run_tag,
            'raw_train_config': self.raw_train_config,
            'raw_env_config': self.raw_env_config
        }, filepath)

def load_checkpoint(checkpoint_file: str):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    history = checkpoint['history']
    run_tag = checkpoint['run_tag']
    raw_train_config = checkpoint['raw_train_config']
    raw_env_config = checkpoint['raw_env_config']
    model = TurboZeroResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    raw_train_config = checkpoint['raw_train_config']
    optimizer = torch.optim.AdamW(model.parameters(), raw_train_config['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, history, run_tag, raw_train_config, raw_env_config


