import numpy as np
import os


class HARLAttacker:

    def __init__(
            self,
            attack_type="none",
            model_path=None,
            budget=0.35,
            action_space_low=0.0,
            action_space_high=1.0,
            action_dim=5,
    ):
        self.attack_type = attack_type.lower()
        self.budget = budget
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.action_dim = action_dim
        self.model = None

        if self.attack_type in ["act", "dyn", "grad"]:
            if model_path is None:
                raise ValueError(f"{attack_type} attack requires model_path")
            self.model = self._load_model(model_path)
            print(f"[Attacker] type={attack_type}, model={model_path}")
        elif self.attack_type == "random":
            print(f"[Attacker] type=random (random action baseline)")
        elif self.attack_type == "none":
            print(f"[Attacker] type=none (no-attack baseline)")
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

    def _load_model(self, model_path):
        if os.path.isdir(model_path):
            candidates = [
                os.path.join(model_path, "best_model", "best_model.zip"),
                os.path.join(model_path, "best_model.zip"),
                os.path.join(model_path, "final_model.zip"),
                os.path.join(model_path, "model.zip"),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Attack model not found in {model_path}.\n"
                    f"Tried: {candidates}"
                )
            model_path = found

        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"[Attacker] Loaded SB3 PPO model: {model_path}")
            return model
        except ImportError:
            pass

        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(model_path)
            print(f"[Attacker] Loaded SB3 RecurrentPPO model: {model_path}")
            return model
        except (ImportError, Exception):
            pass

        raise ImportError(
            "Cannot load attack model. Please install stable-baselines3:\n"
            "  pip install stable-baselines3\n"
            "  pip install sb3-contrib  # if using LSTM policy"
        )

    def get_attack_action(self, victim_obs, normal_action=None, deterministic=True):
        if self.attack_type == "none":
            return normal_action

        if self.attack_type == "random":
            shape = normal_action.shape if normal_action is not None else (self.action_dim,)
            return np.random.uniform(
                self.action_space_low, self.action_space_high, size=shape
            ).astype(np.float32)

        if self.attack_type in ["act", "dyn"]:
            action, _ = self.model.predict(victim_obs, deterministic=deterministic)
            return np.clip(action, self.action_space_low, self.action_space_high)

        if self.attack_type == "grad":
            assert normal_action is not None, "grad attack requires normal_action"
            worst_action, _ = self.model.predict(victim_obs, deterministic=deterministic)
            direction = worst_action - normal_action
            attack_action = normal_action + self.budget * np.sign(direction)
            return np.clip(attack_action, self.action_space_low, self.action_space_high)

        raise ValueError(f"Unknown attack type: {self.attack_type}")

    def get_batch_attack_actions(self, victim_obs_batch, normal_actions_batch=None,
                                 deterministic=True):
        n = victim_obs_batch.shape[0]
        results = []

        for i in range(n):
            obs = victim_obs_batch[i]
            normal = normal_actions_batch[i] if normal_actions_batch is not None else None
            action = self.get_attack_action(obs, normal, deterministic)
            results.append(action.flatten())

        return np.array(results)


class RLlibAttacker:

    def __init__(self, checkpoint_path, attack_type="ACT", hidden_dim=128, budget=0.35):
        import torch
        from ray.rllib.policy.policy import Policy

        self.attack_type = attack_type.lower()
        self.budget = budget
        self.hidden_dim = hidden_dim

        self.policy = Policy.from_checkpoint(checkpoint_path)
        self.hidden = [
            torch.zeros(hidden_dim),
            torch.zeros(hidden_dim),
        ]

        print(f"[RLlibAttacker] Loaded: {checkpoint_path}")

    def init_hidden(self):
        import torch
        self.hidden = [
            torch.zeros(self.hidden_dim),
            torch.zeros(self.hidden_dim),
        ]

    def get_attack_action(self, victim_obs, normal_action=None, **kwargs):
        if self.attack_type == "grad":
            worst_action, self.hidden, _ = self.policy.compute_single_action(
                victim_obs, state=self.hidden
            )
            direction = worst_action - normal_action
            return normal_action + self.budget * np.sign(direction)
        else:
            action, self.hidden, _ = self.policy.compute_single_action(
                victim_obs, state=self.hidden
            )
            return action
