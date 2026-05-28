REGISTRY = {}

from .episode_runner import EpisodeRunner
# from .episode_runner_v2_defense import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
