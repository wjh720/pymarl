from .q_learner import QLearner
from .coma_learner import COMALearner
from .maddpg_learner import MADDPGLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["maddpg_learner"] = MADDPGLearner
