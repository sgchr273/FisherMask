from .random_sampling import RandomSampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .entropy_sampling import EntropySampling
from .least_confidence_dropout import LeastConfidenceDropout
from .margin_sampling_dropout import MarginSamplingDropout
from .entropy_sampling_dropout import EntropySamplingDropout
from .kmeans_sampling import KMeansSampling
from .kcenter_greedy import KCenterGreedy
from .bayesian_active_learning_disagreement_dropout import BALDDropout
from .core_set import CoreSet
from .adversarial_bim import AdversarialBIM
from .adversarial_deepfool import AdversarialDeepFool
from .active_learning_by_learning import ActiveLearningByLearning
from .badge_sampling  import BadgeSampling
from .baseline_sampling  import BaselineSampling
from .bait_sampling  import BaitSampling
from .fisher_mask_sampling import FishMaskSampling
from .deter_unc_sampling import ResNetDUQEntropySampling
from .fish_ent_sampling import FishEntSampling

#if it gives module not found error for some file, do this
# git add path to File
# git commit -m 'some text'
# git push