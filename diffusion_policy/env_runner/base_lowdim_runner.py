from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class BaseLowdimRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        raise NotImplementedError()

    def close(self):
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()
