import torch.multiprocessing as mp
from A3C import ActorCritic
from ICM import ICM
from shared_adam import SharedAdam
from worker import worker  # main loop of the program


class ParallelEnv:
    def __init__(self, env_id, input_shape, n_actions, icm, n_threads=8):
        names = [str(i) for i in range(1, n_threads + 1)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()  # share the memory of the global agent
        global_optim = SharedAdam(global_actor_critic.parameters())

        # weather to use ICM or not
        if not icm:
            global_icm = None
            global_icm_optim = None
        else:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters())

        # define our process
        self.ps = [mp.Process(target=worker, args=(
            name, input_shape, n_actions, global_actor_critic, global_icm, global_optim, global_icm_optim, env_id,
            n_threads, icm)) for name in names]

        # start and join our threads
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
