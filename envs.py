from ssbm_gym.ssbm_env import BaseEnv, isDying
from copy import deepcopy

def make_env(frame_limit, options):
    def _init():
        env = GoHighEnv(frame_limit=frame_limit, options=options)
        return env
    return _init


def GoHighEnvVec(num_envs, frame_limit=1e9, options={}):
    return SubprocVecEnv([make_env(frame_limit=frame_limit, options=options) for _ in range(num_envs)])


class GoHighEnv(BaseEnv):
    def __init__(self, **kwargs):
        BaseEnv.__init__(self, **kwargs)
        self._embed_obs = MinimalEmbedGame()

    @property
    def action_space(self):
        if self._action_space is not None:
            return self._action_space
        else:
            from ssbm_gym.spaces import MinimalActionSpace
            self._action_space = MinimalActionSpace()
            return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = self._embed_obs
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def compute_reward(self):
        r = 0.0
        controller = self.obs.players[self.pid].controller
        for c in [controller.button_A, controller.button_Y, controller.stick_MAIN.x]:
            r -= abs(c) / 100.0

        if self.prev_obs is not None:
            # This is necesarry because the character might be dying during multiple frames
            if not isDying(self.prev_obs.players[self.pid]) and \
               isDying(self.obs.players[self.pid]):
                r -= 1.0
            
        #     # We give a reward of -0.01 for every percent taken. The max() ensures that not reward is given when a character dies
        #     r -= 0.01 * max(0, self.obs.players[self.pid].percent - self.prev_obs.players[self.pid].percent)

        r += self.obs.players[0].y / 50 / 60
        return r

    def step(self, action):
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        obs = self.api.step([self.action_space.from_index(action)])
        self.obs = obs
        reward = self.compute_reward()
        done = self.is_terminal()
        infos = dict({'frame': self.obs.frame})

        return self.embed_obs(self.obs), reward, done, infos



class MinimalEmbedPlayer():
    def __init__(self):
        self.n = 3

    def __call__(self, player_state):
        # percent = player_state.percent/100.0
        # facing = player_state.facing
        x = player_state.x/10.0
        y = player_state.y/10.0
        # invulnerable = 1.0 if player_state.invulnerable else 0
        # hitlag_frames_left = player_state.hitlag_frames_left/10.0
        # hitstun_frames_left = player_state.hitstun_frames_left/10.0
        # shield_size = player_state.shield_size/100.0
        in_air = 1.0 if player_state.in_air else 0.0

        return [
                # percent,
                # facing,
                x, y,
                # invulnerable,
                # hitlag_frames_left,
                # hitstun_frames_left,
                # shield_size,
                in_air
            ]


class MinimalEmbedGame():
    def __init__(self):
        self.embed_player = MinimalEmbedPlayer()
        self.n = self.embed_player.n

    def __call__(self, game_state):
        player0 = self.embed_player(game_state.players[0])
        # player1 = self.embed_player(game_state.players[1])

        return player0  # + player1  # concatenates lists


import multiprocessing
import cloudpickle
import pickle

class CloudpickleWrapper(object):
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(): 
    def __init__(self, env_fns, start_method=None):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True


    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return obs


    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True


    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]


    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()


    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

