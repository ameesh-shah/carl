from gym.envs.registration import register

register(
    id='MBRLCartPole-v0',
    entry_point='env.cartpole:CartPoleEnv'
)

register(
    id='MBRLDuckietown-v0',
    entry_point='env.duckietown:Duckietown'
)

register(
    id='MBRLHalfCheetahDisabled-v0',
    entry_point='env.half_cheetah_disabled:HalfCheetahEnv'
)

register(
    id='PointmassEasy-v0',
    entry_point='env.pointmass:PointmassEnv',
    kwargs={'difficulty': 0}
)

register(
    id='PointmassMedium-v0',
    entry_point='env.pointmass:PointmassEnv',
    kwargs={'difficulty': 1}
)
