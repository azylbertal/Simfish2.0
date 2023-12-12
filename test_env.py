import json
from Environment.SimFishEnv import BaseEnvironment
import matplotlib.pyplot as plt

env_variables = json.load(open('Environment/1_env.json', 'r'))
test_env = BaseEnvironment(env_variables=env_variables)


#test_env.reset()

res = test_env.step(action=0)

print(res)


# print(observation.shape)
# print(reward)
# print(internal_state)
# print(done)
# print(full_masked_image.shape)

# plt.figure()
# plt.imshow(full_masked_image.get())
# plt.figure()
# plt.imshow(observation[:, :, 0].T)
# plt.figure()
# plt.imshow(observation[:, :, 1].T)

# plt.show()