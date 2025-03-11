import os
import matplotlib.pyplot as plt
import json

curdir = os.path.dirname(os.path.abspath(__file__))
with open(curdir + "/RECORD_baseline750.json", "r") as f:
   data1 = json.load(f)

with open(curdir + "/RECORD_inpnorm750.json", "r") as f:
   data2 = json.load(f)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(data1["actor_loss"])
ax1.plot(data2["actor_loss"])
ax1.set_title('Actor (policy) Loss')

ax2.plot(range(0, 750), data1["critic_loss"], label = 'No input normalization')
ax2.plot(range(0,750), data2["critic_loss"], label= 'Input normalization')
#ax2.legend()
ax2.set_title('Critic (V) Loss')
plt.xlabel('Epochs')

ax3.plot(range(0, 750), data1["reward"])
ax3.plot(range(0, 750), data2["reward"])
ax3.set_title('Reward for each trajectory')

plt.tight_layout()
fig.legend()
# fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1.plot(data2["actor_loss"])
# ax1.set_title('Actor (policy) Loss')
# ax2.plot(data2["critic_loss"])
# ax2.set_title('Critic (V) Loss')
# ax3.plot(data2["reward"])
# ax3.set_title('Reward for each trajectory')
# plt.tight_layout()
plt.show()