from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        # 初始化经验回放池，使用deque来存储
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        # 向缓冲区添加一个新的经验
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        # 从缓冲区中随机抽取 batch_size 个样本
        return random.sample(self.buffer, batch_size)

    def size(self):
        # 返回缓冲区的当前大小
        return len(self.buffer)

'''
----------------------------------------------------<>
buffer = ReplayBuffer(100)
buffer.add("mikumiku", "hello", 1, "mikumiku")
buffer.add("state1", "action1", 1.0, "state2")
buffer.add("state2", "action2", 0.5, "state3")
print(buffer.size())
print(buffer.sample(2))
----------------------------------------------------<>
response:
3
[('mikumiku', 'hello', 1, 'mikumiku'), ('state1', 'action1', 1.0, 'state2')]
----------------------------------------------------<>
'''