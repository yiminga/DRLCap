import math
import os
import tensorflow as tf
import numpy as np

os.system("echo JQX_ard_123 | sudo -S nvidia-smi -pm 1")
os.system("echo JQX_ard_123 | sudo -S nvidia-smi -pl 257")

# 强化学习 State：(1)GPU频率，(2)GPU利用率，(3)显存利用率，(4)实时功耗，(4)实时温度，(6)显存频率，(7)当前的power cap
GPU_LABELS = ('CLOCKS_GPU'
              , 'UTIL_GPU'
              , 'UTIL_MEM'
              , 'POWER'
              , 'TEMP'
              )
MINS = {'CLOCKS_GPU': 300, 'UTIL_GPU': 0, 'UTIL_MEM': 0, 'POWER': 20, 'TEMP': 30}
MAXS = {'CLOCKS_GPU': 2100, 'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 260, 'TEMP': 90}
BUCKETS = {'CLOCKS_GPU': 120, 'UTIL_GPU': 10, 'UTIL_MEM': 10, 'POWER': 12, 'TEMP': 30}
gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)

# GPU power limit   GeForce RTX 2080 Ti  100~280 TDP:257
POWER_IN_STATE=1
gpu_limit = 257
GPU = range(120,258,1)
gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}

MEM =[405,810,5000,6800,7000]
clock_mem_bucket ={MEM[i]: i for i in range(len(MEM))}

# RL config
N_FEATURES = 7  # 特征，作为深度神经网络的输入
EPSILON = 0.1 # 最终的探索率
LEARNING_RATE = 0.0001  # 学习率
REWARD_DECAY = 0.95  # Bellman Fuction中的gamma，意思是回报的折损率
REPLACE_TARGET_ITER = 10  # 每*次迭代就用输出替代目标函数  500 {50} 50
MEMORY_SIZE = 500  # replay的内存大小                   100 {500} 1000
BATCH_SIZE = 32  # 批大小,神经网络的训练都是一批一批的，这个变量指的就是每批的大小


# get state
def state():
    os.system(
        'nvidia-smi --format=csv,noheader,nounits --filename=state.csv  --query-gpu=power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu')
    os.system('sleep 0.5')
    with open('state.csv', 'r') as fo:
        states_lines = fo.readlines()
        for states in states_lines:
            states = states.replace(',', '')
            power_gpu = float(states.split()[0])
            clock_gpu = float(states.split()[1])
            clock_mem = float(states.split()[2])
            util_gpu = float(states.split()[3])
            util_memory = float(states.split()[4])
            temp = float(states.split()[5])
    stats = {
        'GPUL': gpu_limit,
        'CLOCKS_GPU': clock_gpu,
        'CLOCKS_MEM':clock_mem,
        'UTIL_GPU': util_gpu,
        'UTIL_MEM': util_memory,
        'POWER': power_gpu,
        'TEMP': temp}
    print(stats)

    # GPU states
    gpu_all_mins = np.array([MINS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_all_maxs = np.array([MAXS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_widths = np.divide(np.array(gpu_all_maxs) - np.array(gpu_all_mins), gpu_num_buckets)  # divide /

    gpu_raw_no_pow = [stats[k] for k in GPU_LABELS]  # wym modify
    gpu_raw_no_pow = np.clip(gpu_raw_no_pow, gpu_all_mins, gpu_all_maxs)  # clip set data at range(min max)

    gpu_raw_floored = gpu_raw_no_pow - gpu_all_mins
    gpu_state = np.divide(gpu_raw_floored, gpu_widths)
    gpu_state = np.clip(gpu_state, 0, gpu_num_buckets - 1)

    gpu_state = np.append(gpu_state, [clock_mem_bucket[stats['CLOCKS_MEM']]]) # Add mem frequency index to end of state:
    if POWER_IN_STATE:
        # Add power cap index to end of state:
        gpu_state = np.append(gpu_state, [gpu_to_bucket[stats['GPUL']]])

    gpu_state = [int(x) for x in gpu_state]
    print("gpu_state: ", gpu_state)

    return gpu_state, stats


# DDQN with priori
np.random.seed(1)
tf.set_random_seed(1)

print(tf.__version__)

# Deep Q Network off-policy
# Story data with its priority in the tree.
class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):

        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features=N_FEATURES,
            learning_rate=LEARNING_RATE,
            reward_decay=REWARD_DECAY,
            # e_greedy=E_GREEDY,
            replace_target_iter=REPLACE_TARGET_ITER,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            # e_greedy_increment=E_GREEDY_INCREMENT,
            output_graph=False,
            prioritized=True,
            sess=tf.Session(),
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = EPSILON
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # start 12/7/21
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        print('restore ckpt:')
        saver = tf.train.Saver()
        saver.restore(sess, '/home/wym/project/cpu_gpu/RL/net/save_net.ckpt')

        # end 12/7/21
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net 1 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names,
                                     trainable=True)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names, trainable=True)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,
                                     trainable=True)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,
                                     trainable=True)

        # ------------------ build target_net 1 ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1_ = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names,
                                      trainable=False)
                b1_ = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names, trainable=False)

            with tf.variable_scope('l2'):
                w2_ = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,
                                      trainable=False)
                b2_ = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,
                                      trainable=False)

        # 提取神经网络变量
        print('restore ckpt:')
        saver = tf.train.Saver()
        saver.restore(sess, '/home/wym/project/cpu_gpu/RL/net/save_net.ckpt')

        # ------------------ build evaluate_net 2 ------------------
        with tf.variable_scope('eval_net'):
            # 载入存储的变量
            sess.run(w1)
            sess.run(b1)
            sess.run(w2)
            sess.run(b2)
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net 2 ------------------
        with tf.variable_scope('target_net'):
            # 载入存储的变量
            sess.run(w1_)
            sess.run(b1_)
            sess.run(w2_)
            sess.run(b2_)
            l1_ = tf.nn.relu(tf.matmul(self.s_, w1_) + b1_)
            self.q_next = tf.matmul(l1_, w2_) + b2_

        # 载入神经网络框架
        print('restore model:')
        saver = tf.train.Saver()
        saver.restore(sess, '/home/wym/project/cpu_gpu/RL/net/model')

    def store_transition(self, s, a, r, s_):
        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:  # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        print("epsilon:", self.epsilon)
        observation = observation[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            print("max!!!!!!!!!!!!!!!!!!!!!")
        else:
            action = np.random.randint(0, self.n_actions)
            print("random!!!!!!!!!!!!!!!!!!!!!")
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        #  DQN
        # q_next, q_eval = self.sess.run(
        #         [self.q_next, self.q_eval],
        #         feed_dict={self.s_: batch_memory[:, -self.n_features:],
        #                    self.s: batch_memory[:, :self.n_features]})

        # DDQN wym
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # DQN
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # DDQN wym
        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target,
                                                                self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_maxself.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


sess = tf.Session()
with tf.variable_scope('DQN_with_prioritized_replay'):
    RL = DQNPrioritizedReplay(
        n_actions=len(GPU)
    )

obeservation, states = state()
# while True:
for step in range(200000):
    # RL choose action based on observation
    action = RL.choose_action(np.array(obeservation))
    gpu_limit = GPU[action]
    print('power cap:', gpu_limit)
    os.system("echo JQX_ard_123 | sudo -S nvidia-smi -pl %s " % (gpu_limit))  # take action

    power_g = math.floor(states['POWER'])
    fre_g = states['CLOCKS_GPU']

    # get next obserbation and reward
    obeservation_, states_ = state()

    # define Reward
    power_g_ = math.floor(states_['POWER'])
    fre_g_ = states_['CLOCKS_GPU']
    power = power_g_ - power_g
    fre = fre_g_ - fre_g
    print('power :', power, fre)
    #
    # if power > 0 and fre > 0:
    #     reward = 3
    # elif power > 0 and fre < 0:
    #     reward = 2
    # elif power > 0 and fre == 0:
    #     reward = 1
    # elif power < 0 and fre > 0:
    #     reward = 9
    # elif power < 0 and fre < 0:
    #     reward = 8
    # elif power < 0 and fre == 0:
    #     reward = 7
    # elif power == 0 and fre > 0:
    #     reward = 6
    # elif power == 0 and fre > 0:
    #     reward = 5
    # elif power == 0 and fre == 0:
    #     reward = 4
    # else:
    #     reward = 0
    # print('reward : ', reward)
    # if power >  0  and  fre > 0:
    #     if power > 150:
    #         reward = 31
    #     elif 100 < power <= 150:
    #         reward = 32
    #     elif  50 < power <= 100:
    #         reward =33
    #     elif 40 < power <=50:
    #         reward = 34
    #     elif  30 < power <=40:
    #         reward = 35
    #     elif 20 < power <= 30:
    #         reward = 36
    #     elif 10 < power <= 20:
    #         reward = 37
    #     elif 5 < power <= 10:
    #         reward = 38
    #     elif 3 < power <= 5:
    #         reward = 39
    #     else:
    #         reward = 40
    # elif power > 0 and fre < 0:
    #     if power > 150:
    #         reward = 21
    #     elif 100 < power <= 150:
    #         reward = 22
    #     elif 50 < power <= 100:
    #         reward = 23
    #     elif 40 < power <= 50:
    #         reward = 24
    #     elif 30 < power <= 40:
    #         reward = 25
    #     elif 20 < power <= 30:
    #         reward = 26
    #     elif 10 < power <= 20:
    #         reward = 27
    #     elif 5 < power <= 10:
    #         reward = 28
    #     elif 3 < power <= 5:
    #         reward = 29
    #     else:
    #         reward = 30
    # elif power > 0 and fre == 0:
    #     if power > 150:
    #         reward = 11
    #     elif 100 < power <= 150:
    #         reward = 12
    #     elif 50 < power <= 100:
    #         reward = 13
    #     elif 40 < power <= 50:
    #         reward = 14
    #     elif 30 < power <= 40:
    #         reward = 15
    #     elif 20 < power <= 30:
    #         reward = 16
    #     elif 10 < power <= 20:
    #         reward = 17
    #     elif 5 < power <= 10:
    #         reward = 18
    #     elif 3 < power <= 5:
    #         reward = 19
    #     else:
    #         reward = 20
    # elif power < 0 and fre > 0:
    #     if power <= -150:
    #         reward = 100
    #     elif -150 < power <= -100:
    #         reward = 99
    #     elif -100 < power <= -50:
    #         reward = 98
    #     elif -50 < power <= -40:
    #         reward = 97
    #     elif -40 < power <= -30:
    #         reward = 96
    #     elif -30 < power <= -20:
    #         reward = 95
    #     elif -20 < power <= -10:
    #         reward = 94
    #     elif -10 < power <= -5:
    #         reward = 93
    #     elif -5 < power <= -3:
    #         reward = 92
    #     else:
    #         reward = 91
    # elif power < 0 and fre < 0:
    #     if power <= -150:
    #         reward = 90
    #     elif -150 < power <= -100:
    #         reward = 89
    #     elif -100 < power <= -50:
    #         reward = 88
    #     elif -50 < power <= -40:
    #         reward = 87
    #     elif -40 < power <= -30:
    #         reward = 86
    #     elif -30 < power <= -20:
    #         reward = 85
    #     elif -20 < power <= -10:
    #         reward = 84
    #     elif -10 < power <= -5:
    #         reward = 83
    #     elif -5 < power <= -3:
    #         reward = 82
    #     else:
    #         reward = 81
    # elif power < 0 and fre == 0:
    #     if power <= -150:
    #         reward = 80
    #     elif -150 < power <= -100:
    #         reward = 79
    #     elif -100 < power <= -50:
    #         reward = 78
    #     elif -50 < power <= -40:
    #         reward = 77
    #     elif -40 < power <= -30:
    #         reward = 76
    #     elif -30 < power <= -20:
    #         reward = 75
    #     elif -20 < power <= -10:
    #         reward = 74
    #     elif -10 < power <= -5:
    #         reward = 73
    #     elif -5 < power <= -3:
    #         reward = 72
    #     else:
    #         reward = 71
    # elif power == 0 and fre > 0:
    #     if fre > 300:
    #         reward = 68
    #     elif 150 < fre <= 300:
    #         reward = 66
    #     elif 75 < fre <= 150:
    #         reward = 64
    #     elif 30 < fre <= 75:
    #         reward = 62
    #     else:
    #         reward = 60
    # elif power == 0 and fre < 0:
    #     if fre <= -300:
    #         reward = 42
    #     elif -300 < fre <= -150:
    #         reward = 44
    #     elif -150 < fre <= -75:
    #         reward = 46
    #     elif -75 < fre <= -30:
    #         reward = 48
    #     else:
    #         reward =50
    # elif power == 0 and fre == 0:
    #     reward = 55
    # else:
    #     reward = 0
    if power <= 0:
        if -45 <= fre:
            reward = 10
        elif -90 <= fre < -45:
            reward = -1
        elif -135 <= fre < -90:
            reward = -2
        elif fre < -135:
            reward= -3
    if power > 0:
        if fre <= 45:
            reward = -1
        elif 45 < fre <= 90:
            reward = -2
        elif 90 < fre <= 135:
            reward = -3
        elif fre > 135:
            reward = -4
    print('reward : ', reward)
    RL.store_transition(obeservation, action, reward, obeservation_)

    # if (step > MEMORY_SIZE):
    RL.learn()

    # swap observation
    states = states_
    obeservation = obeservation_
    print('step-----------------:', step)
    print('\n')

os.system("echo JQX_ard_123 | sudo -S nvidia-smi -pl 257")
print('end')
