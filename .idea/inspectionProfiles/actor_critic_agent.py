import numpy as np
import tensorflow as tf
import gym

import Fre_env as env
import Other_method as Oth
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# 超参数
OUTPUT_GRAPH = False
MAX_EPISODE = 1
DISPLAY_REWARD_THRESHOLD = 200  # 刷新阈值
MAX_STEPS = 150 # 最大迭代次数
RENDER = False  # 渲染开关
GAMMA = 0.9  # 衰变值
LR_A = 0.001  # Actor学习率
LR_C = 0.01  # Critic学习率

#env = gym.make('CartPole-v0')
#env.seed(1)
#env = env.unwrapped

# N_F = env.observation_space.shape[0]  # 状态空间
# N_A = env.action_space.n  # 动作空间

#K = 5



class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # 获取所有操作的概率
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


if __name__ == "__main__":

    T = 20
    r11 = np.zeros(shape=(T), dtype=float)
    r21 = np.zeros(shape=(T), dtype=float)
    r31 = np.zeros(shape=(T), dtype=float)
    r71 = np.zeros(shape=(T), dtype=float)

    N = 100
    Apply_num = N
    M = 20

    for k in range(T):
        K = k + 1
        N_F = N * M * K
        N_A = K
        tf.reset_default_graph()
        with tf.Session() as sess:
            #sess = tf.Session()
            actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)  # 初始化Actor
            critic = Critic(sess, n_features=N_F, lr=LR_C)  # 初始化Critic
            sess.run(tf.global_variables_initializer())  # 初始化参数

            #if OUTPUT_GRAPH:
                #tf.summary.FileWriter("logs/", sess.graph)  # 输出日志

            # 开始迭代过程 对应伪代码部分
            # for i_episode in range(MAX_EPISODE):
            state, Location_matrix, A3C_Allocation_matrix = env.reset(N, M, K)  # 环境初始化
            reward_all = 0
            arg_num = 0
            #track_r = []  # 每回合的所有奖励
            for step in range (MAX_STEPS):
                #if RENDER: env.render()
                action = actor.choose_action(state)  # Actor选取动作
                next_state, reward, A3C_Allocation_matrix = env.step(reward_all, arg_num, A3C_Allocation_matrix,
                                                                     action, Location_matrix, N, M, K)  # 环境反馈
                #if done: r = -20  # 回合结束的惩罚
                reward_all += reward
                arg_num += 1
                if arg_num == Apply_num:
                    arg_num = 0

                #track_r.append(r)  # 记录回报值r
                td_error = critic.learn(state, reward, next_state)  # Critic 学习
                actor.learn(state, action, td_error)  # Actor 学习
                state = next_state
            # 回合结束, 打印回合累积奖励
            r7 = reward_all
            r71[k] = r7
        r1, r2, r3 = Oth.run_process(N, M, K, Location_matrix)
        r11[k] = r1
        r21[k] = r2
        r31[k] = r3
        print(r1, r2, r3, r7)
        print(r1,r2,r3,r7)

    t = np.arange(1 , T + 1)
    plt.plot(t, np.log(r11 + 1e-5), color='r', linestyle=':', marker=None, label='random')
    plt.plot(t, np.log(r21 + 1e-5), color='c', linestyle='-.', marker=None, label='Greedy')
    plt.plot(t, np.log(r31 + 1e-5), color='y', linestyle='-', marker=None, label='Ep_Greedy')
    plt.plot(t, np.log(r71 + 1e-5), color='b', linestyle='-', marker=None, label='A3C')
    plt.xlabel("Frequence_Num")
    plt.ylabel("H")
    plt.title("Frequence_number_influence")
    plt.legend()
    #plt.savefig("频点A_C初步分析")
    plt.show()