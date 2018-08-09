#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:07:57 2018

@author: wfd
"""
import retro     
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        self.state_input = tf.placeholder("float",[None,self.state_dim[0],self.state_dim[1],self.state_dim[2]])
        # network weights
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable("weight", [5, 5, 3, 32], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(self.state_input, conv1_weights, strides=[1,1,1,1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        
        with tf.name_scope('layer2-pool1'):
            pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        with tf.variable_scope('layer3-conv2'):
            conv2_weights = tf.get_variable("weight", [5, 5, 32, 48], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [48], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        with tf.name_scope('layer4-pool2'):
            pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        with tf.variable_scope('layer5-conv3'):
            conv3_weights = tf.get_variable("weight", [5, 5, 48, 64], 
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1,1,1,1], padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        with tf.name_scope('layer6-pool3'):
            pool3 = tf.nn.max_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool3, [-1, nodes])
        with tf.variable_scope('layer7-fc1'):
            fc1_weights = tf.get_variable("weights", [nodes, 512], 
                                          initializer = tf.truncated_normal_initializer(stddev=0.1))
            fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            fc1 = tf.nn.dropout(fc1, 0.5)
        with tf.variable_scope('layer8-fc2'):
            fc2_weights = tf.get_variable("weights", [512, 8],
                                          initializer = tf.truncated_normal_initializer(stddev=0.1))
            fc2_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.1))
            self.Q_value = tf.matmul(fc1, fc2_weights) + fc2_biases
        
        #W1 = self.weight_variable([self.state_dim[1],self.state_dim[0],20])
        #b1 = self.bias_variable([20])
        #W2 = self.weight_variable([20,self.action_dim])
        #b2 = self.bias_variable([self.action_dim])
        # input layer
        #self.state_input = tf.placeholder("float",[None,self.state_dim[0],self.state_dim[1],self.state_dim[2]])
        # hidden layers
        #h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        #self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = list(action)
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
                self.y_input:y_batch,
                self.action_input:action_batch,
                self.state_input:state_batch
                })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
                self.state_input: [state]
                })[0]
        if random.random() <= self.epsilon:
            l = [0]*self.action_dim
            l[np.random.randint(0, 7)] = 1
            return tuple(l)
        else:
            l = [0]*self.action_dim
            l[np.argmax(Q_value)] = 1
            return tuple(l)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self,state):
        l = [0]*self.action_dim
        l[np.argmax(self.Q_value.eval(feed_dict = {
                self.state_input: [state]
                })[0])] = 1
        return tuple(l)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial) 

def processing(state):
    state = state[:,:160,:]
    return state

def renew(next_state, reward, done, info):
    next_state = next_state[:,:160,:]
    reward = info['score2'] - info['score1']
    return next_state, reward, done, info
#........................................................................
ENV_NAME = 'Boxing-Atari2600'
EPISODE = 10000 # Episode limitation
STEP = 3000 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
def main():
    # initialize OpenAI Gym env and dqn agent
    env=retro.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = processing(env.reset())
        # Train
        for step in range(STEP):
            env.render()
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state,reward,done,info = env.step(action)
            next_state,reward,done,info = renew(next_state,reward,done,info)
            # Define reward for agent
            #reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
        for i in range(TEST):
            state = processing(env.reset())
            for j in range(STEP):
                env.render()
                if j%20 == 0:
                    action = agent.action(state) # direct action for test
                state,reward,done,info = env.step(action)
                state,reward,done,info = renew(state,reward,done,info)
                total_reward += reward
            if done:
                break
        ave_reward = total_reward/TEST
        print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        if ave_reward >= 2000000:
            break


if __name__ == '__main__':
    main()