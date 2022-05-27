import logging
import math
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, InputLayer, LeakyReLU
from tensorflow.keras import initializers, activations
import tensorflow.keras
import tensorflow
import numpy as np
from collections import deque
import random
import copy

class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.players_order = players_order
        self.reward = 0
        self.epsilon = 0.2
        self.gamma = 0.5
        self.learning_rate = 0.0001
        self.first_layer_size = 1200
        self.second_layer_size = 1000
        self.third_layer_size = 900
        self.batch_size = 40
        self.weight_update_on_move = 30
        self.load_weights = True
        self.train = True
        self.memory = deque(maxlen=2000)
        self.logger = logging.getLogger('AI')
        self.q_model = None
        self.target_model = None
        self.attack_move_limiter = 50
        self.last_largest_region = (0,)
        self.output_layer_size = (29*29) + 1
        #self.output_layer_size = (29 * 29)
        self.moves = None

    def create_move_combinations(self):
        self.moves = []
        for i in range(1, 30):
            for j in range(1, 30):
                self.moves.append((i, j))
        self.moves.append((-1, -1))

    def get_largest_region_size(self, board, player_name):
        players_regions = board.get_players_regions(player_name)
        largest_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == largest_region_size]
        largest_region_areas = []
        for region in max_sized_regions:
            for area in region:
                largest_region_areas.append(area)
        return largest_region_size, largest_region_areas

    def normalize_input(self, input_data):
        return 2*((input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data)))-1


    # gpu version

    def create_nn(self, input_shape=(29, 19)):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(self.first_layer_size, name='d1', bias_initializer='zeros', kernel_initializer=tensorflow.initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(self.second_layer_size, name='d2', bias_initializer='zeros', kernel_initializer=tensorflow.initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.05))
        #model.add(Dense(self.third_layer_size, name='d3', activation=activations.tanh, bias_initializer='zeros', kernel_initializer='random_normal'))
        model.add(Dense(self.output_layer_size, name='dout', bias_initializer='zeros', kernel_initializer=tensorflow.initializers.glorot_uniform()))
        model.add(LeakyReLU(alpha=0.05))
        #print("output shape", model.output_shape)
        optimizer = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def get_available_attacks(self, board, player_name):
        available_attacks = list()
        attacks_iterator = possible_attacks(board, player_name)
        for source, target in attacks_iterator:
            available_attacks.append((source.get_name(), target.get_name()))
        return available_attacks

    def initialize_models(self):
        self.q_model = AI.create_nn(self)
        self.target_model = AI.create_nn(self)
        self.target_model.set_weights(self.q_model.get_weights())

    def DeepQLearning(self, board, player_name, num_of_turns):
        if self.train and len(self.memory):
            #print('awgew')
            calculated_reward = AI.calculate_reward(self, board, player_name)
            AI.update_memory(self, board, calculated_reward)
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for state, new_state, action_index, reward in minibatch:
                q_input = AI.create_state_representation(self, state, self.player_name)
                q_input = AI.normalize_input(self, q_input)
                q_pred = self.q_model.predict(q_input[np.newaxis, :, :])[0]

                t_input = AI.create_state_representation(self, new_state, self.player_name)
                t_input = AI.normalize_input(self, t_input)
                t_pred = self.target_model.predict(t_input[np.newaxis, :, :])[0]
                q_pred[action_index] = reward + self.gamma * max(t_pred)
                #print('p Q, rd, nQ', tempa, reward, q_pred[action_index])
                fit_x = AI.create_state_representation(self, state, self.player_name)
                fit_x = AI.normalize_input(self, fit_x)
                q_pred = q_pred.reshape(q_pred.shape[0])
                #print('Rewardy su: ',q_pred)
                self.q_model.fit(fit_x[np.newaxis, :, :], q_pred[np.newaxis, :], epochs=1, verbose=0)
            if self.train:
                self.q_model.save_weights('q_model_weights.h5')
                if self.weight_update_on_move == 0:
                    print('Prepisanie vah z q_modelu do target_modelu')
                    self.target_model.set_weights(self.q_model.get_weights())
                    self.weight_update_on_move = 30
                else:
                    self.weight_update_on_move -= 1

    def get_index_of_move_from_allmoves(self, command):
        for i in range(len(self.moves)):
            if self.moves[i] == command:
                return i
        return -1

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        # TODO posledny test bol bez -1 -1 v dostupnych tahoch tak to vratit s5
        np.set_printoptions(suppress=True)
        self.board = board
        AI.create_move_combinations(self)

        if not self.q_model or not self.target_model:
            AI.initialize_models(self)
            if self.load_weights:
                print('nacitava vahy')
                self.q_model.load_weights('q_model_weights.h5')
                self.target_model.load_weights('q_model_weights.h5')

        list_of_possible_attacks = AI.get_available_attacks(self, board, self.player_name)

        if not len(list_of_possible_attacks):
            #print('ENdturnuje lebo niesu dostupne ziadne moves')
            return EndTurnCommand()

        #list_of_possible_attacks.append((-1, -1))
        valid_q_values = []
        nn_input = AI.create_state_representation(self, board, self.player_name)
        nn_input = AI.normalize_input(self, nn_input)
        #print('Normalizovany input, ', nn_input)
        # Input ma byt 3 dimenzie a vracia 3d pole, tak preto [0] na konci
        predicted_q_values = self.target_model.predict(nn_input[np.newaxis, :, :])[0]
        predicted_q_values = predicted_q_values.reshape(predicted_q_values.shape[0])
        for possible_attack in list_of_possible_attacks:
            #print(possible_attack)
            #print("index all moves", self.moves.index(possible_attack))
            valid_q_values.append(predicted_q_values[self.moves.index(possible_attack)])
        max_q_move_from_possible = valid_q_values.index(max(valid_q_values))
        command = list_of_possible_attacks[max_q_move_from_possible]
        move_index_from_all_moves = AI.get_index_of_move_from_allmoves(self, command)
        if self.train:
            if np.random.rand() <= self.epsilon:
                print('10% sanca')
                random_move_index = np.random.randint(0, len(list_of_possible_attacks))
                command = list_of_possible_attacks[random_move_index]
                move_index_from_all_moves = AI.get_index_of_move_from_allmoves(self, command)
                AI.DeepQLearning(self, board, self.player_name, nb_turns_this_game)
            else:
                AI.DeepQLearning(self, board, self.player_name, nb_turns_this_game)

            if command and command != (-1, -1):
                AI.insert_to_memory(self, board, None, move_index_from_all_moves, self.reward)
                print('Odosiela utok: ', command[0], command[1])
                return BattleCommand(command[0], command[1])
            else:
                if command == (-1, -1):
                    AI.insert_to_memory(self, board, None, move_index_from_all_moves, self.reward)
                    print(' Endturnuje lebo -1-1 ma najlepsie Q value')
                    return EndTurnCommand()
            print('End turn just in case')
            return EndTurnCommand()

        if not self.train:
            if command and command != (-1, -1):
                print('Odosiela utok: ', command[0], command[1])
                return BattleCommand(command[0], command[1])
            print('Nema command')
            return EndTurnCommand()
        print('EndTurnuje')
        return EndTurnCommand()

    def update_memory(self, new_board, new_reward):
        last_memory = self.memory.pop()
        AI.insert_to_memory(self, last_memory[0], new_board, last_memory[2], new_reward)

    def calculate_reward(self, new_state, player_name):
        selected_memory = self.memory[-1]
        memory_board = selected_memory[0]
        action_index = selected_memory[2]
        calculated_reward = 0
        largest_region_size_before, largest_region_contains_before = AI.get_largest_region_size(self, memory_board, player_name)
        largest_region_size_now, largest_region_contains_now = AI.get_largest_region_size(self, new_state, player_name)
        list_of_player_areas_before = memory_board.get_player_areas(player_name)
        list_of_player_areas_now = new_state.get_player_areas(player_name)
        num_of_areas_before = len(list_of_player_areas_before)
        num_of_areas_now = len(list_of_player_areas_now)
        #print('Pocita pre index:', action_index)
        atk_area_index, def_area_index = self.moves[action_index]
        if atk_area_index == -1 or def_area_index == -1:
            atk_success_chance = 0
        else:
            atk_success_chance = probability_of_successful_attack(memory_board, atk_area_index, def_area_index)

        if num_of_areas_now > num_of_areas_before:
            calculated_reward += abs(num_of_areas_now - num_of_areas_before) * 0.0001
        else:
            calculated_reward -= abs(num_of_areas_now - num_of_areas_before) * 0.0001

        if largest_region_size_now > largest_region_size_before:
            calculated_reward += abs(largest_region_size_now - largest_region_size_before) * 0.0005
        else:
            calculated_reward -= abs(largest_region_size_now - largest_region_size_before) * 0.0005

        if atk_success_chance > 0.5:
            calculated_reward += 0.0002
        else:
            calculated_reward -= 0.0002
        return calculated_reward

    def create_state_representation(self, board, player_name):
        state_array = np.empty((29, 19), dtype='int')
        for area_index in range(1, 30):
            current_values = np.full(19, -1, dtype='int')
            current_area = board.get_area(area_index)
            area_id = current_area.get_name()
            current_values[0] = area_id
            area_owner_id = current_area.get_owner_name()
            current_values[1] = area_owner_id
            can_attack = current_area.can_attack()
            current_values[2] = can_attack
            dice_value = current_area.get_dice()
            current_values[3] = dice_value
            largest_region_size, largest_region_contains = AI.get_largest_region_size(self, board, player_name)
            if area_id in largest_region_contains:
                current_values[4] = 1
            else:
                current_values[4] = 0
            neighbors = current_area.get_adjacent_areas()
            temp = 5
            for neighbor_id in neighbors:
                current_values[temp] = neighbor_id
                atk = board.get_area(area_id)
                deff = board.get_area(neighbor_id)
                atkpw = atk.get_dice()
                deffpw = deff.get_dice()
                if atkpw == 1 or deffpw == 1:
                    q = 0
                else:
                    q = probability_of_successful_attack(board, area_id, neighbor_id)
                current_values[temp + 7] = q * 100
                temp += 1
            state_array[area_index-1] = current_values
        return np.array(state_array)

    def insert_to_memory(self, state, next_state, performed_move_index, reward):
        self.memory.append((state, next_state, performed_move_index, reward))
