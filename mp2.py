#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: szczurpi
Spring, 2024
Artificial Intelligence
Machine Problem 2 Base Code
Agent vs Monster Minimax Search with alpha/beta pruning
Description:
    In this game, the agent has to grab the gold and escape to the exit position.
    The agent has to do this while trying to escape from the monster,
    which will eat the agent if they are co-located.
    The monster can move in four directions, except the walls.
        It can also stay in the same position.
    The agent can move in the same directions, but can also:
        grab the gold (if co-located with it), and
        build a wall in one of four directions, assuming the square is empty
    The agent must move or build a wall in each turn (cannot do nothing)

"""

'''
Name: Savannah Stumpf
Date: 4/6/2026
Course Name: Artificial Intelligence
Semester: Spring 2026
Assignment Name: Machine Problem 2
'''

import numpy as np
import random
import math
import time

class GenGameBoard: 
    """
    Class responsible for representing the game board and game playing methods
    """
    num_pruned = 0 # counts number of pruned branches due to alpha/beta
    MAX_DEPTH = 12  # max depth before applying evaluation function
    depth = 0      # current depth within minimax search
    
    UP = 'w'
    DOWN = 's'
    LEFT = 'a'
    RIGHT = 'd'
    UP_BUILD = 'wb'
    DOWN_BUILD = 'sb'
    LEFT_BUILD = 'ab'
    RIGHT_BUILD = 'db'
    
    def __init__(self, board_size=4):
        """
        Constructor method - initializes each position variable and the board
        """
        self.board_size = board_size  # Holds the size of the board
        self.marks = np.empty((board_size, board_size),dtype='str')  # Holds the mark for each position
        self.marks[:,:] = ' '
        self.marks[1,1] = '#'
        self.marks[2,1] = '#'
        self.marks[1,3] = '#'
        self.marks[1,4] = '#'
        self.marks[4,1] = '#'
        self.marks[4,2] = '#'
        self.marks[4,4] = '#'
        self.has_gold = False
        self.monster_pos = (0,0)
        self.player_pos = (4,0)
        self.gold_pos = (1,2)
        self.exit_pos = (3,0)
        self.max_moves = self.board_size * 2 + 1
        self.num_moves = 0
        self.depth_reached = 0
        self.move_when_grabbed_gold = -1
    
    def print_board(self): 
        """
        Prints the game board using current marks
        """ 
        # Prthe column numbers
        print(' ',end='')
        for j in range(self.board_size):
            print(" "+str(j+1), end='')        
        
        # Prthe rows with marks
        print("")
        for i in range(self.board_size):
            # Prthe line separating the row
            print(" ",end='')
            for j in range(self.board_size):
                print("--",end='')
            
            print("-")

            # Prthe row number
            print(i+1,end='')
            
            # Prthe marks on self row
            for j in range(self.board_size):
                if (i,j)==self.monster_pos:
                    print("|M",end='') 
                elif (i,j)==self.gold_pos and not self.has_gold:                    
                    print("|G",end='')
                elif (i,j)==self.player_pos:
                    print("|P",end='')                 
                else:
                    print("|"+self.marks[i][j],end='')
            
            print("|")
                
        
        # Prthe line separating the last row
        print(" ",end='')
        for j in range(self.board_size):
            print("--",end='')
        
        print("-")
    
    def make_move(self, action, player_move):
        """
        Makes the move for either player or monster
        """  
        assert action in self.get_actions(player_move)
        
        # Make the move
        if player_move:
            if action==self.UP:
                self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
            elif action==self.DOWN:
                self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
            elif action==self.LEFT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
            elif action==self.RIGHT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
            elif action==self.UP_BUILD:
                self.marks[self.player_pos[0]-1, self.player_pos[1]] = '#'
            elif action==self.DOWN_BUILD:
                self.marks[self.player_pos[0]+1, self.player_pos[1]] = '#'                
            elif action==self.LEFT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]-1] = '#'
            elif action==self.RIGHT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1]+1] = '#'
            self.num_moves = self.num_moves + 1
            # Check for gold co-location
            if not board.has_gold and board.player_pos==board.gold_pos:
                board.has_gold = True
                self.move_when_grabbed_gold = self.num_moves
        else:
            if action==self.UP:
                self.monster_pos = (self.monster_pos[0] - 1, self.monster_pos[1])
            elif action==self.DOWN:
                self.monster_pos = (self.monster_pos[0] + 1, self.monster_pos[1])
            elif action==self.LEFT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] - 1)
            elif action==self.RIGHT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] + 1)
    
    def game_won(self, player_move):
        """
        Determines whether a game winning condition exists for the player or monster
        """
        if player_move:
            if self.has_gold and self.player_pos == self.exit_pos:
                return True
            else:
                return False
        else:
            if self.num_moves == self.max_moves or self.monster_pos == self.player_pos:
                return True
            else:
                return False

    def get_actions(self, player_move):
        '''Generates a list of possible moves'''
        moves = []
        
        if player_move:
            if self.player_pos[0]>0 and self.marks[self.player_pos[0]-1, self.player_pos[1]]==' ':
                moves.append(self.UP)
                moves.append(self.UP_BUILD)
            if self.player_pos[0]<self.marks.shape[0]-1 and self.marks[self.player_pos[0]+1, self.player_pos[1]]==' ':
                moves.append(self.DOWN)
                moves.append(self.DOWN_BUILD)
            if self.player_pos[1]>0 and self.marks[self.player_pos[0], self.player_pos[1]-1]==' ':
                moves.append(self.LEFT)
                moves.append(self.LEFT_BUILD)
            if self.player_pos[1]<self.marks.shape[1]-1 and self.marks[self.player_pos[0], self.player_pos[1]+1]==' ':
                moves.append(self.RIGHT)
                moves.append(self.RIGHT_BUILD)
        else:
            if self.monster_pos[0]>0 and self.marks[self.monster_pos[0]-1, self.monster_pos[1]]==' ':
                moves.append(self.UP)
            if self.monster_pos[0]<self.marks.shape[0]-1 and self.marks[self.monster_pos[0]+1, self.monster_pos[1]]==' ':
                moves.append(self.DOWN)
            if self.monster_pos[1]>0 and self.marks[self.monster_pos[0], self.monster_pos[1]-1]==' ':
                moves.append(self.LEFT)
            if self.monster_pos[1]<self.marks.shape[1]-1 and self.marks[self.monster_pos[0], self.monster_pos[1]+1]==' ':
                moves.append(self.RIGHT)
            moves.append('') # stay move
                   
        return moves
    
    def no_more_moves(self, player_move):
        """
        Determines whether there are any moves left for player or monster
        """
        return len(self.get_actions(player_move))==0
    
    # this will be the state that the game is restored to after backtracking
    def save_current_state(self):
        state = {
            'player_pos': self.player_pos,
            'monster_pos': self.monster_pos,
            'has_gold': self.has_gold,
            'num_moves': self.num_moves,
            'move_when_grabbed_gold': self.move_when_grabbed_gold,
            'marks': self.marks.copy()
        }
        return state
    
    # reversing the move that was made
    def backtrack(self, state):
        self.player_pos = state['player_pos']
        self.monster_pos = state['monster_pos']
        self.has_gold = state['has_gold']
        self.num_moves = state['num_moves']
        self.move_when_grabbed_gold = state['move_when_grabbed_gold']
        self.marks[:, :] = state['marks']
    
    # Determining the move of the monster:
    def alpha_beta_search(self):
        value, move = self.max_value(-math.inf, math.inf, 0)
        return move
       
    # Evaluating the utility of moves:    
    def evaluate(self):
        
        if self.monster_pos == self.player_pos:
            return 10
        
        if self.player_pos == self.exit_pos and self.has_gold:
            return 10
        
        # heuristic score starts at 0. Higher score = better for monster
        score = 0
        
        # Manhattan distance for distance between monster and player
        monster_distance = abs(self.monster_pos[0] - self.player_pos[0]) + abs(self.monster_pos[1] - self.player_pos[1])
        score += (10 - monster_distance) # closer distance will be a higher score
        
        if self.has_gold == False:
            gold_distance = abs(self.gold_pos[0] - self.player_pos[0]) + abs(self.gold_pos[1] - self.player_pos[1])
            score += gold_distance # farther gold distance from player will be a higher score for the monster
        
        exit_distance = abs(self.exit_pos[0] - self.player_pos[0]) + abs(self.exit_pos[1] - self.player_pos[1])
        score += exit_distance # the monster also has a higher score if player if further from exit
        
        return score
            
    # Maximizing the monster's turn:        
    def max_value(self, alpha, beta, depth): 
        self.depth_reached = max(self.depth_reached, depth)

        # Checking if the game is terminated:
        if self.game_won(False) or self.no_more_moves(False) or depth == self.MAX_DEPTH:
            return self.evaluate(), None
            
        v = -math.inf
        best_action = None
        
        for action in self.get_actions(False):
            current_state = self.save_current_state()
            self.make_move(action, False)
            v2, move = self.min_value(alpha, beta, depth + 1)
            self.backtrack(current_state)
            
            if v2 > v:
                v = v2
                best_action = action
                
            alpha = max(alpha, v)
            
            # pruning:
            if v >= beta:
                self.num_pruned += 1
                break 
        
        return v, best_action    
            
            
    # Minimizing the player's turn:
    def min_value(self, alpha, beta, depth):
        self.depth_reached = max(self.depth_reached, depth)

        # Checking to see if the game is terminated:
        if self.game_won(True) or self.no_more_moves(True) or depth == self.MAX_DEPTH:
            return self.evaluate(), None
        
        v = math.inf
        
        for action in self.get_actions(True):
            current_state = self.save_current_state()
            self.make_move(action, True)
            v2, move = self.max_value(alpha, beta, depth + 1)
            self.backtrack(current_state)
            
            if v2 < v:
                v = v2
                
            beta = min(beta, v)
            
            # Pruning:
            if v <= alpha:
                self.num_pruned += 1
                break
            
        return v, None

    # TODO - self method should run alpha-beta search to determine the value of each move
    # Then make best move for the computer by placing the mark in the best spot
    def make_comp_move(self):
        t1 = time.perf_counter()
        self.depth_reached = 0
        self.num_pruned = 0
        # This code chooses a random computer move
        # COMMENT THIS OUT AFTER YOU IMPLEMENT ALPHA-BETA SEARCH
        '''
        possible_moves = self.get_actions(False)
        rand_move_index = random.randrange(len(possible_moves))
        self.make_move(possible_moves[rand_move_index], False)
        '''
        
        # TODO: Make AI move - UNCOMMENT THIS AFTER YOU IMPLEMENT ALPHA-BETA SEARCH
        best_action = self.alpha_beta_search()
        if best_action == None:
            print("GAME OVER")
            print("You Lost!")
            exit()

        self.make_move(best_action, False)

        te = time.perf_counter() - t1
        print('DEPTH REACHED: ', self.depth_reached)
        print('Number pruned due to a/b: ', self.num_pruned)
        print('COMP MOVE TIME =',round(te,2),'seconds')
    

###########################            
### Program starts here ###
###########################        

# Print out the header info
print("CLASS: Artificial Intelligence, Lewis University")
print("NAME: Savannah Stumpf")

# Define constants
LOST = 0
WON = 1 
 
#wrongInput = False
# Get the board size from the user
#board_size = int(input("Please enter the size of the board n (e.g. n=3,4,5,...): "))
board_size = 5
       
# Create the game board of the given size and print it
board = GenGameBoard(board_size)
board.print_board()  
        
# Start the game loop
while True:
    # *** Player's move ***        
    
    # Try to make the move and check if it was possible
    # If not possible get col,row inputs from player    
    print("Player's Move #", (board.num_moves+1))
    possible_moves = board.get_actions(True)
    move = input("Choose your move "+str(possible_moves)+": ")
    while move not in possible_moves:
        print("Not a valid move")
        move = input("Choose your move "+str(possible_moves)+": ")
    board.make_move(move, True)
    
    # Display the board
    board.print_board()
            
    # Check for ending condition
    # If game is over, check if player won and end the game
    if board.game_won(True):
        # Player won
        result = WON
        break
    elif board.no_more_moves(True):
        # No moves left -> lost
        result = LOST
        break
            
    # *** Computer's move ***
    board.make_comp_move()
    
    # Print out the board again
    board.print_board()    
    
    # Check for ending condition
    # If game is over, check if computer won and end the game
    if board.game_won(False):
        # Computer won
        result = LOST
        break
        
# Check the game result and print out the appropriate message
print("GAME OVER")
if result==WON:
    print("You Won!")            
elif result==LOST:
    print("You Lost!")
else: 
    print("It was a draw!")

