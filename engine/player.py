import numpy as np
from .map import Map

class Player:
    MAX_BOMB_RADIUS = 5
    MAX_BOMB_CAPACITY = 5
    STOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    PLACE_BOMB = 5
    
    def __init__(self, player_id, row, col):
        self.id = player_id
        self.x = row
        self.y = col
        self.alive = True
        # self.bomb_capacity = 1
        self.bombs_left = 1
        self.bomb_radius_bonus = 0
    
    def move(self, dx, dy, grid, players, bombs):
        if not self.alive:
            return
        
        new_x = self.x + dx
        new_y = self.y + dy
    
        if not (0 < new_x < grid.shape[0] - 1 and 0 < new_y < grid.shape[1] - 1):
            return
        if grid[new_x, new_y] in [Map.WALL, Map.BOX]:
            return
        if any(b.x == new_x and b.y == new_y for b in bombs):
            return
        
        # players can overlap
        self.x = new_x
        self.y = new_y
        
        # if two players move into the same cell to collect an item, they try to steal from each other, but eventually breaking it... :D
        if grid[self.x, self.y] in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY] and any(p.x == self.x and p.y == self.y and p.id != self.id for p in players):
            grid[self.x, self.y] = Map.GRASS
        
        if grid[self.x, self.y] == Map.ITEM_RADIUS:
            self.bomb_radius_bonus = min(self.bomb_radius_bonus + 1, self.MAX_BOMB_RADIUS - 1)
            grid[self.x, self.y] = Map.GRASS
        elif grid[self.x, self.y] == Map.ITEM_CAPACITY:
            self.bombs_left = min(self.bombs_left + 1, self.MAX_BOMB_CAPACITY)
            grid[self.x, self.y] = Map.GRASS