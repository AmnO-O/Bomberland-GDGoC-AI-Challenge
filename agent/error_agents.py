# simulates agents that violate the rules

class TimeOutAgent:
    def __init__(self):
        pass
    
    def act(self, obs):
        while True:
            pass
        
class InvalidActionAgent:
    def __init__(self):
        pass
    
    def act(self, obs):
        return 6

class NoBombAgent:
    def __init__(self):
        pass
    
    def act(self, obs):
        return 5

# class WrongDeviceAgent:
#     def __init__(self):
#         pass
    
#     def act(self, obs):
#         import torch
#         x = torch.tensor([1.0], device='cuda')
#         return 0