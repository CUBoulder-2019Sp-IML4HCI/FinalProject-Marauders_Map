import pygame
import asyncio

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
(width, height) = (600, 600)


class Visualizer():
    def __init__(self, *args, **kwargs):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TUFF")
        self.screen.fill(WHITE)
        pygame.display.update()
        self.events = {}
    
    async def main_viz_loop(self):
        running = True
        while running:
            ev = pygame.event.get()

            for event in ev:
                if event.type == pygame.MOUSEBUTTONUP:
                    self.drawCircle()
                    pygame.display.update()

                if event.type == pygame.QUIT:
                    running = False
        return True
    
    def getPos(self):
        #unnecessary now
        pos = pygame.mouse.get_pos()
        return (pos)
    
    def drawCircle(self,pos,name=None):
        '''
        @param : pos : tuple representing x and y
        '''
        #todo : selecting color based on name
        pygame.draw.circle(self.screen, BLUE, pos, 5)
        pygame.display.update()
        

