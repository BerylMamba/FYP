import pygame
import pygame_menu
from queue import PriorityQueue, deque

WIDTH = 800
HEIGHT = 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.init()
pygame.display.set_caption("Pathfinding Visualisation FYP")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for Nodes in row:
            Nodes.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def make_grid(rows, width):
    gap = width // rows
    grid = [[Nodes(i, j, gap, rows) for j in range(rows)] for i in range(rows)]
    return grid

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

class Nodes:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.colour = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.colour == RED

    def is_open(self):
        return self.colour == GREEN

    def is_barrier(self):
        return self.colour == BLACK

    def is_start(self):
        return self.colour == ORANGE

    def is_end(self):
        return self.colour == TURQUOISE

    def reset(self):
        self.colour = WHITE

    def make_start(self):
        self.colour = ORANGE

    def make_closed(self):
        self.colour = RED

    def make_open(self):
        self.colour = GREEN

    def make_barrier(self):
        self.colour = BLACK

    def make_end(self):
        self.colour = TURQUOISE

    def make_path(self):
        self.colour = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

def astar(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {Nodes: float("inf") for row in grid for Nodes in row}
    g_score[start] = 0
    f_score = {Nodes: float("inf") for row in grid for Nodes in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def bfs(draw, grid, start, end):
    queue = deque()
    queue.append(start)

    came_from = {}  # dictionary to store the node came from
    
    visited = {start} # set to keep track of visited nodes
    
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current = queue.popleft() # get the current node from the queue
        
        # iof the current node is the end node, reconstruct and draw the path
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        
        # check the neighbors of the current node
        for neighbor in current.neighbors:
            if neighbor not in visited:
                came_from[neighbor] = current
                visited.add(neighbor)
                queue.append(neighbor)
                neighbor.make_open()
        
        draw()
        
        if current != start:
            current.make_closed()
    

    return False

def dijkstras(draw, grid, start, end):
    count = 0
    
    open_set = PriorityQueue()

    open_set.put((0, count, start))

    came_from = {}

    distance = {Nodes: float("inf") for row in grid for Nodes in row}
    distance[start] = 0

    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
           
            temp_distance = distance[current] + 1
            
            if temp_distance < distance[neighbor]:
                came_from[neighbor] = current
                distance[neighbor] = temp_distance

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((distance[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()

    return False

def dfs(draw, grid, start, end):
    stack = [start]
    visited = {Nodes: False for row in grid for Nodes in row}
    came_from = {}
    
    while stack:
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                pygame.quit()

        current = stack.pop()

        if not visited[current]:
            visited[current] = True
            current.make_open()

            if current == end:
                reconstruct_path(came_from, end, draw)
                end.make_end()
                return True

            # prioritising the exploration towards the top-left
            neighbors = []
            # prioritising 'Up' and 'Left' directions to make sure that the movement heads towards top-left
            if current.row > 0 and not grid[current.row - 1][current.col].is_barrier():  # Up
                neighbors.append(grid[current.row - 1][current.col])
            if current.col > 0 and not grid[current.row][current.col - 1].is_barrier():  # Left
                neighbors.append(grid[current.row][current.col - 1])
            if current.row < len(grid) - 1 and not grid[current.row + 1][current.col].is_barrier():  # Down
                neighbors.append(grid[current.row + 1][current.col])
            if current.col < len(grid[0]) - 1 and not grid[current.row][current.col + 1].is_barrier():  # Right
                neighbors.append(grid[current.row][current.col + 1])

            for neighbor in reversed(neighbors):  # reverse to process Up and Left first
                if not visited[neighbor]:
                    stack.append(neighbor)
                    came_from[neighbor] = current

        draw()

        if current != start:
            current.make_closed()

    return False

def set_algorithm(algorithm_name, win, width, menu, algorithm_select):
    def run():
        ROWS = 50
        grid = make_grid(ROWS, width)
        start = end = None

        def draw_grid_and_handle_events():
            nonlocal start, end, grid
            draw(win, grid, ROWS, width)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True 
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width)
                    Nodes = grid[row][col]
                    if not start and Nodes != end:
                        start = Nodes
                        start.make_start()
                    elif not end and Nodes != start:
                        end = Nodes
                        end.make_end()
                    elif Nodes != end and Nodes != start:
                        Nodes.make_barrier()
                        
                elif pygame.mouse.get_pressed()[2]: # RIGHT
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None
     
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and start and end:
                        for row in grid:
                            for Nodes in row:
                                Nodes.update_neighbors(grid)
                        algorithm_select[algorithm_name](lambda: draw(win, grid, ROWS, width), grid, start, end)
                    if event.key == pygame.K_c:
                        start = None
                        end = None
                        grid = make_grid(ROWS, width)
                    if event.key == pygame.K_ESCAPE:  # Back to menu
                        menu.enable()
                        return True

        menu.disable()
        stop = False
        while not stop:
            stop = draw_grid_and_handle_events()
            pygame.display.update()

    return run

def main():
    pygame.init()
    menu = pygame_menu.Menu('Choose Pathfinding Algorithm', WIDTH, HEIGHT, theme=pygame_menu.themes.THEME_BLUE)

    algorithm_dict = {
        'A*': astar,
        'Breadth-First Search': bfs,
        'Dijkstra\'s': dijkstras,
        'Depth-First Search': dfs
    }

    # stop messing around with this, it isnt going to fix the errors
    def create_button_callback(algorithm_name):
        def button_callback():
            set_algorithm(algorithm_name, WIN, WIDTH, menu, algorithm_dict)()
        return button_callback

    for name in algorithm_dict:
        menu.add.button(name, create_button_callback(name))
    
    menu.add.label("", font_size=20)  # adds space between the options and exit
    
    menu.add.button('Quit', pygame_menu.events.EXIT)
    
    menu.add.label("", font_size=20) # adds space between exit and the instructions
    menu.add.label("", font_size=20)  

    info_text = "INSTRUCTIONS:\n" \
                "Left Click: Place nodes\n" \
                "Right Click: Remove nodes\n" \
                "Enter/Return: Start selected algorithm\n" \
                "C: Clear the grid\n" \
                "ESC: Go back to the menu"
    menu.add.label(info_text, max_char=-1, font_size=16)

    while True:
        if menu.is_enabled():
            menu.mainloop(WIN)
        pygame.display.flip()

if __name__ == "__main__":
    main()
