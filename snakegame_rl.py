import numpy as np
import time


LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


class SnakeGame:

    def __init__(self, n=20):
        self.convert_coordinate_to_label = lambda x: x[0] * n + x[1]
        self.gamma = 0.75
        self.n = n
        self.board = np.full((n, n), " ")

        self.left_coordinates = [(i, 0) for i in range(self.n)]
        self.right_coordinates = [(i, 9) for i in range(self.n)]
        self.up_coordinates = [(0, i) for i in range(self.n)]
        self.down_coordinates = [(9, i) for i in range(self.n)]

        all_coordinates = []
        all_shapes = []
        for coordinate in self.left_coordinates:
            all_coordinates.append(coordinate)
            all_shapes.append(">")
        for coordinate in self.right_coordinates:
            all_coordinates.append(coordinate)
            all_shapes.append("<")
        for coordinate in self.up_coordinates:
            all_coordinates.append(coordinate)
            all_shapes.append("v")
        for coordinate in self.down_coordinates:
            all_coordinates.append(coordinate)
            all_shapes.append("^")

        rand_start_idx = np.random.randint(0, len(all_coordinates))
        self.snake = [all_coordinates[rand_start_idx]]
        self.place_snake_on_board(all_shapes[rand_start_idx])
        self.food_coordinate = self.get_random_food_coordinate()
        self.board[self.food_coordinate] = "o"
        self.consume = False  # if food is consumed by snake, then True, else False . By default , False as we already placed a food at start.

    @property
    def snake_len(self):
        return len(self.snake)

    @property
    def snake_head(self):
        return self.snake[-1]

    def place_snake_on_board(self, head):
        for coordinate in self.snake[:-1]:
            self.board[coordinate] = "-"
        self.board[self.snake[-1]] = head

    def erase_previous_snake(self):
        for coordinate in self.snake:
            self.board[coordinate] = " "

    def get_random_food_coordinate(self):
        available_coordinates = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == " ":
                    available_coordinates.append((i, j))
        rand_idx = np.random.choice(list(range(len(available_coordinates))), 1)[0]
        return available_coordinates[rand_idx]

    def get_next_coordinate(self):
        available_coordinates = []
        heads = []
        x, y = self.snake_head
        if self.board[self.snake_head] == "v":
            if y + 1 < self.n and self.board[x, y + 1] in [" ", "o"]:
                available_coordinates.append((x, y + 1))
                heads.append(">")
            if y - 1 >= 0 and self.board[x, y - 1] in [" ", "o"]:
                available_coordinates.append((x, y - 1))
                heads.append("<")
            if x + 1 < self.n and self.board[x + 1, y] in [" ", "o"]:
                available_coordinates.append((x + 1, y))
                heads.append("v")
        elif self.board[self.snake_head] == "^":
            if y + 1 < self.n and self.board[x, y + 1] in [" ", "o"]:
                available_coordinates.append((x, y + 1))
                heads.append(">")
            if y - 1 >= 0 and self.board[x, y - 1] in [" ", "o"]:
                available_coordinates.append((x, y - 1))
                heads.append("<")
            if x - 1 >= 0 and self.board[x - 1, y] in [" ", "o"]:
                available_coordinates.append((x - 1, y))
                heads.append("^")
        elif self.board[self.snake_head] == "<":
            if y - 1 >= 0 and self.board[x, y - 1] in [" ", "o"]:
                available_coordinates.append((x, y - 1))
                heads.append("<")
            if x + 1 < self.n and self.board[x + 1, y] in [" ", "o"]:
                available_coordinates.append((x + 1, y))
                heads.append("v")
            if x - 1 >= 0 and self.board[x - 1, y] in [" ", "o"]:
                available_coordinates.append((x - 1, y))
                heads.append("^")
        elif self.board[self.snake_head] == ">":
            if y + 1 < self.n and self.board[x, y + 1] in [" ", "o"]:
                available_coordinates.append((x, y + 1))
                heads.append(">")
            if x + 1 < self.n and self.board[x + 1, y] in [" ", "o"]:
                available_coordinates.append((x + 1, y))
                heads.append("v")
            if x - 1 >= 0 and self.board[x - 1, y] in [" ", "o"]:
                available_coordinates.append((x - 1, y))
                heads.append("^")
        if len(available_coordinates) > 0:
            rand_idx = np.random.randint(0, len(available_coordinates), 1)[0]
            return available_coordinates[rand_idx], heads[rand_idx]
        else:
            return None, None

    def update_reward_matrix(self, M, next_coordinate, reward_matrix):
        max_value = np.max(reward_matrix[self.convert_coordinate_to_label(next_coordinate), :,
                           self.convert_coordinate_to_label(self.food_coordinate)])
        cur_x, cur_y = self.snake_head
        next_x, next_y = next_coordinate
        if cur_x == next_x:
            if cur_y < next_y:
                action = 1
            else:
                action = 0
        else:
            if cur_x < next_x:
                action = 3
            else:
                action = 2
        reward_matrix[self.convert_coordinate_to_label(self.snake_head), action, self.convert_coordinate_to_label(
            self.food_coordinate)] = M[self.convert_coordinate_to_label(
            self.snake_head), action] + self.gamma * max_value
        # print(reward)
        if np.max(reward_matrix) > 0:
            return np.sum(reward_matrix / np.max(reward_matrix) * 100)
        else:
            return 0

    def train(self, reward_matrix=None):
        if self.consume:
            self.food_coordinate = self.get_random_food_coordinate()
            self.board[self.food_coordinate] = "o"
            self.consume = False
        M = self.initialize_point_matrix()

        next_coordinate, next_head = self.get_next_coordinate()
        if next_coordinate is None:
            # print("No next coordinate")
            return -1
        else:
            score = self.update_reward_matrix(M, next_coordinate, reward_matrix)
            if self.board[next_coordinate] == "o":
                self.erase_previous_snake()
                self.snake.append(next_coordinate)
                # self.snake_len += 1
                # self.snake_head = next_coordinate
                self.place_snake_on_board(next_head)
                self.consume = True
            elif self.board[next_coordinate] == " ":
                self.erase_previous_snake()
                self.snake.append(next_coordinate)
                self.snake = self.snake[1:]
                # self.snake_head = next_coordinate
                self.place_snake_on_board(next_head)
            elif self.board[next_coordinate] == "-":
                # print("SNAKE KILLED ITSELF")
                return -1
            else:
                # print("OTHER REASON")
                return -1
            return score

    def predict(self, reward_matrix):
        if self.consume:
            self.food_coordinate = self.get_random_food_coordinate()
            self.board[self.food_coordinate] = "o"
            self.consume = False
        M = self.initialize_mask_matrix()

        next_coordinate, next_head = self.get_next_coordinate_from_reward_matrix(reward_matrix, M)
        if next_coordinate is None:
            # print("No next coordinate")
            return -1
        elif 0 <= next_coordinate[0] < self.board.shape[0] and 0 <= next_coordinate[1] < self.board.shape[1]:
            if self.board[next_coordinate] == "o":
                self.erase_previous_snake()
                self.snake.append(next_coordinate)
                # self.snake_len += 1
                # self.snake_head = next_coordinate
                self.place_snake_on_board(next_head)
                self.consume = True
            elif self.board[next_coordinate] == " ":
                self.erase_previous_snake()
                self.snake.append(next_coordinate)
                self.snake = self.snake[1:]
                # self.snake_head = next_coordinate
                self.place_snake_on_board(next_head)
            elif self.board[next_coordinate] == "-":
                print("SNAKE KILLED ITSELF")
                return -1
            else:
                print("OTHER REASON", self.board[next_coordinate])
                return -1
            return 0
        else:
            print("SNAKE HIT THE WALL")
            return -1

    def get_next_coordinate_from_reward_matrix(self, reward_matrix, mask_matrix):
        new_reward_matrix = reward_matrix[:, :, self.convert_coordinate_to_label(self.food_coordinate)] + mask_matrix
        cur_x, cur_y = self.snake_head
        print(mask_matrix[self.convert_coordinate_to_label(self.snake_head), :])
        print(reward_matrix[self.convert_coordinate_to_label(self.snake_head), :, self.convert_coordinate_to_label(self.food_coordinate)])
        action = np.where(new_reward_matrix[self.convert_coordinate_to_label(self.snake_head), :] == np.max(
            new_reward_matrix[self.convert_coordinate_to_label(self.snake_head), :]))[0]
        if action.shape[0] > 1:
            action = int(np.random.choice(action, size=1))
        else:
            action = int(action)
        if action == 0:
            return (cur_x, cur_y - 1), "<"
        elif action == 1:
            return (cur_x, cur_y + 1), ">"
        elif action == 2:
            return (cur_x - 1, cur_y), "^"
        else:
            return (cur_x + 1, cur_y), "v"

    def start(self, reward_matrix):
        while self.snake_len < self.n * self.n:
            board_string = self.print_snake_board()
            print(board_string)
            check = self.predict(reward_matrix)
            if check == -1:
                print("GAME OVER!!! SNAKE CRASHED WITH ITSELF.")
                break
            time.sleep(0.1)
            for i in range(self.n + 4 + 2):
                print(LINE_UP, end=LINE_CLEAR)
                time.sleep(0.01)

    def print_snake_board(self):
        print_string = ""
        for i in range(-1, self.board.shape[0] + 1):
            for j in range(-1, self.board.shape[1] + 1):
                if i == self.board.shape[0] or i == -1 or j == self.board.shape[1] or j == -1:
                    print_string += "*  "
                    # print("*", sep="", end="  ")
                else:
                    print_string += self.board[i, j] + "  "
                    # print(self.board[i, j], sep="", end="  ")
            print_string += "\n"
        print_string += "Score: " + str(self.snake_len)
        print_string += "\n"
        return print_string

    @staticmethod
    def print_snake_board_alone(board):
        print_string = ""
        for i in range(-1, board.shape[0] + 1):
            for j in range(-1, board.shape[1] + 1):
                if i == board.shape[0] or i == -1 or j == board.shape[1] or j == -1:
                    print_string += "*  "
                    # print("*", sep="", end="  ")
                else:
                    print_string += board[i, j] + "  "
                    # print(self.board[i, j], sep="", end="  ")
            print_string += "\n"
        # print_string += "Score: " + str(snake_len)
        print_string += "\n"
        return print_string

    def initialize_point_matrix(self):
        # 4 means 0=left(j-1), 1=right(j+1), 2=up(i-1), 3=down(i+1)
        M = np.zeros((self.board.shape[0] * self.board.shape[1], 4))
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                # if self.board[i, j] == " ":
                if j + 1 < self.n:
                    if self.board[i, j + 1] == "o":
                        M[self.convert_coordinate_to_label((i, j)), 1] = 100
                else:
                    M[self.convert_coordinate_to_label((i, j)), 1] = -1000
                if j - 1 >= 0:
                    if self.board[i, j - 1] == "o":
                        M[self.convert_coordinate_to_label((i, j)), 0] = 100
                else:
                    M[self.convert_coordinate_to_label((i, j)), 0] = -1000
                if i + 1 < self.n:
                    if self.board[i + 1, j] == "o":
                        M[self.convert_coordinate_to_label((i, j)), 3] = 100
                else:
                    M[self.convert_coordinate_to_label((i, j)), 3] = -1000
                if i - 1 >= 0:
                    if self.board[i - 1, j] == "o":
                        M[self.convert_coordinate_to_label((i, j)), 2] = 100
                else:
                    M[self.convert_coordinate_to_label((i, j)), 2] = -1000
        return M

    def initialize_mask_matrix(self):
        M = np.zeros((self.board.shape[0] * self.board.shape[1], 4))
        i, j = self.snake_head
        if j+1 >= self.n or self.board[i, j+1] == "-" or self.board[i, j] == "<":
            M[self.convert_coordinate_to_label((i, j)), 1] = -np.inf
        else:
            point = (i, j+1)
            if self.board[point] == " ":
                tmp_snake = self.snake[1:] + [point]
            elif self.board[point] == "o":
                tmp_snake = self.snake + [point]
            tmp_board = np.full((self.n, self.n), " ")
            for coordinate in tmp_snake[:-1]:
                tmp_board[coordinate] = "-"
            tmp_board[tmp_snake[-1]] = ">"
            tmp_cluster_dict = SnakeGame.create_cluster_dict(tmp_board)
            for cluster, points in tmp_cluster_dict.items():
                if point in points:
                    M[self.convert_coordinate_to_label((i, j)), 1] = 1000 * (
                                len(points) / (self.board.shape[0] * self.board.shape[1]))
                    break
            del tmp_board, tmp_snake, tmp_cluster_dict
        if j-1 < 0 or self.board[i, j-1] == "-" or self.board[i, j] == ">":
            M[self.convert_coordinate_to_label((i, j)), 0] = -np.inf
        else:
            point = (i, j-1)
            if self.board[point] == " ":
                tmp_snake = self.snake[1:] + [point]
            elif self.board[point] == "o":
                tmp_snake = self.snake + [point]
            tmp_board = np.full((self.n, self.n), " ")
            for coordinate in tmp_snake[:-1]:
                tmp_board[coordinate] = "-"
            tmp_board[tmp_snake[-1]] = "<"
            tmp_cluster_dict = SnakeGame.create_cluster_dict(tmp_board)
            for cluster, points in tmp_cluster_dict.items():
                if point in points:
                    M[self.convert_coordinate_to_label((i, j)), 0] = 1000 * (
                                len(points) / (self.board.shape[0] * self.board.shape[1]))
                    break
            del tmp_board, tmp_snake, tmp_cluster_dict
        if i+1 >= self.n or self.board[i+1, j] == "-" or self.board[i, j] == "^":
            M[self.convert_coordinate_to_label((i, j)), 3] = -np.inf
        else:
            point = (i+1, j)
            if self.board[point] == " ":
                tmp_snake = self.snake[1:] + [point]
            elif self.board[point] == "o":
                tmp_snake = self.snake + [point]
            tmp_board = np.full((self.n, self.n), " ")
            for coordinate in tmp_snake[:-1]:
                tmp_board[coordinate] = "-"
            tmp_board[tmp_snake[-1]] = "v"
            tmp_cluster_dict = SnakeGame.create_cluster_dict(tmp_board)
            for cluster, points in tmp_cluster_dict.items():
                if point in points:
                    M[self.convert_coordinate_to_label((i, j)), 3] = 1000 * (
                                len(points) / (self.board.shape[0] * self.board.shape[1]))
                    break
            del tmp_board, tmp_snake, tmp_cluster_dict
        if i-1 < 0 or self.board[i-1, j] == "-" or self.board[i, j] == "v":
            M[self.convert_coordinate_to_label((i, j)), 2] = -np.inf
        else:
            point = (i-1, j)
            if self.board[point] == " ":
                tmp_snake = self.snake[1:] + [point]
            elif self.board[point] == "o":
                tmp_snake = self.snake + [point]
            tmp_board = np.full((self.n, self.n), " ")
            for coordinate in tmp_snake[:-1]:
                tmp_board[coordinate] = "-"
            tmp_board[tmp_snake[-1]] = "^"
            tmp_cluster_dict = SnakeGame.create_cluster_dict(tmp_board)
            for cluster, points in tmp_cluster_dict.items():
                if point in points:
                    M[self.convert_coordinate_to_label((i, j)), 2] = 1000 * (
                                len(points) / (self.board.shape[0] * self.board.shape[1]))
                    break
            del tmp_board, tmp_snake, tmp_cluster_dict
        return M

    @staticmethod
    def cluster(matrix, cluster_matrix, start, cluster_id, cluster_points):
        x, y = start
        cluster_matrix[x, y] = cluster_id
        cluster_points.remove(start)
        if x + 1 < cluster_matrix.shape[0] and cluster_matrix[x + 1, y] == 0:
            SnakeGame.cluster(matrix, cluster_matrix, (x + 1, y), cluster_id, cluster_points)
        if x - 1 >= 0 and cluster_matrix[x - 1, y] == 0:
            SnakeGame.cluster(matrix, cluster_matrix, (x - 1, y), cluster_id, cluster_points)
        if y + 1 < cluster_matrix.shape[1] and cluster_matrix[x, y + 1] == 0:
            SnakeGame.cluster(matrix, cluster_matrix, (x, y + 1), cluster_id, cluster_points)
        if y - 1 >= 0 and cluster_matrix[x, y - 1] == 0:
            SnakeGame.cluster(matrix, cluster_matrix, (x, y - 1), cluster_id, cluster_points)

    @staticmethod
    def create_cluster_dict(matrix):
        cluster_matrix = np.zeros(matrix.shape, dtype=np.int32)
        cluster_points = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] in ["-"]:
                    cluster_matrix[i][j] = -1
                else:
                    cluster_points.append((i, j))
        cluster_id = 0
        while len(cluster_points) > 0:
            start = cluster_points[np.random.randint(0, len(cluster_points))]
            cluster_id += 1
            SnakeGame.cluster(matrix, cluster_matrix, start, cluster_id, cluster_points)
        cluster_dict = {}
        for i in range(1, cluster_id + 1):
            cluster_dict[i] = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if cluster_matrix[i, j] > 0:
                    cluster_dict[int(cluster_matrix[i, j])].append((i, j))
        return cluster_dict



