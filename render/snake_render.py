import pygame
import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from envs.minatar.environment import Environment
from envs.minatar.environments.snake import Env

class SnakeRenderer:
    def __init__(self, cell_size=60, assets_path="render/snake"):
        pygame.init()
        self.cell_size = cell_size
        self.assets_path = assets_path
        self.width = 8 * cell_size
        self.height = 8 * cell_size

        self.sprites = {}
        self.snake_sprites = {}
        self.load_sprites()
        self.font = pygame.font.Font(None, 36)


    def load_sprites(self):
        sprite_files = {
            'apple': 'apple.png',
            'wall': 'brick-wall.png',
            'obstacle': 'brick-wall.png',
            'head': 'head.png',
            'thinking': 'thinking.png',
            'idea': 'idea.png',
        }

        for name, filename in sprite_files.items():
            filepath = os.path.join(self.assets_path, filename)
            if name == 'head':
                self.sprites['head_down'] = pygame.image.load(filepath)
                self.sprites['head_down'] = pygame.transform.scale(self.sprites['head_down'], (self.cell_size, self.cell_size))
                self.sprites['head_up'] = pygame.transform.rotate(self.sprites['head_down'], 180)
                self.sprites['head_left'] = pygame.transform.rotate(self.sprites['head_down'], -90)
                self.sprites['head_right'] = pygame.transform.rotate(self.sprites['head_down'], 90)
            elif name in ['thinking', 'idea']:
                if os.path.exists(filepath):
                    sprite = pygame.image.load(filepath)
                    sprite = pygame.transform.scale(sprite, (int(self.cell_size * 0.8), int(self.cell_size * 0.8)))
                    self.sprites[name] = sprite
                else:
                    raise FileNotFoundError(f"Sprite file {filename} not found in {self.assets_path}. Please ensure the file exists.")
            elif os.path.exists(filepath):
                sprite = pygame.image.load(filepath)
                sprite = pygame.transform.scale(sprite, (self.cell_size, self.cell_size))
                self.sprites[name] = sprite
            else:
              raise FileNotFoundError(f"Sprite file {filename} not found in {self.assets_path}. Please ensure the file exists.")
        print(f"Loaded sprites: {list(self.sprites.keys())}")


        snake_path = os.path.join(self.assets_path, 'snake.png')
        if os.path.exists(snake_path):
            snake_sheet = pygame.image.load(snake_path)
            sheet_width, sheet_height = snake_sheet.get_size()
            sprite_width = sheet_width // 2
            sprite_height = sheet_height // 2

            head_rect = pygame.Rect(0, 0, sprite_width, sprite_height)
            head_sprite = snake_sheet.subsurface(head_rect)
            head_sprite = pygame.transform.scale(head_sprite, (self.cell_size, self.cell_size))

            straight_rect = pygame.Rect(sprite_width, 0, sprite_width, sprite_height)
            straight_sprite = snake_sheet.subsurface(straight_rect)
            straight_sprite = pygame.transform.scale(straight_sprite, (self.cell_size, self.cell_size))

            tail_rect = pygame.Rect(0, sprite_height, sprite_width, sprite_height)
            tail_sprite = snake_sheet.subsurface(tail_rect)
            tail_sprite = pygame.transform.scale(tail_sprite, (self.cell_size, self.cell_size))

            turn_rect = pygame.Rect(sprite_width, sprite_height, sprite_width, sprite_height)
            turn_sprite = snake_sheet.subsurface(turn_rect)
            turn_sprite = pygame.transform.scale(turn_sprite, (self.cell_size, self.cell_size))
            self.snake_sprites['head_up'] = head_sprite
            self.snake_sprites['head_right'] = pygame.transform.rotate(head_sprite, -90)
            self.snake_sprites['head_down'] = pygame.transform.rotate(head_sprite, 180)
            self.snake_sprites['head_left'] = pygame.transform.rotate(head_sprite, 90)

            self.snake_sprites['tail_left'] = tail_sprite
            self.snake_sprites['tail_up'] = pygame.transform.rotate(tail_sprite, -90)  
            self.snake_sprites['tail_right'] = pygame.transform.rotate(tail_sprite, 180)
            self.snake_sprites['tail_down'] = pygame.transform.rotate(tail_sprite, 90)
            self.snake_sprites['straight_vertical'] = straight_sprite
            self.snake_sprites['straight_horizontal'] = pygame.transform.rotate(straight_sprite, 90)
            self.snake_sprites['turn_up_left'] = turn_sprite
            self.snake_sprites['turn_up_right'] = pygame.transform.rotate(turn_sprite, -90)
            self.snake_sprites['turn_down_right'] = pygame.transform.rotate(turn_sprite, 180)
            self.snake_sprites['turn_down_left'] = pygame.transform.rotate(turn_sprite, 90)
        else:
            raise FileNotFoundError(f"Snake sprite sheet not found at {snake_path}. Please ensure the file exists.")

    def get_snake_body_sprite(self, prev_pos, curr_pos, next_pos, board_size):
        if prev_pos is None or next_pos is None:
            return self.snake_sprites['straight_horizontal']

        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        next_x, next_y = next_pos
        from_dir = (prev_x - curr_x, prev_y - curr_y)
        to_dir = (next_x - curr_x, next_y - curr_y)

        if from_dir[0] == 0 and to_dir[0] == 0:  # 垂直
            return self.snake_sprites['straight_vertical']
        elif from_dir[1] == 0 and to_dir[1] == 0:  # 水平
            return self.snake_sprites['straight_horizontal']

        dirs = sorted([from_dir, to_dir])
        if dirs == [(-1, 0), (0, 1)]:
            return self.snake_sprites['turn_up_left']
        elif dirs == [(0, 1), (1, 0)]:
            return self.snake_sprites['turn_up_right']
        elif dirs == [(0, -1), (1, 0)]:
            return self.snake_sprites['turn_down_right']
        elif dirs == [(-1, 0), (0, -1)]:
            return self.snake_sprites['turn_down_left']

        return self.snake_sprites['straight_horizontal']

    def get_snake_tail_sprite(self, prev_pos, curr_pos):
        if prev_pos is None:
            return self.snake_sprites['tail_left']

        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos

        direction = (curr_x - prev_x, curr_y - prev_y)

        direction_map = {
            (1, 0): 'tail_right',
            (-1, 0): 'tail_left',
            (0, 1): 'tail_up',
            (0, -1): 'tail_down'
        }

        return self.snake_sprites.get(direction_map.get(direction, 'tail_left'),
                                     self.snake_sprites['tail_left'])

    def render(self, env : Env, show_thinking=None):
        size = env.B * self.cell_size
        surface = pygame.Surface((size, size))
        surface.fill((0, 51, 51))
        for i in range(env.B):
            for j in range(env.B):
                pos = pygame.Rect(j * self.cell_size, (env.B - 1 - i) * self.cell_size,
                                 self.cell_size, self.cell_size)
                if i == 0 or i == env.B - 1 or j == 0 or j == env.B - 1:
                    continue
                if (i + j) % 2 == 0:
                    surface.fill((0, 77, 77), pos)
                else:
                    surface.fill((0, 102, 102), pos)

        for (x, y) in env.obstacle:
            pos = (x * self.cell_size, (env.B-1-y) * self.cell_size)
            surface.blit(self.sprites['obstacle'], pos)

        for (x, y) in env.food:
            pos = (x * self.cell_size, (env.B-1-y) * self.cell_size)
            if env.food_attributes[x][y][1] > 0:
                surface.blit(self.sprites['apple'], pos)
                life = env.food_attributes[x][y][0]
                self.draw_life_bar(surface, life, 12, pos)

        for i, (x, y) in enumerate(env.snake):
            pos = (x * self.cell_size, (env.B-1-y) * self.cell_size)
            if i == len(env.snake) - 1:
                direction_map = {'R': 'right', 'D': 'down', 'L': 'left', 'U': 'up'}
                if len(env.snake) == 1:
                    head_sprite = self.sprites[f'head_{direction_map.get(env.dir, "up")}']
                else:
                    head_sprite = self.snake_sprites[f'head_{direction_map.get(env.dir, "up")}']
                surface.blit(head_sprite, pos)
            elif i == 0:
                prev_pos = env.snake[1] if len(env.snake) > 1 else None
                tail_sprite = self.get_snake_tail_sprite(prev_pos, (x, y))
                surface.blit(tail_sprite, pos)
            else:
                prev_pos = env.snake[i+1] if i < len(env.snake) - 1 else None
                next_pos = env.snake[i-1] if i > 0 else None
                body_sprite = self.get_snake_body_sprite(prev_pos, (x, y), next_pos, env.B)
                surface.blit(body_sprite, pos)

        if show_thinking is not None and len(env.snake) > 0:
            head_x, head_y = env.snake[-1]
            head_pos_x = head_x * self.cell_size
            head_pos_y = (env.B - 1 - head_y) * self.cell_size

            thinking_offset_x = self.cell_size * 0.7
            thinking_offset_y = -self.cell_size * 0.5
            thinking_x = head_pos_x + thinking_offset_x
            thinking_y = head_pos_y + thinking_offset_y

            thinking_x = max(0, min(thinking_x, size - self.cell_size * 0.6))
            thinking_y = max(0, min(thinking_y, size - self.cell_size * 0.6))

            if show_thinking:
                surface.blit(self.sprites['thinking'], (thinking_x, thinking_y))
            else:
                surface.blit(self.sprites['idea'], (thinking_x, thinking_y))

        self.draw_game_info(surface, env)

        crop_margin = self.cell_size // 4 * 3
        cropped_size = size - 2 * crop_margin
        cropped_surface = pygame.Surface((cropped_size, cropped_size))

        crop_rect = pygame.Rect(crop_margin, crop_margin, cropped_size, cropped_size)
        cropped_surface.blit(surface, (0, 0), crop_rect)
        return cropped_surface

    def draw_life_bar(self, surface, life, max_life, pos):
        bar_width = self.cell_size - 4
        bar_height = 6
        bar_x = pos[0] + 2
        bar_y = pos[1] + 10  

        background_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(surface, (100, 0, 0), background_rect)

        life_ratio = max(0, life / max_life)

        if life_ratio > 0:
            life_width = int(bar_width * life_ratio)
            life_rect = pygame.Rect(bar_x, bar_y, life_width, bar_height)

            if life_ratio > 0.6:
                color = (0, 255, 0)
            elif life_ratio > 0.3:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)

            pygame.draw.rect(surface, color, life_rect)

        pygame.draw.rect(surface, (255, 255, 255), background_rect, 1)

    def draw_game_info(self, surface, env):
        info_text = f"Turn: {env.game_turn}, Score: {env.reward}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))

        text_rect = text_surface.get_rect()
        text_rect.topleft = (self.cell_size, self.cell_size)
        bg_rect = text_rect.inflate(10, 5)
        pygame.draw.rect(surface, (0, 0, 0), bg_rect)
        surface.blit(text_surface, text_rect)

        if env.terminal:
            if env.game_turn < 100:
                end_text = f"GAME OVER! REWARD {env.reward}"
                color = (255, 0, 0)
            else:
                end_text = f"REWARD {env.reward}"
                color = (0, 255, 0)

            end_surface = self.font.render(end_text, True, color)
            end_rect = end_surface.get_rect()
            end_rect.center = (self.width // 2, self.height // 2)

            overlay = pygame.Surface((self.width, self.height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            surface.blit(overlay, (0, 0))
            surface.blit(end_surface, end_rect)


    def surface_to_pil(self, surface):
        data = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), data)
        return img
