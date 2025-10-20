import pygame
import os


class FreewayRender:
    def __init__(self, cell_size=60):
        pygame.init()
        self.cell_size = cell_size
        self.width = 9 * cell_size  # 9 columns
        self.height = 10 * cell_size  # 10 rows
        self.assets_path = os.path.join(os.path.dirname(__file__), "assets", "freeway")
        self.sprites = {}
        self.load_sprites()
        self.font = pygame.font.Font(None, 36)

    def load_sprites(self):
        sprite_files = {
            "chicken": "chicken.png",
            "car_1": "car1.png",
            "car_2": "car2.png",
            "car_3": "car3.png",
            "car_4": "car4.png",
            "grey": "grey.png",
            "yellow": "yellow.png",
            "grass": "grass.png",
            "target": "map-pin.png",
            "hit": "hit.png",
        }

        for name, filename in sprite_files.items():
            filepath = os.path.join(self.assets_path, filename)
            if os.path.exists(filepath):
                sprite = pygame.image.load(filepath)
                if name.startswith("car_"):
                    length = int(name.split("_")[1])
                    sprite = pygame.transform.scale(
                        sprite, (self.cell_size * length, self.cell_size)
                    )
                else:
                    sprite = pygame.transform.scale(
                        sprite, (self.cell_size * 0.95, self.cell_size * 0.95)
                    )
                self.sprites[name] = sprite
            else:
                raise FileNotFoundError(
                    f"Sprite file {filename} not found in {self.assets_path}."
                )

    def render(self, env):
        surface = pygame.Surface((self.width, self.height))
        surface.fill((255, 255, 255))
        for i in range(10):  # rows
            for j in range(9):  # columns
                pos = (j * self.cell_size, i * self.cell_size)
                if i == 0 or i == 9:
                    surface.blit(self.sprites["grass"], pos)
                else:
                    if j == 4:
                        surface.blit(self.sprites["yellow"], pos)
                    else:
                        surface.blit(self.sprites["grey"], pos)
                    if j < 8:
                        line_x = (j + 1) * self.cell_size - 1
                        pygame.draw.line(
                            surface,
                            (255, 255, 255),
                            (line_x, i * self.cell_size),
                            (line_x, (i + 1) * self.cell_size),
                            1,
                        )
        for car in env.cars:
            x, y, timer, speed, length = car
            if x is None or speed is None:
                continue

            is_right = speed > 0
            car_y = y * self.cell_size

            if is_right:
                car_x = (x - length + 1) * self.cell_size
            else:
                car_x = x * self.cell_size

            if car_x + length * self.cell_size > 0 and car_x < self.width:
                sprite = self.get_vehicle_sprite(length, is_right)
                surface.blit(sprite, (car_x, car_y))

        player_x = 4 * self.cell_size
        player_y = env.pos * self.cell_size
        target_x = player_x
        target_y = 0
        surface.blit(self.sprites["target"], (target_x, target_y))
        surface.blit(self.sprites["chicken"], (player_x, player_y))
        self.draw_game_info(surface, env)

        return surface

    def get_vehicle_sprite(self, vehicle_length, is_right_direction):
        sprite_name = f"car_{vehicle_length}"
        base_sprite = self.sprites[sprite_name]

        if is_right_direction:
            return base_sprite
        else:
            return pygame.transform.flip(base_sprite, True, False)

    def draw_game_info(self, surface, env):
        info_text = f"Turn: {env.game_turn}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))

        text_rect = text_surface.get_rect()
        text_rect.topleft = (self.cell_size // 4, self.cell_size // 4)
        bg_rect = text_rect.inflate(10, 5)
        pygame.draw.rect(surface, (0, 0, 0), bg_rect)
        surface.blit(text_surface, text_rect)

        if env.terminal:
            if env.game_turn < 100:
                end_text = f"SUCCESS in {env.game_turn} turns!"
                color = (0, 255, 0)
            else:
                end_text = "GAME OVER"
                color = (255, 0, 0)

            end_surface = pygame.font.Font(None, 48).render(end_text, True, color)
            end_rect = end_surface.get_rect()
            end_rect.center = (self.width // 2, self.height // 2)

            overlay = pygame.Surface((self.width, self.height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            surface.blit(overlay, (0, 0))
            surface.blit(end_surface, end_rect)
