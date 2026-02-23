import pygame
import numpy as np
import random
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Game constants
GRID_SIZE = 20                # 20x20 grid
CELL_SIZE = 30                # Pixels per cell (reduced to fit more cells)
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

class CooperativeShooterGame:
    def __init__(self, render=True):
        self.render_enabled = render
        if render:
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Cooperative Shooter - Optimized (Free Movement + Any Direction Bullets)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 20)

        # Game state
        self.reset()

        # Q-learning parameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Action space
        # Pilot: 8-direction move + wait = 9 actions
        self.pilot_actions = 9
        # Gunner: 8-direction move + 4-direction shoot + wait = 13 actions
        self.gunner_actions = 13

        # Q-tables
        self.q_pilot = defaultdict(lambda: np.zeros(self.pilot_actions))
        self.q_gunner = defaultdict(lambda: np.zeros(self.gunner_actions))

    def reset(self):
        """Reset game state"""
        # Pilot initial position
        self.pilot_pos = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
        # Gunner initial position, not overlapping with pilot
        self.gunner_pos = self._random_empty_pos(self.pilot_pos)

        # Alien list [x, y, health]
        self.aliens = []
        for _ in range(3):
            self.spawn_alien()

        # Bullet list [x, y, dx, dy]
        self.bullets = []

        # Resources
        self.energy = 100      # Pilot energy
        self.ammo = 50         # Gunner ammo

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.alien_move_counter = 0
        self.old_alien_count = len(self.aliens)

    def _random_empty_pos(self, exclude_pos=None):
        """Generate a random position not colliding with the given one"""
        while True:
            pos = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            if exclude_pos is None or pos != exclude_pos:
                return pos

    def spawn_alien(self):
        """Spawn an alien at a random position, avoid overlapping with players"""
        while True:
            x = random.randint(0, GRID_SIZE-1)
            y = random.randint(0, GRID_SIZE-1)
            if [x, y] != self.pilot_pos and [x, y] != self.gunner_pos:
                self.aliens.append([x, y, 2])   # Health 2
                break

    def get_state_pilot(self):
        """Pilot state encoding: position + relative position to nearest alien + energy level"""
        if self.aliens:
            nearest = min(self.aliens, key=lambda a: abs(a[0]-self.pilot_pos[0]) + abs(a[1]-self.pilot_pos[1]))
            rel_x = np.clip(nearest[0] - self.pilot_pos[0], -3, 3) + 3   # 0..6
            rel_y = np.clip(nearest[1] - self.pilot_pos[1], -3, 3) + 3
        else:
            rel_x, rel_y = 3, 3
        # Discretize own position (0..19) compress to 0..4 to reduce state space
        px = self.pilot_pos[0] // 4   # 0..4
        py = self.pilot_pos[1] // 4
        energy_level = min(self.energy // 20, 4)
        return (px, py, rel_x, rel_y, energy_level)

    def get_state_gunner(self):
        """Gunner state encoding: position + relative alien pos + ammo level + relative pilot pos"""
        if self.aliens:
            nearest = min(self.aliens, key=lambda a: abs(a[0]-self.gunner_pos[0]) + abs(a[1]-self.gunner_pos[1]))
            rel_x = np.clip(nearest[0] - self.gunner_pos[0], -3, 3) + 3
            rel_y = np.clip(nearest[1] - self.gunner_pos[1], -3, 3) + 3
        else:
            rel_x, rel_y = 3, 3
        # Relative position to pilot (gunner needs to know where pilot is for cooperation)
        rel_pilot_x = np.clip(self.pilot_pos[0] - self.gunner_pos[0], -3, 3) + 3
        rel_pilot_y = np.clip(self.pilot_pos[1] - self.gunner_pos[1], -3, 3) + 3
        # Own position compressed
        gx = self.gunner_pos[0] // 4
        gy = self.gunner_pos[1] // 4
        ammo_level = min(self.ammo // 10, 4)
        return (gx, gy, rel_x, rel_y, rel_pilot_x, rel_pilot_y, ammo_level)

    def choose_action_pilot(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.pilot_actions-1)
        return np.argmax(self.q_pilot[state])

    def choose_action_gunner(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.gunner_actions-1)
        return np.argmax(self.q_gunner[state])

    def execute_action_pilot(self, action):
        """Pilot executes action: 0-7 move, 8 wait"""
        moves = [(-1,-1), (-1,0), (-1,1),
                 (0,-1),          (0,1),
                 (1,-1),  (1,0),  (1,1)]
        if action < 8:
            dx, dy = moves[action]
            new_x = np.clip(self.pilot_pos[0] + dx, 0, GRID_SIZE-1)
            new_y = np.clip(self.pilot_pos[1] + dy, 0, GRID_SIZE-1)
            # Cannot overlap with gunner
            if [new_x, new_y] != self.gunner_pos:
                self.pilot_pos = [new_x, new_y]
                self.energy -= 1
        else:  # wait
            self.energy = min(100, self.energy + 0.5)

    def execute_action_gunner(self, action):
        """Gunner executes action:
           0-7: move
           8-11: shoot (up, down, left, right)
           12: wait
        """
        # Move
        if action < 8:
            moves = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]
            dx, dy = moves[action]
            new_x = np.clip(self.gunner_pos[0] + dx, 0, GRID_SIZE-1)
            new_y = np.clip(self.gunner_pos[1] + dy, 0, GRID_SIZE-1)
            # Cannot overlap with pilot
            if [new_x, new_y] != self.pilot_pos:
                self.gunner_pos = [new_x, new_y]
                # Moving does not consume ammo, but might consume stamina? Not here
        # Shoot
        elif action < 12:
            if self.ammo > 0:
                # 8-direction shoot mapping: 8->up, 9->down, 10->left, 11->right
                dir_map = {8: (0,-1), 9: (0,1), 10: (-1,0), 11: (1,0)}
                dx, dy = dir_map[action]
                self.bullets.append([self.gunner_pos[0], self.gunner_pos[1], dx, dy])
                self.ammo -= 1
        # Wait
        else:
            pass   # do nothing

    def update_game(self):
        """Update game logic"""
        self.steps += 1

        # Aliens move (every 2 steps)
        self.alien_move_counter += 1
        if self.alien_move_counter >= 2:
            self.alien_move_counter = 0
            for alien in self.aliens:
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
                alien[0] = np.clip(alien[0] + dx, 0, GRID_SIZE-1)
                alien[1] = np.clip(alien[1] + dy, 0, GRID_SIZE-1)

        # Bullets move
        for bullet in self.bullets[:]:
            bullet[0] += bullet[2]
            bullet[1] += bullet[3]
            # Remove if out of bounds
            if bullet[0] < 0 or bullet[0] >= GRID_SIZE or bullet[1] < 0 or bullet[1] >= GRID_SIZE:
                self.bullets.remove(bullet)
                continue

            # Bullet hits alien
            hit = False
            for alien in self.aliens[:]:
                if bullet[0] == alien[0] and bullet[1] == alien[1]:
                    alien[2] -= 1
                    if alien[2] <= 0:
                        self.aliens.remove(alien)
                        self.score += 10
                    self.bullets.remove(bullet)
                    hit = True
                    break
            if hit:
                continue

        # Aliens attack pilot and gunner
        for alien in self.aliens:
            if alien[0] == self.pilot_pos[0] and alien[1] == self.pilot_pos[1]:
                self.energy -= 10
            if alien[0] == self.gunner_pos[0] and alien[1] == self.gunner_pos[1]:
                self.ammo -= 5   # Gunner loses ammo when collided

        # Game over conditions
        if self.energy <= 0 or self.ammo <= 0:
            self.game_over = True

        # Randomly spawn new aliens
        if len(self.aliens) < 5 and random.random() < 0.02:
            self.spawn_alien()

    def calculate_rewards(self, alien_count_before):
        """Calculate rewards based on change in alien count"""
        alien_killed = alien_count_before - len(self.aliens)
        reward_pilot = 0.1   # Base survival reward
        reward_gunner = 0.1

        if alien_killed > 0:
            # Kill reward: each gets 5 points, plus cooperation bonus
            reward_pilot += alien_killed * 5
            reward_gunner += alien_killed * 5
            # Additional cooperation bonus: each gets +2 if an alien was killed
            reward_pilot += 2
            reward_gunner += 2

        # Energy and ammo management
        if self.energy < 20:
            reward_pilot -= 0.5
        if self.ammo < 10:
            reward_gunner -= 0.5

        # Penalty for alien collisions is already reflected in resource loss, no extra penalty here

        return reward_pilot, reward_gunner

    def step(self, pilot_action, gunner_action):
        """Execute one step, return transition info"""
        # Record state before update
        old_state_pilot = self.get_state_pilot()
        old_state_gunner = self.get_state_gunner()
        alien_count_before = len(self.aliens)

        # Execute actions
        self.execute_action_pilot(pilot_action)
        self.execute_action_gunner(gunner_action)

        # Update game
        self.update_game()

        # Calculate rewards
        r_p, r_g = self.calculate_rewards(alien_count_before)

        # Get new states
        new_state_pilot = self.get_state_pilot()
        new_state_gunner = self.get_state_gunner()

        done = self.game_over

        return (old_state_pilot, old_state_gunner,
                pilot_action, gunner_action,
                r_p, r_g,
                new_state_pilot, new_state_gunner, done)

    def update_q(self, old_s_p, old_s_g, act_p, act_g, r_p, r_g, new_s_p, new_s_g, done):
        """Update both Q-tables"""
        # Pilot update
        old_q_p = self.q_pilot[old_s_p][act_p]
        if done:
            target_p = r_p
        else:
            next_max_p = np.max(self.q_pilot[new_s_p])
            target_p = r_p + self.gamma * next_max_p
        self.q_pilot[old_s_p][act_p] = (1 - self.alpha) * old_q_p + self.alpha * target_p

        # Gunner update
        old_q_g = self.q_gunner[old_s_g][act_g]
        if done:
            target_g = r_g
        else:
            next_max_g = np.max(self.q_gunner[new_s_g])
            target_g = r_g + self.gamma * next_max_g
        self.q_gunner[old_s_g][act_g] = (1 - self.alpha) * old_q_g + self.alpha * target_g

    def render(self):
        if not self.render_enabled:
            return
        self.screen.fill(BLACK)

        # Draw grid
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (40,40,40), (x,0), (x,WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, (40,40,40), (0,y), (WINDOW_SIZE,y))

        # Draw aliens
        for alien in self.aliens:
            rect = pygame.Rect(alien[0]*CELL_SIZE, alien[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, RED, rect)
            health_text = self.font.render(str(alien[2]), True, WHITE)
            self.screen.blit(health_text, (alien[0]*CELL_SIZE+5, alien[1]*CELL_SIZE+5))

        # Draw bullets
        for bullet in self.bullets:
            center = (bullet[0]*CELL_SIZE + CELL_SIZE//2, bullet[1]*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(self.screen, YELLOW, center, 4)

        # Draw pilot (green border square)
        pilot_rect = pygame.Rect(self.pilot_pos[0]*CELL_SIZE, self.pilot_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, GREEN, pilot_rect, 3)

        # Draw gunner (purple circle)
        gunner_center = (self.gunner_pos[0]*CELL_SIZE + CELL_SIZE//2, self.gunner_pos[1]*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(self.screen, PURPLE, gunner_center, 8)

        # Display info
        info = f"Score: {self.score}  Energy: {self.energy:.1f}  Ammo: {self.ammo}  Steps: {self.steps}"
        text = self.font.render(info, True, WHITE)
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def run_episode(self, max_steps=300):
        self.reset()
        total_r_p = 0
        total_r_g = 0
        for step in range(max_steps):
            if self.render_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None, None

            state_p = self.get_state_pilot()
            state_g = self.get_state_gunner()
            act_p = self.choose_action_pilot(state_p)
            act_g = self.choose_action_gunner(state_g)

            (old_p, old_g, act_p, act_g, r_p, r_g, new_p, new_g, done) = self.step(act_p, act_g)

            self.update_q(old_p, old_g, act_p, act_g, r_p, r_g, new_p, new_g, done)

            total_r_p += r_p
            total_r_g += r_g

            if self.render_enabled:
                self.render()
                pygame.time.delay(30)   # Slow down a bit for observation

            if done:
                break

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_r_p, total_r_g

def main():
    game = CooperativeShooterGame(render=True)
    episodes = 500
    for ep in range(episodes):
        total_p, total_g = game.run_episode(max_steps=300)
        if total_p is None:
            break
        if ep % 10 == 0:
            print(f"Episode {ep:4d} | Pilot reward: {total_p:6.1f} | Gunner reward: {total_g:6.1f} | "
                  f"Score: {game.score:3d} | Energy: {game.energy:.1f} | Ammo: {game.ammo} | Epsilon: {game.epsilon:.3f}")
    pygame.quit()

if __name__ == "__main__":
    main()