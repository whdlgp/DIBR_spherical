import cv2
import numpy as np
import pygame


class InteractiveViewer:
    def __init__(self, render_function, screen_size, fps=30, move_speed=600.0, mouse_sensitivity=0.002, max_pitch=np.deg2rad(85.0)):
        # Rendering parameters
        self.render_function = render_function
        self.screen_size = screen_size
        self.fps = fps

        # Camera control parameters
        self.move_speed = move_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.max_pitch = max_pitch
        self.camera_position = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0

    def process_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        return running

    def update_camera_rotation(self):
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        self.yaw += mouse_dx * self.mouse_sensitivity
        self.pitch = float(np.clip(self.pitch - mouse_dy * self.mouse_sensitivity, -self.max_pitch, self.max_pitch))

    def update_camera_position(self, dt):
        keys = pygame.key.get_pressed()
        forward = np.array([np.sin(self.yaw), 0.0, np.cos(self.yaw)])
        right = np.array([np.cos(self.yaw), 0.0, -np.sin(self.yaw)])
        movement = np.zeros(3, dtype=np.float64)
        if keys[pygame.K_w]:
            movement += forward
        if keys[pygame.K_s]:
            movement -= forward
        if keys[pygame.K_d]:
            movement += right
        if keys[pygame.K_a]:
            movement -= right
        if keys[pygame.K_SPACE]:
            movement[1] -= 1.0
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            movement[1] += 1.0

        length = np.linalg.norm(movement)
        speed = self.move_speed * (2 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
        if length > 0:
            movement /= length
            self.camera_position += movement * speed * dt

    def make_camera_rt(self):
        sin_yaw, cos_yaw = np.sin(self.yaw), np.cos(self.yaw)
        sin_pitch, cos_pitch = np.sin(self.pitch), np.cos(self.pitch)
        R = np.array([
            [cos_yaw, 0.0, -sin_yaw],
            [sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch],
            [sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch],
        ], dtype=np.float64)
        t = self.camera_position.reshape(3, 1)
        return R, t

    def render_frame(self, screen, clock):
        R, t = self.make_camera_rt()
        frame = self.render_function(R, t)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(frame.tobytes(), self.screen_size, "RGB")
        screen.blit(surface, (0, 0))
        pygame.display.set_caption(f"DIBR Interactive Demo | FPS: {clock.get_fps():.1f} / {self.fps}")
        pygame.display.flip()

    def run(self):
        # Initialize interactive viewer
        pygame.init()
        screen = pygame.display.set_mode(self.screen_size)
        clock = pygame.time.Clock()

        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

        try:
            running = True
            while running:
                dt = min(clock.tick(self.fps) / 1000.0, 0.1)
                running = self.process_events()
                if not running:
                    break
                self.update_camera_rotation()
                self.update_camera_position(dt)
                self.render_frame(screen, clock)
        finally:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            pygame.quit()