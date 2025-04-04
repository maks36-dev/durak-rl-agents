import pygame
import os
import sys
import pickle

from src.po_mcts.env import DurakEnv
from src.po_mcts.agent import POMCTSAgent, POMCTSNode
from src.po_mcts.greedy_player import GreedyAgent


INITIAL_WIDTH = 1280
INITIAL_HEIGHT = 720
FPS = 30
CARD_WIDTH = 80
CARD_HEIGHT = 120

ASSETS_DIR = os.path.join(r"..\assets", "cards")

RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['spades', 'clubs', 'hearts', 'diamonds']


def load_card(id):
    filename = f"{RANKS[id // 4]}_{SUITS[id % 4]}.png"
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (CARD_WIDTH, CARD_HEIGHT))
        return image
    return None


def draw_deck(screen, deck, x, y):
    if len(deck) > 0:
        card_back_path = os.path.join(ASSETS_DIR, "card_back.png")
        if os.path.exists(card_back_path):
            card_back = pygame.image.load(card_back_path)
            card_back = pygame.transform.scale(card_back, (CARD_WIDTH, CARD_HEIGHT))
            screen.blit(card_back, (x, y))


def draw_table(env, current_width, current_height, screen):
    table_cards = env.table_cards
    for idx in range(6):
        field_x = current_width // 2 - 3 * CARD_WIDTH + idx * CARD_WIDTH
        field_y = current_height // 2 - CARD_HEIGHT // 2
        pygame.draw.rect(screen, (200, 200, 200), (field_x, field_y, CARD_WIDTH, CARD_HEIGHT), 2)
        if idx < len(table_cards):
            if table_cards[idx][0] is not None:
                card = table_cards[idx][0]
                card_image = load_card(card)
                if card_image:
                    screen.blit(card_image, (field_x, field_y))
            if table_cards[idx][1] is not None:
                card = table_cards[idx][1]
                card_image = load_card(card)
                if card_image:
                    screen.blit(card_image, (field_x, field_y + 30))


def draw_card_back(screen, x, y):
    card_back_path = os.path.join(ASSETS_DIR, "card_back.png")
    if os.path.exists(card_back_path):
        card_back = pygame.image.load(card_back_path)
        card_back = pygame.transform.scale(card_back, (CARD_WIDTH, CARD_HEIGHT))
        screen.blit(card_back, (x, y))
    else:
        pygame.draw.rect(screen, (255, 0, 0), (x, y, CARD_WIDTH, CARD_HEIGHT))


def draw_hand(screen, hand, start_x, start_y, is_mouse_moving, face_up=True):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    x_offset = 0
    for card in hand:
        card_image = load_card(card)
        card_x = start_x + x_offset
        card_y = start_y

        if (card_x <= mouse_x <= card_x + CARD_WIDTH // 2 and
            card_y <= mouse_y <= card_y + CARD_HEIGHT and
            face_up and not is_mouse_moving):
            if face_up and card_image:
                enlarged_card = pygame.transform.scale(card_image, (int(CARD_WIDTH * 1.2), int(CARD_HEIGHT * 1.2)))
                screen.blit(enlarged_card, (card_x - 10, card_y - 20))
            else:
                draw_card_back(screen, card_x - 10, card_y - 20)
        else:
            if face_up and card_image:
                screen.blit(card_image, (card_x, card_y))
            else:
                draw_card_back(screen, card_x, card_y)

        x_offset += 30


def draw_button(screen, text, x, y, width, height, color, text_color, font):
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, text_color)
    text_x = x + (width - text_surface.get_width()) // 2
    text_y = y + (height - text_surface.get_height()) // 2
    screen.blit(text_surface, (text_x, text_y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((INITIAL_WIDTH, INITIAL_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Дурак")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    background_path = os.path.join("..", "assets", "background.png")
    if os.path.exists(background_path):
        background = pygame.image.load(background_path)
        background = pygame.transform.scale(background, (INITIAL_WIDTH, INITIAL_HEIGHT))
    else:
        background = None

    env = DurakEnv()
    env.reset()

    computer = POMCTSAgent(num_simulations=1000)
    greedy_agent = GreedyAgent()

    current_width = INITIAL_WIDTH
    current_height = INITIAL_HEIGHT

    dragging_card = None
    drag_offset_x = 0
    drag_offset_y = 0

    button_width = 120
    button_height = 40
    take_button_x = current_width // 2 - button_width - 100
    take_button_y = current_height - 40
    pass_button_x = current_width // 2 + 100
    pass_button_y = current_height - 40

    game_mode = None


    selecting_mode = True
    while selecting_mode:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                current_width, current_height = event.w, event.h
                screen = pygame.display.set_mode((current_width, current_height), pygame.RESIZABLE)
                if background:
                    background = pygame.transform.scale(background, (current_width, current_height))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                pve_button_rect = pygame.Rect(current_width // 2 - 150, current_height // 2 - 50, 300, 60)
                cvc_button_rect = pygame.Rect(current_width // 2 - 150, current_height // 2 + 20, 300, 60)
                if pve_button_rect.collidepoint(mouse_x, mouse_y):
                    game_mode = "pve"
                    selecting_mode = False
                elif cvc_button_rect.collidepoint(mouse_x, mouse_y):
                    game_mode = "cvc"
                    selecting_mode = False

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill((34, 139, 34))

        title_surf = font.render("Выберите режим игры:", True, (255, 255, 255))
        screen.blit(title_surf, (current_width // 2 - title_surf.get_width() // 2,
                                 current_height // 2 - 120))

        pve_button_rect = pygame.Rect(current_width // 2 - 150, current_height // 2 - 50, 300, 60)
        pygame.draw.rect(screen, (70, 130, 180), pve_button_rect)
        txt1 = font.render("Игрок / Компьютер", True, (255, 255, 255))
        screen.blit(txt1, (pve_button_rect.centerx - txt1.get_width() // 2,
                           pve_button_rect.centery - txt1.get_height() // 2))

        cvc_button_rect = pygame.Rect(current_width // 2 - 150, current_height // 2 + 20, 300, 60)
        pygame.draw.rect(screen, (180, 100, 50), cvc_button_rect)
        txt2 = font.render("Компьютер / Компьютер", True, (255, 255, 255))
        screen.blit(txt2, (cvc_button_rect.centerx - txt2.get_width() // 2,
                           cvc_button_rect.centery - txt2.get_height() // 2))

        pygame.display.flip()


    step = 0
    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                current_width, current_height = event.w, event.h
                screen = pygame.display.set_mode((current_width, current_height), pygame.RESIZABLE)
                if background:
                    background = pygame.transform.scale(background, (current_width, current_height))

                take_button_x = current_width // 2 - button_width - 10
                take_button_y = current_height - 100
                pass_button_x = current_width // 2 + 10
                pass_button_y = current_height - 100


            if game_mode == "pve":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos

                    if (take_button_x <= mouse_x <= take_button_x + button_width and
                            take_button_y <= mouse_y <= take_button_y + button_height):
                        env.step(None)
                    elif (pass_button_x <= mouse_x <= pass_button_x + button_width and
                          pass_button_y <= mouse_y <= pass_button_y + button_height):
                        env.step(None)
                    else:
                        for idx, card in enumerate(env.hands[0]):
                            card_x = current_width // 2 - (len(env.hands[0]) * 30) // 2 + idx * 30
                            card_y = current_height - CARD_HEIGHT - 50
                            if card_x <= mouse_x <= card_x + CARD_WIDTH // 2 and card_y <= mouse_y <= card_y + CARD_HEIGHT:
                                dragging_card = card
                                drag_offset_x = mouse_x - card_x
                                drag_offset_y = mouse_y - card_y
                                env.hands[0].remove(card)
                                break

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if dragging_card is not None:
                        mouse_x, mouse_y = event.pos
                        dropped_on_field = False
                        for idx in range(6):
                            field_x = current_width // 2 - 3 * CARD_WIDTH + idx * CARD_WIDTH
                            field_y = current_height // 2 - CARD_HEIGHT // 2
                            if (field_x <= mouse_x <= field_x + CARD_WIDTH // 2 and
                                field_y <= mouse_y <= field_y + CARD_HEIGHT):

                                env.hands[0].append(dragging_card)
                                env.step(dragging_card)
                                dragging_card = None
                                dropped_on_field = True
                                break

                        if not dropped_on_field and dragging_card:
                            env.hands[0].append(dragging_card)
                            dragging_card = None

        if game_mode == "cvc":
            step = step + 1 if step < 30 else 0
            if env.state == 0 and not env.done:
                if step == 0:
                    action = computer.act(env)
                    env.step(action)

            if env.state == 1 and not env.done:
                if step == 15:
                    action = greedy_agent.act(env)
                    env.step(action)


        if game_mode == "pve":
            if env.state == 1:
                action = computer.act(env)
                env.step(action)


        _, payoff, done = env.get_current_obs()
        if done:
            print(f"Win: {'Игрок' if payoff == 1 else 'Компьютер' if payoff == -1 else 'Ничья'}")
            running = False


        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill((34, 139, 34))

        if game_mode == 'pve':
            draw_button(screen, "Взять", take_button_x, take_button_y, button_width, button_height, (0, 128, 0),
                        (255, 255, 255), font)
            draw_button(screen, "Отбой", pass_button_x, pass_button_y, button_width, button_height, (128, 0, 0),
                        (255, 255, 255), font)


        draw_table(env, current_width, current_height, screen)

        computer_y = 50
        player_y = current_height - CARD_HEIGHT - 50
        player_x = current_width // 2 - (len(env.hands[0]) * 30) // 2
        computer_x = current_width // 2 - (len(env.hands[1]) * 30) // 2


        if game_mode == "pve":
            draw_hand(screen, env.hands[1], computer_x, computer_y, dragging_card, face_up=False)
        else:
            draw_hand(screen, env.hands[1], computer_x, computer_y, dragging_card, face_up=True)

        if game_mode == "pve":
            draw_hand(screen, env.hands[0], player_x, player_y, dragging_card, face_up=True)
        else:
            draw_hand(screen, env.hands[0], player_x, player_y, dragging_card, face_up=True)

        if dragging_card is not None:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            card_x = mouse_x - drag_offset_x
            card_y = mouse_y - drag_offset_y
            dragging_card_image = load_card(dragging_card)
            if dragging_card_image:
                screen.blit(dragging_card_image, (card_x, card_y))

        if env.deck:
            deck_x = 50
            deck_y = current_height // 2 - CARD_HEIGHT // 2

            last_card = env.deck[-1]
            last_card_image = load_card(last_card)
            if last_card_image:
                rotated_card = pygame.transform.rotate(last_card_image, 90)
                rotated_card_x = deck_x + CARD_WIDTH - CARD_HEIGHT // 2
                rotated_card_y = deck_y + CARD_HEIGHT // 2 - CARD_WIDTH // 2
                screen.blit(rotated_card, (rotated_card_x, rotated_card_y))
            draw_deck(screen, env.deck, deck_x, deck_y)

        if game_mode == 'cvc':
            player_agent_name = pygame.Rect(current_width // 2 - 150, current_height // 2 - 50 - 320, 300, 60)
            txt1 = font.render("Greedy Agent", True, (255, 255, 255))
            screen.blit(txt1, (player_agent_name.centerx - txt1.get_width() // 2,
                               player_agent_name.centery - txt1.get_height() // 2))

            computer_agent_name = pygame.Rect(current_width // 2 - 150, current_height // 2 + 20 + 280, 300, 60)
            txt2 = font.render("PO MCTS Agent", True, (255, 255, 255))
            screen.blit(txt2, (computer_agent_name.centerx - txt2.get_width() // 2,
                               computer_agent_name.centery - txt2.get_height() // 2))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
