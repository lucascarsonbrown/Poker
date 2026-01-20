"""
Interactive Poker Game - Play against the AI.

Run with: python play.py
"""

from src.environment import PokerEnvironment


def format_cards(cards):
    """Format card list for display."""
    if not cards:
        return "None"
    return " ".join(cards)


def display_game_state(env, player_hand):
    """Display current game state."""
    state = env.get_game_state()

    stage_names = {
        1: "Starting",
        2: "Preflop",
        3: "Flop",
        4: "Turn",
        5: "River",
        6: "Showdown"
    }

    print("\n" + "=" * 50)
    print(f"Stage: {stage_names.get(state['stage'], 'Unknown')}")
    print(f"Pot: ${state['pot']}")
    print(f"Community Cards: {format_cards(state['community_cards'])}")
    print("-" * 50)
    print(f"Your Hand: {format_cards(player_hand)}")
    print(f"Your Balance: ${state['players'][0]['balance']}")
    print(f"Your Bet: ${state['players'][0]['current_bet']}")
    print("-" * 50)
    print(f"AI Balance: ${state['players'][1]['balance']}")
    print(f"AI Bet: ${state['players'][1]['current_bet']}")
    print("=" * 50)


def get_player_action(env):
    """Get action from the player."""
    state = env.get_game_state()
    valid = state['valid_actions']
    pot = state['pot']
    balance = state['players'][0]['balance']

    # Calculate bet sizes
    bet_min = max(pot // 3, 1)
    bet_mid = pot
    bet_max = balance

    print("\nValid actions:")
    if "k" in valid:
        print("  [k] Check")
    if "c" in valid:
        to_call = state['players'][1]['current_bet'] - state['players'][0]['current_bet']
        print(f"  [c] Call (${to_call})")
    if "f" in valid:
        print("  [f] Fold")

    print("  Bet/Raise options:")
    print(f"      [1] Small  - ${bet_min}")
    print(f"      [2] Medium - ${bet_mid}")
    print(f"      [3] All-in - ${bet_max}")
    print("      Or type a dollar amount (e.g., 300)")

    while True:
        action = input("\nYour action: ").strip().lower()

        # Check for basic actions
        if action in ["k", "check"]:
            if "k" in valid:
                return "k"
            print("Cannot check - there's a bet to call.")
            continue
        elif action in ["c", "call"]:
            return "c"
        elif action in ["f", "fold"]:
            return "f"

        # Check for numbered bet options
        if action == "1":
            return "bMIN"
        elif action == "2":
            return "bMID"
        elif action in ["3", "allin", "all-in", "all in"]:
            return "bMAX"

        # Check for named bet sizes
        if action in ["bmin", "small", "min"]:
            return "bMIN"
        elif action in ["bmid", "medium", "mid", "pot"]:
            return "bMID"
        elif action in ["bmax", "max", "big"]:
            return "bMAX"

        # Try to parse as a dollar amount
        try:
            # Handle formats like "b300", "b 300", "$300", or just "300"
            amount_str = action.replace("b", "").replace("$", "").strip()
            amount = int(amount_str)

            # Map to closest abstract action
            if amount >= bet_max * 0.8:
                print(f"${amount} -> All-in (${bet_max})")
                return "bMAX"
            elif amount >= (bet_min + bet_mid) / 2:
                print(f"${amount} -> Medium bet (${bet_mid})")
                return "bMID"
            else:
                print(f"${amount} -> Small bet (${bet_min})")
                return "bMIN"
        except ValueError:
            pass

        print("Invalid action. Try: k, c, f, 1, 2, 3, or a dollar amount")


def play_round(env):
    """Play one round of poker."""
    env.start_new_round()

    # Get player's hand
    player = env.get_player(0)
    player_hand = [str(c) for c in player.hand]

    while not env.end_of_round():
        state = env.get_game_state()
        display_game_state(env, player_hand)

        if state['player_in_play'] == 0:
            # Human's turn
            action = get_player_action(env)
            env.handle_game_stage(action)
        else:
            # AI's turn
            print("\nAI is thinking...")
            env.handle_game_stage()
            print("AI has acted.")

    # Show result
    display_game_state(env, player_hand)

    winners = env.get_winner_indices()

    # Show AI's hand if showdown
    if env.showdown:
        ai_hand = [str(c) for c in env.get_player(1).hand]
        print(f"\nAI's Hand: {format_cards(ai_hand)}")

    if 0 in winners and 1 in winners:
        print("\n*** TIE! Pot split. ***")
    elif 0 in winners:
        print("\n*** YOU WIN! ***")
    else:
        print("\n*** AI WINS! ***")

    return 0 in winners


def main():
    print("=" * 50)
    print("   POKER AI - Texas Hold'em")
    print("   Play against the trained CFR AI")
    print("=" * 50)

    # Create environment
    env = PokerEnvironment()
    env.add_player()  # Human (index 0)
    env.add_ai_player("models")  # AI (index 1)

    wins = 0
    rounds = 0

    while True:
        rounds += 1
        print(f"\n\n{'#' * 50}")
        print(f"   ROUND {rounds}")
        print(f"{'#' * 50}")

        if play_round(env):
            wins += 1

        print(f"\nRecord: {wins} wins / {rounds} rounds")

        play_again = input("\nPlay another round? (y/n): ").strip().lower()
        if play_again not in ["y", "yes", ""]:
            break

    print("\nThanks for playing!")
    print(f"Final record: {wins} wins / {rounds} rounds ({100*wins/rounds:.1f}%)")


if __name__ == "__main__":
    main()
