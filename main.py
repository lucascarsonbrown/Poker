"""
Poker Calculator - Demo Entry Point

This demonstrates basic usage of the poker calculator API.
"""

from src.calculator import PokerCalculator


def demo_equity():
    """Demonstrate equity calculation."""
    print("=== Hand Equity Calculator ===\n")

    calc = PokerCalculator()

    # Example hands
    examples = [
        (["Ah", "Kh"], [], "Ace-King suited (preflop)"),
        (["Ah", "Kh"], ["Qh", "Jh", "2c"], "Flush draw + straight draw"),
        (["As", "Ad"], ["Kh", "Kd", "7c"], "Aces vs Kings on board"),
        (["7h", "2d"], ["Ah", "Kd", "Qs"], "7-2 offsuit (worst hand)"),
    ]

    for hole, community, desc in examples:
        equity = calc.get_equity(hole, community, simulations=5000)
        print(f"{desc}")
        print(f"  Hole: {hole}, Community: {community if community else 'None'}")
        print(f"  Equity: {equity:.1%}\n")


def demo_action():
    """Demonstrate AI action recommendation."""
    print("=== AI Action Recommendations ===\n")

    calc = PokerCalculator()

    # Scenario: We have AK on a Q-J-T flop (open-ended straight)
    result = calc.get_ai_action(
        hole_cards=["Ah", "Kd"],
        community_cards=["Qh", "Jd", "Tc"],
        pot_size=100,
        to_call=50,
        stack_size=500
    )

    print("Scenario: AK with open-ended straight draw")
    print(f"  Hole cards: Ah Kd")
    print(f"  Board: Qh Jd Tc")
    print(f"  Pot: 100, To call: 50")
    print(f"\n  Recommended action: {result['action']}")
    print(f"  Bet amount: {result['amount']}")
    print(f"  Hand equity: {result['equity']:.1%}")
    print(f"  Strategy: {result['strategy']}")


def demo_compare():
    """Demonstrate hand comparison."""
    print("\n=== Hand Comparison ===\n")

    calc = PokerCalculator()

    board = ["Ah", "Kd", "Qc", "Js", "2h"]
    hand1 = ["Th", "9h"]  # Broadway straight
    hand2 = ["Ac", "Kh"]  # Two pair

    result = calc.compare_hands(board, hand1, hand2)

    print(f"Board: {' '.join(board)}")
    print(f"Hand 1: {' '.join(hand1)} (has straight)")
    print(f"Hand 2: {' '.join(hand2)} (two pair)")
    print(f"\nWinner: {'Hand 1' if result == 1 else 'Hand 2' if result == -1 else 'Tie'}")


if __name__ == "__main__":
    print("=" * 50)
    print("  POKER CALCULATOR DEMO")
    print("=" * 50)
    print()

    demo_equity()
    demo_action()
    demo_compare()

    print("\n" + "=" * 50)
    print("  Demo complete!")
    print("  See README.md for full API documentation.")
    print("=" * 50)
