"""
Streamlit poker app — play Texas Hold'em against the CFR AI.
Run with: streamlit run app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.environment import PokerEnvironment
from src.ai.abstraction import calculate_equity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Poker AI", page_icon="🃏", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1f0d; }
    section[data-testid="stSidebar"] { display: none; }

    .block-container { padding-top: 2rem; }
    .log-entry { font-size: 0.85rem; padding: 3px 0; border-bottom: 1px solid #1e3a1e; }
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; font-weight: bold; margin-left: 4px;
    }
    .badge-dealer { background: #f1c40f; color: #000; }
    .badge-sb     { background: #3498db; color: #fff; }
    .badge-bb     { background: #e74c3c; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ── Card rendering ────────────────────────────────────────────────────────────
SUIT_SYMBOL = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
SUIT_COLOR  = {"h": "#c0392b", "d": "#c0392b", "c": "#111", "s": "#111"}
RANK_DISPLAY = {"T": "10", "J": "J", "Q": "Q", "K": "K", "A": "A"}

def card_html(card: str, hidden: bool = False, size: int = 1) -> str:
    w, h, rf, sf = int(56*size), int(80*size), int(18*size), int(22*size)
    if hidden:
        return f"""<div style="
            display:inline-flex;align-items:center;justify-content:center;
            width:{w}px;height:{h}px;border-radius:8px;
            background:linear-gradient(135deg,#1a4a7a,#0d2a4a);
            border:2px solid #4a8ab5;font-size:{sf}px;margin:4px;color:#4a8ab5;">
            🂠</div>"""
    rank, suit = card[:-1], card[-1]
    label = RANK_DISPLAY.get(rank, rank)
    color = SUIT_COLOR[suit]
    symbol = SUIT_SYMBOL[suit]
    return f"""<div style="
        display:inline-flex;flex-direction:column;align-items:center;
        justify-content:center;width:{w}px;height:{h}px;border-radius:8px;
        background:white;border:2px solid #ccc;
        box-shadow:2px 3px 6px rgba(0,0,0,.4);
        font-weight:bold;color:{color};margin:4px;line-height:1.1;">
        <span style="font-size:{rf}px;">{label}</span>
        <span style="font-size:{sf}px;">{symbol}</span>
    </div>"""

def cards_html(cards, hidden=False, size=1):
    return "".join(card_html(c, hidden, size) for c in cards)

def blank_card_html(n, size=1):
    w, h = int(56*size), int(80*size)
    s = f"""<div style="display:inline-block;width:{w}px;height:{h}px;
        border-radius:8px;background:#0a1a0a;border:2px dashed #1e3a1e;margin:4px;"></div>"""
    return s * n

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    if "env" not in st.session_state:
        env = PokerEnvironment()
        env.add_player()
        env.add_ai_player(str(Path(__file__).parent / "models"))
        st.session_state.env = env

    defaults = {
        "phase":          "lobby",   # lobby | dealing | playing | showdown
        "player_hand":    [],
        "ai_hand":        [],
        "action_log":     [],        # list of dicts: {text, style}
        "balance_history": [],
        "round_num":      0,
        "last_result":    None,
        "dealer_pos":     None,      # 0=player, 1=AI
        "sb_pos":         None,
        "bb_pos":         None,
        "last_ai_action": None,
        "pot_won":        0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
env: PokerEnvironment = st.session_state.env

# ── Logging helpers ───────────────────────────────────────────────────────────
def log(text: str, style: str = "normal"):
    st.session_state.action_log.append({"text": text, "style": style})

def log_action(who: str, action: str, amount: int = 0):
    icon = "🧑" if who == "you" else "🤖"
    name = "You" if who == "you" else "AI"
    if action == "fold":
        msg = f"{icon} **{name}** folded"
        style = "fold"
    elif action == "check":
        msg = f"{icon} **{name}** checked"
        style = "check"
    elif action == "call":
        msg = f"{icon} **{name}** called ${amount:,}"
        style = "call"
    elif action == "bet":
        msg = f"{icon} **{name}** raised to ${amount:,}"
        style = "bet"
    else:
        msg = f"{icon} **{name}**: {action}"
        style = "normal"
    log(msg, style)

# ── Game helpers ──────────────────────────────────────────────────────────────
def advance_ai():
    while not env.end_of_round():
        state = env.get_game_state()
        if state["player_in_play"] == 0:
            break
        before_bet = state["players"][1]["current_bet"]
        before_pot = state["pot"]
        env.handle_game_stage()
        after_state = env.get_game_state()
        after_bet = after_state["players"][1]["current_bet"]

        # Figure out what the AI just did
        hist = env.history
        last = hist[-1] if hist else ""
        if last == "f":
            log_action("ai", "fold")
        elif last == "k":
            log_action("ai", "check")
        elif last == "c":
            to_call = after_bet - before_bet
            log_action("ai", "call", to_call)
        elif last.startswith("b"):
            log_action("ai", "bet", int(last[1:]))

def start_round():
    st.session_state.round_num += 1
    st.session_state.phase = "dealing"
    st.session_state.action_log = []
    st.session_state.last_result = None
    st.session_state.last_ai_action = None
    st.session_state.pot_won = 0

    env.start_new_round()

    # Record blind positions
    n = len(env.players)
    d = env.dealer_button_position
    sb = (d + 2) % n   # in 2-player: dealer = small blind
    bb = (d + 1) % n
    st.session_state.dealer_pos = d
    st.session_state.sb_pos = sb
    st.session_state.bb_pos = bb

    sb_name = "You" if sb == 0 else "AI"
    bb_name = "You" if bb == 0 else "AI"
    log(f"🃏  Round {st.session_state.round_num} — new hand", "header")
    log(f"🔵  Small blind (${env.SMALL_BLIND}): **{sb_name}**", "blind")
    log(f"🔴  Big blind (${env.BIG_BLIND}): **{bb_name}**", "blind")

    st.session_state.player_hand = [str(c) for c in env.get_player(0).hand]
    st.session_state.ai_hand     = [str(c) for c in env.get_player(1).hand]

    log("🂠  Cards dealt", "deal")
    st.session_state.phase = "playing"
    advance_ai()

    if env.end_of_round():
        finish_round()

def stage_label(stage):
    return {2:"Preflop", 3:"Flop", 4:"Turn", 5:"River", 6:"Showdown"}.get(stage, "")

def on_new_street(state):
    """Log when a new community street is revealed."""
    s = state["stage"]
    if s == 3:
        log(f"🎴  **Flop**: {' '.join(state['community_cards'][:3])}", "street")
    elif s == 4:
        log(f"🎴  **Turn**: {state['community_cards'][3]}", "street")
    elif s == 5:
        log(f"🎴  **River**: {state['community_cards'][4]}", "street")

_last_stage = None

def apply_action(action: str):
    global _last_stage
    state_before = env.get_game_state()
    stage_before = state_before["stage"]

    env.handle_game_stage(action)

    state_after = env.get_game_state()
    stage_after = state_after["stage"]

    # Log player action
    if action == "f":
        log_action("you", "fold")
    elif action == "k":
        log_action("you", "check")
    elif action == "c":
        to_call = state_before["players"][1]["current_bet"] - state_before["players"][0]["current_bet"]
        log_action("you", "call", to_call)
    elif action.startswith("b"):
        log_action("you", "bet", int(action[1:]))

    # Log new street if one was dealt
    if stage_after != stage_before and stage_after in (3, 4, 5):
        on_new_street(state_after)

    if env.end_of_round():
        finish_round()
        return

    advance_ai()

    state_now = env.get_game_state()
    if state_now["stage"] != stage_after and state_now["stage"] in (3, 4, 5):
        on_new_street(state_now)

    if env.end_of_round():
        finish_round()

def finish_round():
    st.session_state.phase = "showdown"
    winners = env.get_winner_indices()
    pot = env.total_pot_balance

    if 0 in winners and 1 in winners:
        st.session_state.last_result = "tie"
        st.session_state.pot_won = pot // 2
        log(f"🤝  **Split pot** — each wins ${pot//2:,}", "result-tie")
    elif 0 in winners:
        st.session_state.last_result = "win"
        st.session_state.pot_won = pot
        log(f"🏆  **You win** ${pot:,}!", "result-win")
    else:
        st.session_state.last_result = "lose"
        log(f"💀  **AI wins** ${pot:,}", "result-lose")

    if env.showdown:
        ai = [str(c) for c in env.get_player(1).hand]
        log(f"🂠  AI had: {' '.join(ai)}", "reveal")

    balance = env.get_player(0).player_balance
    st.session_state.balance_history.append(
        {"Round": st.session_state.round_num, "Stack": int(balance)}
    )

# ── Log style map ─────────────────────────────────────────────────────────────
LOG_STYLES = {
    "header":     "color:#f1c40f;font-weight:bold;",
    "blind":      "color:#85c1e9;",
    "deal":       "color:#aaa;font-style:italic;",
    "street":     "color:#2ecc71;font-weight:bold;",
    "fold":       "color:#e74c3c;",
    "check":      "color:#95a5a6;",
    "call":       "color:#3498db;",
    "bet":        "color:#e67e22;font-weight:bold;",
    "result-win": "color:#2ecc71;font-weight:bold;font-size:1.05em;",
    "result-lose":"color:#e74c3c;font-weight:bold;",
    "result-tie": "color:#f39c12;font-weight:bold;",
    "reveal":     "color:#9b59b6;",
    "normal":     "color:#ccc;",
}

def render_log():
    for entry in reversed(st.session_state.action_log):
        style = LOG_STYLES.get(entry["style"], LOG_STYLES["normal"])
        st.markdown(
            f'<div class="log-entry" style="{style}">{entry["text"]}</div>',
            unsafe_allow_html=True,
        )

# ── Badge helpers ─────────────────────────────────────────────────────────────
def player_badges(idx: int) -> str:
    badges = ""
    if st.session_state.dealer_pos == idx:
        badges += '<span class="badge badge-dealer">D</span>'
    if st.session_state.sb_pos == idx:
        badges += '<span class="badge badge-sb">SB</span>'
    if st.session_state.bb_pos == idx:
        badges += '<span class="badge badge-bb">BB</span>'
    return badges

st.title("Poker AI")
st.caption("Explore a hand or play heads-up against the AI. Demo chips only.")
st.link_button("View source on GitHub", "https://github.com/lucascarsonbrown/Poker")
mode = st.radio("Mode", ["Play against AI", "Hand calculator"], horizontal=True)

if mode == "Hand calculator":
    st.subheader("Try a hand")
    st.write("Choose your cards and an optional board to estimate your chance of winning or tying against one random opponent.")
    deck = [rank + suit for rank in "AKQJT98765432" for suit in "shdc"]
    def card_label(card):
        return RANK_DISPLAY.get(card[0], card[0]) + SUIT_SYMBOL[card[1]]
    with st.form("calculator"):
        hole = st.multiselect("Your two cards", deck, default=["Ah", "Kd"], max_selections=2, format_func=card_label)
        board = st.multiselect("Community cards (0, 3, 4, or 5)", deck, max_selections=5, format_func=card_label)
        simulations = st.select_slider("Simulations", options=[500, 2000, 5000], value=2000)
        submitted = st.form_submit_button("Calculate", type="primary")
    if submitted:
        if len(hole) != 2:
            st.error("Choose exactly two hole cards.")
        elif len(board) not in (0, 3, 4, 5):
            st.error("Choose no board cards, or a flop (3), turn (4), or river (5).")
        elif len(set(hole + board)) != len(hole + board):
            st.error("Each card can appear only once. Your hand and board overlap.")
        else:
            with st.spinner("Simulating hands…"):
                probability = calculate_equity(hole, board, n=simulations)
            st.markdown(cards_html(hole), unsafe_allow_html=True)
            if board:
                st.markdown(cards_html(board), unsafe_allow_html=True)
            st.metric("Win or tie probability", f"{probability:.1%}")
            st.progress(probability)
            st.caption(f"Monte Carlo estimate from {simulations:,} hands. The current calculator counts ties as wins; this is not split-pot equity. Results vary between runs and do not model the AI's range.")
    st.stop()

st.caption("Blinds: 100 / 200 · Starting stack: 2,500 · Empty stacks refill on the next hand. The engine uses simplified betting rules.")
if st.button("Reset session"):
    st.session_state.clear()
    st.rerun()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    phase = st.session_state.phase

    if phase == "lobby":
        st.markdown("""
        <div style="text-align:center;padding:60px 0;">
            <div style="font-size:4rem;">🃏</div>
            <h1 style="color:#f1c40f;">Texas Hold'em</h1>
            <p style="color:#aaa;font-size:1.1rem;">vs CFR AI — blinds $100 / $200</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        state = env.get_game_state()
        stage = state["stage"]
        community = state["community_cards"]
        pot = state["pot"]
        p_bet  = int(state["players"][0]["current_bet"])
        ai_bet = int(state["players"][1]["current_bet"])
        # Bets are deducted by the engine at the end of each street.
        p_bal  = int(state["players"][0]["balance"])
        ai_bal = int(state["players"][1]["balance"])

        show_ai_cards = phase == "showdown" and env.showdown

        # ── AI row ────────────────────────────────────────────────────────────
        ai_badge_html = player_badges(1) if phase in ("playing","showdown") else ""
        st.markdown(
            f'<div style="color:#ccc;font-size:1rem;margin-bottom:4px;">'
            f'🤖 <b>AI</b>{ai_badge_html}'
            f' &nbsp; <span style="color:#aaa;">stack ${ai_bal:,}</span>'
            + (f' &nbsp; <span style="color:#e67e22;">bet ${ai_bet:,}</span>' if ai_bet else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        if show_ai_cards:
            st.markdown(cards_html(st.session_state.ai_hand), unsafe_allow_html=True)
        else:
            st.markdown(cards_html(["??","??"], hidden=True), unsafe_allow_html=True)

        st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

        # ── Board ─────────────────────────────────────────────────────────────
        street = stage_label(stage)
        st.markdown(
            f'<div style="color:#f1c40f;font-size:0.9rem;letter-spacing:1px;">'
            f'{street.upper() if street else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        board = cards_html(community, size=1.1) + blank_card_html(5 - len(community), size=1.1)
        st.markdown(
            f'<div style="background:#0a1f0a;border-radius:12px;padding:12px 8px;'
            f'display:inline-block;margin:4px 0;">{board}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:#f1c40f;font-size:1.1rem;margin-top:6px;">💰 Pot: ${pot:,}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

        # ── Player row ────────────────────────────────────────────────────────
        p_badge_html = player_badges(0) if phase in ("playing","showdown") else ""
        st.markdown(
            f'<div style="color:#ccc;font-size:1rem;margin-bottom:4px;">'
            f'🧑 <b>You</b>{p_badge_html}'
            f' &nbsp; <span style="color:#aaa;">stack ${p_bal:,}</span>'
            + (f' &nbsp; <span style="color:#e67e22;">bet ${p_bet:,}</span>' if p_bet else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(cards_html(st.session_state.player_hand, size=1.1), unsafe_allow_html=True)

        st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

        # ── Equity bar ───────────────────────────────────────────────────────
        if st.session_state.player_hand and phase == "playing":
            equity = calculate_equity(st.session_state.player_hand, community, n=600)
            eq_pct = equity * 100
            bar_color = "#2ecc71" if eq_pct >= 50 else "#e74c3c"
            st.markdown(
                f'<div style="font-size:0.85rem;color:#aaa;margin-bottom:2px;">'
                f'Win or tie vs random hand: <b style="color:{bar_color}">{eq_pct:.0f}%</b></div>',
                unsafe_allow_html=True,
            )
            st.progress(equity)

        # ── Result banner ─────────────────────────────────────────────────────
        if phase == "showdown":
            result = st.session_state.last_result
            if result == "win":
                st.success(f"🏆  You win ${st.session_state.pot_won:,}!")
            elif result == "lose":
                st.error("💀  AI wins this hand.")
            else:
                st.warning(f"🤝  Split — you each get ${st.session_state.pot_won:,}")

        # ── Action buttons ────────────────────────────────────────────────────
        elif phase == "playing" and state["player_in_play"] == 0:
            valid    = state["valid_actions"]
            balance  = p_bal
            to_call  = ai_bet - p_bet
            min_raise = max(ai_bet * 2, pot // 3)
            small    = max(min_raise, pot // 3)

            st.markdown('<div style="color:#aaa;font-size:0.85rem;margin-bottom:6px;">Your action:</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            c = 0

            if "k" in valid:
                if cols[c].button("✋ Check", use_container_width=True):
                    apply_action("k"); st.rerun()
                c += 1
            if "c" in valid:
                if cols[c].button(f"📞 Call ${to_call:,}", use_container_width=True):
                    apply_action("c"); st.rerun()
                c += 1
            if cols[c].button(f"⬆️ Raise ${min(small, balance):,}", use_container_width=True, disabled=balance <= ai_bet):
                apply_action(f"b{min(small, balance)}"); st.rerun()
            c += 1
            if cols[c].button(f"💣 Pot ${min(max(pot, min_raise), balance):,}", use_container_width=True, disabled=balance <= ai_bet):
                apply_action(f"b{min(max(pot, min_raise), balance)}"); st.rerun()
            c += 1
            if cols[c].button(f"🚀 All-in ${balance:,}", use_container_width=True, disabled=balance <= ai_bet):
                apply_action(f"b{balance}"); st.rerun()

            if "f" in valid:
                if st.button("❌ Fold", use_container_width=False):
                    apply_action("f"); st.rerun()

    # ── New round button ──────────────────────────────────────────────────────
    st.markdown("<div style='margin:20px 0 8px;'></div>", unsafe_allow_html=True)
    label = "🃏  Deal Cards" if phase == "lobby" else "🔄  Next Round"
    if st.button(label, type="primary", use_container_width=False, disabled=phase == "playing"):
        start_round()
        st.rerun()

# ── Right panel ───────────────────────────────────────────────────────────────
with right:
    st.markdown('<div style="color:#f1c40f;font-weight:bold;margin-bottom:8px;">📋 Hand log</div>', unsafe_allow_html=True)
    if st.session_state.action_log:
        render_log()
    else:
        st.markdown('<div style="color:#555;font-size:0.85rem;">Actions appear here during the hand.</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0;border-top:1px solid #1e3a1e;'></div>", unsafe_allow_html=True)

    st.markdown('<div style="color:#f1c40f;font-weight:bold;margin-bottom:8px;">📈 Stack history</div>', unsafe_allow_html=True)
    if st.session_state.balance_history:
        df = pd.DataFrame(st.session_state.balance_history)
        st.line_chart(df.set_index("Round")["Stack"], color="#2ecc71")
        latest = st.session_state.balance_history[-1]["Stack"]
        starting = env.starting_balance
        delta  = latest - starting
        sign   = "+" if delta >= 0 else ""
        col1, col2 = st.columns(2)
        col1.metric("Stack", f"${latest:,}")
        col2.metric("P&L", f"{sign}${delta:,}", delta_color="normal")
        rn = st.session_state.round_num
        st.markdown(
            f'<div style="color:#555;font-size:0.8rem;margin-top:8px;">{rn} round{"s" if rn!=1 else ""} played</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div style="color:#555;font-size:0.85rem;">Chart appears after the first hand.</div>', unsafe_allow_html=True)
