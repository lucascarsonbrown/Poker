"""
Bit position lookup table for fast hand evaluation.
"""

def generate_table():
    """Generate lookup table mapping single-bit values to their position."""
    table = {}
    for i in range(57):
        table[1 << i] = i
    return table
