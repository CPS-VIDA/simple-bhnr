import temporal_logic.signal_tl as stl

SIGNALS = (
    'p_x', 'p_y', 'v_x', 'v_y', 'theta', 'avel',
    'contact_l', 'contact_r'
)

p_x, p_y, v_x, v_y, theta, avel, contact_l, contact_r = stl.signals(SIGNALS)

POSITION_SPEC = stl.F()
