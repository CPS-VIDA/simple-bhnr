import temporal_logic.signal_tl as stl


SIGNALS = ('x', 'x_dot', 'theta', 'theta_dot')
x, x_dot, theta, theta_dot = stl.signals(SIGNALS)
SPEC = stl.G(stl.F(x_dot < abs(0.01)) & (abs(theta) < 5)
             & (abs(x) < 0.5) & stl.F(abs(theta) < 1))
