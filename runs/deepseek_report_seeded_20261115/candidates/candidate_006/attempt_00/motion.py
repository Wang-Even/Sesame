def generate(t, home, m):
    freq = 2.45
    A = 0.285
    center = 1.615
    stance_flex = 0.15
    swing_lift = 0.57
    base_z = 0.04673357
    phase0 = 0.115
    pi = 3.141592653589793
    two_pi = 6.283185307179586

    pA = freq * t + phase0
    pA = pA - m.floor(pA)
    pB = pA + 0.5
    pB = pB - m.floor(pB)

    cA = m.cos(two_pi * pA)
    cB = m.cos(two_pi * pB)

    R1 = center + A * cA
    R2 = pi - center + A * cB
    L1 = pi - center - A * cB
    L2 = center - A * cA

    swingA = pA >= 0.5
    swingB = pB >= 0.5

    sA = (pA - 0.5) * 2.0
    sB = (pB - 0.5) * 2.0
    archA = m.sin(pi * sA)
    archB = m.sin(pi * sB)

    flexA = m.where(swingA, stance_flex + swing_lift * archA, stance_flex)
    flexB = m.where(swingB, stance_flex + swing_lift * archB, stance_flex)

    R3 = pi - flexA
    L4 = pi - flexA
    R4 = flexB
    L3 = flexB

    q = m.stack([R1, R3, R2, R4, L1, L3, L2, L4], dim=-1)

    z = 0 * t
    o = 1 + z
    root_pos_w = m.stack([home['speed'] * t, z, base_z * o], dim=-1)
    root_rpy = m.stack([z, z, z], dim=-1)

    contactA = pA < 0.5
    contactB = pB < 0.5
    contacts = m.stack([contactA, contactB, contactB, contactA], dim=-1)

    return {"root_pos_w": root_pos_w, "root_rpy": root_rpy, "joint_pos": q, "contacts": contacts}