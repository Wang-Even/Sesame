def generate(t, home, m):
    z = 0*t
    pA = 2.5*t + 0.125
    pA = pA - m.floor(pA)
    pB = pA + 0.5
    pB = pB - m.floor(pB)
    sA = (pA - 0.5) / 0.5
    wA = m.cos(6.283185307179586*pA)
    pkA = pA + 0.0
    pkA = pkA - m.floor(pkA)
    skA = (pkA - 0.5) / 0.5
    kA = 0.15 + m.where(pkA < 0.5, z, 0.5*m.sin(3.141592653589793*skA))
    cA = pkA < 0.5
    sB = (pB - 0.5) / 0.5
    wB = m.cos(6.283185307179586*pB)
    pkB = pB + 0.0
    pkB = pkB - m.floor(pkB)
    skB = (pkB - 0.5) / 0.5
    kB = 0.15 + m.where(pkB < 0.5, z, 0.5*m.sin(3.141592653589793*skB))
    cB = pkB < 0.5
    q = m.stack([1.55+0.26*wA, 3.141592653589793-kA, 1.591592653589793+0.26*wB, kB, 1.591592653589793-0.26*wB, kB, 1.55-0.26*wA, 3.141592653589793-kA], dim=-1)
    pos = m.stack([home["speed"]*t, z, 0.04673357+z], dim=-1)
    rpy = m.stack([z,z,z],dim=-1)
    contacts = m.stack([cA,cB,cB,cA],dim=-1)
    return {"root_pos_w":pos,"root_rpy":rpy,"joint_pos":q,"contacts":contacts}
