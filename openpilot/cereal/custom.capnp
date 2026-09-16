using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct MoonpilotState @0x81c2f05a394cf4af {  # moonpilot seam: upstream's reserved struct, renamed. Do not change the @0x id, and see AGENTS.md before editing.
  # Normalized lead trajectories, one entry per modelV2.leadsV3 slot (3), in that order.
  # leads[0]/leads[1] correspond to radarState.leadOne/leadTwo.
  leads @0 :List(LeadTrajectory);

  struct LeadTrajectory {
    present @0 :Bool;
    prob @1 :Float32;      # radard's filtered lead probability where radard has a counterpart, the raw model prob otherwise
    probTime @2 :Float32;  # s; the time prob refers to (ModelConstants.LEAD_T_OFFSETS[i])
    source @3 :Source;

    # Anchored so index 0 equals the fused radarState values.
    # x: m forward of the front bumper. y: m in car frame, LEFT POSITIVE (radarState convention).
    t @4 :List(Float32);   # s, ModelConstants.LEAD_T_IDXS
    x @5 :List(Float32);
    y @6 :List(Float32);
    v @7 :List(Float32);   # m/s absolute lead speed
    a @8 :List(Float32);   # m/s^2, signed; negative is deceleration
    xStd @9 :List(Float32);
    yStd @10 :List(Float32);
    vStd @11 :List(Float32);
    aStd @12 :List(Float32);

    # Derived here so every consumer agrees. yawRel: lead heading relative to the ego x
    # axis, atan2(dy/dt, v), left positive, one entry per t.
    # inPathProb: per-sample probability that the lead's lateral Gaussian lies inside the
    # ego path corridor, one entry per t. inPath: those probabilities inverse-variance
    # weighted into one scalar and asymmetrically filtered. 1.0 means fully in path.
    yawRel @13 :List(Float32);
    inPath @14 :Float32;
    inPathProb @15 :List(Float32);
  }

  enum Source {
    none @0;
    vision @1;
    radar @2;
  }
}

struct CustomReserved1 @0xaedffd8f31e7b55d {
}

struct CustomReserved2 @0xf35cc4560bbf6ec2 {
}

struct CustomReserved3 @0xda96579883444c35 {
}

struct CustomReserved4 @0x80ae746ee2596b11 {
}

struct CustomReserved5 @0xa5cd762cd951a455 {
}

struct CustomReserved6 @0xf98d843bfd7004a3 {
}

struct CustomReserved7 @0xb86e6369214c01c8 {
}

struct CustomReserved8 @0xf416ec09499d9d19 {
}

struct CustomReserved9 @0xa1680744031fdb2d {
}

struct CustomReserved10 @0xcb9fd56c7057593a {
}

struct CustomReserved11 @0xc2243c65e0340384 {
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
