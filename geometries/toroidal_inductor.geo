SetFactory("OpenCASCADE");

// -------------------- Parameters --------------------
R  = 2.0;    // major radius of the torus centerline
rc = 0.60;   // core minor radius (iron core surface)
rw = 0.85;   // outer radius of winding-pack region (where J lives)
L  = 6.0;    // outer truncation half-size (box)

// Mesh sizes
h_coil = 0.06;   // target size near/in coil pack
h_air  = 0.25;   // target size far from coil
t_grad = 0.70;   // thickness of size transition band

// -------------------- Hard constraints for your loader --------------------
// Force linear (1st order) elements only
Mesh.ElementOrder = 1;
// Make sure no "recombine" into quads/hexes happens anywhere
Mesh.RecombineAll = 0;
// Write .msh 4.1
Mesh.MshFileVersion = 4.1;

// -------------------- Geometry --------------------
// Outer truncation region
Box(1) = {-L, -L, -L, 2*L, 2*L, 2*L};

// Iron core (excluded from mesh): solid torus of minor radius rc
Torus(2) = {0, 0, 0, R, rc, 2*Pi};

// "Winding pack" region (for refinement / J-support): shell between rc and rw
Torus(3) = {0, 0, 0, R, rw, 2*Pi};
Torus(4) = {0, 0, 0, R, rc, 2*Pi};
coilTmp[] = BooleanDifference{ Volume{3}; Delete; }{ Volume{4}; Delete; };

// Domain without core: (outer box) \ (core)
Omega0[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Fragment to ensure coil is a conforming subvolume of Omega
frags[] = BooleanFragments{ Volume{Omega0[0]}; Delete; }{ Volume{coilTmp[0]}; Delete; };

// -------------------- Identify coil volume (bounding box around the torus) --------------------
eps = 1e-3;
coilVol[] = Volume In BoundingBox{-(R+rw+eps), -(R+rw+eps), -(rw+eps),
                                   (R+rw+eps),  (R+rw+eps),  (rw+eps)};

// -------------------- Mesh size field: refine near coil --------------------
// Use distance to the coil boundary surfaces as an attractor.
coilSurfs[] = Boundary{ Volume{coilVol[]}; };

Field[1] = Distance;
Field[1].FacesList = {coilSurfs[]};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_coil;
Field[2].SizeMax = h_air;
Field[2].DistMin = 0.0;
Field[2].DistMax = t_grad;

Background Field = 2;

// When fields fully control sizes, turn off other size heuristics
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

// Physical groups are optional; your loader ignores them anyway.
// Keeping them doesn't break parsing, but don't rely on them unless you extend the loader.
Physical Volume("Omega") = {frags[]};
Physical Volume("coil")  = {coilVol[]};

Mesh 3;
