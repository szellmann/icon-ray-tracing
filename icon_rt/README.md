# Ex. 05: Hey ICON..

This example demonstrates implementing a more complex volume element type than
voxels, namely the icosahedron shape common in climate and weather models such
as ICON. The example uses OptiX and OWL for cell location; also, a very
simplistic traversal accelerator for Woodcock tracking ICON data inside a
spherical shell is implemented, in lieu of the more general traversal
structures discussed in later chapters. The sample generally works on the CPU,
but doesn't use a BVH so will be very slow.

## TODOs:
- [x] Make accel and non-accel code paths use separate ray gen progs. The
      reason for this being that it is nice show that the ray gen prog per se
      isn't any different than the ray gen used in prior examples, only the
      element type changed
- [ ] Decide if we keep both OptiX user geom and OptiX triangle sampler
