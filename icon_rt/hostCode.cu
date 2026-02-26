// std
#include <fstream>
#include <string>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// icon_rt:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// ==================================================================
// cuda call
// ==================================================================

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif
#define CUDA_SAFE_CALL_X(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__, true); }

inline void cuda_safe_call(
    cudaError_t code, const char *file, int line, bool fatal=false)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
    if (fatal)
      exit(code);
  }
}

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(icon_rt::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance;
  int mode{TRIANGLE_MODE};
  bool accelActive;
#ifdef RTCORE
  OWLGroup trianglesTLAS, userGeomTLAS;
#endif
} g_appState;

namespace icon_rt {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingAE();
extern void woodcockTrackingWithAccel();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: icon_rt\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".ic"))
      g_appState.filepath = arg;
    else if (arg == "-mode") {
      g_appState.mode = std::atoi(argv[++i]);
    }
  }
}

static void toggleRayGen(Pipeline &pl) {
  static bool accelActive=true;
  if (g_appState.accelActive != accelActive) {
    if (g_appState.accelActive) {
#ifdef RTCORE
      pl.setRayGen("woodcockTrackingWithAccel");
#else
      pl.setRayGen(woodcockTrackingWithAccel);
#endif
    } else {
#ifdef RTCORE
      pl.setRayGen("woodcockTrackingAE");
#else
      pl.setRayGen(woodcockTrackingAE);
#endif
    }
    pl.resetAccumulation();
  }
  accelActive = g_appState.accelActive;
}

static void toggleMode(Pipeline &pl, LaunchParams &parms) {
#ifdef RTCORE
  static int mode=TRIANGLE_MODE;
  if (g_appState.mode != mode) {
    if (g_appState.mode==TRIANGLE_MODE) {
      owlParamsSetGroup(pl.owlLaunchParams(), "volume.handle", g_appState.trianglesTLAS);
    } else if (g_appState.mode==USER_GEOM_MODE) {
      owlParamsSetGroup(pl.owlLaunchParams(), "volume.handle", g_appState.userGeomTLAS);
    } else if (g_appState.mode==CUBQL_MODE) {
      //todo
    }
    pl.launchParam("volume.mode", parms.volume.mode) = g_appState.mode;
    pl.resetAccumulation();
  }
  mode = g_appState.mode;
#endif
}

__global__ void computeBounds(box3f *primBounds,
                              const vec3f *vertices,
                              const int *indices,
                              size_t numCells)
{
  size_t cellID = threadIdx.x+blockIdx.x*blockDim.x;
  if (cellID >= numCells) return;

  const int *I = indices + cellID*6;
  const vec3f v0 = vertices[I[0]];
  const vec3f v1 = vertices[I[1]];
  const vec3f v2 = vertices[I[2]];
  const vec3f v3 = vertices[I[3]];
  const vec3f v4 = vertices[I[4]];
  const vec3f v5 = vertices[I[5]];

  // primBounds:
  primBounds[cellID] = box3f(vec3f(1e31f),vec3f(-1e31f));
  primBounds[cellID].extend(v0);
  primBounds[cellID].extend(v1);
  primBounds[cellID].extend(v2);
  primBounds[cellID].extend(v3);
  primBounds[cellID].extend(v4);
  primBounds[cellID].extend(v5);
}

extern "C" int main(int argc, char *argv[]) {

  if (argc < 2) {
    printUsage();
    exit(-1);
  }

  parseCommandLine(argc, argv);

  if (g_appState.filepath.empty()) {
    printUsage();
    exit(-1);
  }

  std::ifstream in(g_appState.filepath);
  if (!in.good()) {
    printUsage();
    exit(-1);
  }

  size_t numCells{0};
  in.seekg(0,in.end);
  numCells = in.tellg()/sizeof(ICONCell);
  in.seekg(0,in.beg);

  std::vector<ICONCell> cells(numCells);
  in.read((char *)cells.data(),sizeof(ICONCell)*numCells);

  box3f volbounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  box1f dataRange(INFINITY, -INFINITY);

#if 0
  cells.clear();

  ICONCell cell;
  cell.lon.x = deg2rad(30);
  cell.lon.y = deg2rad(0);
  cell.lon.z = deg2rad(-30);

  cell.lat.x = deg2rad(0);
  cell.lat.y = deg2rad(90);
  cell.lat.z = deg2rad(0);

  cell.numLayers = 2;
  cell.height[0] = 100.f;
  cell.height[1] = 110.f;
  cell.height[2] = 120.f;
  cell.value[0] = 0.1f;
  cell.value[1] = 1.f;

  volbounds.extend(cell.getBounds());

  cells.push_back(cell);
#endif

  float innerRadius{INFINITY};
  float outerRadius{-INFINITY};
  for (int i=0; i<cells.size(); ++i) {
    ICONCell &cell = cells[i];
    innerRadius = fminf(innerRadius,cell.height[0]);
    outerRadius = fmaxf(outerRadius,cell.height[cell.numLayers]);
    volbounds.extend(cell.getBounds());
    for (int j=0; j<cell.numLayers; ++j) dataRange.extend(cell.value[j]);
  }

  Buffer deviceCells(cells.size(), cells.data());
  ICONGrid deviceGrid;
  deviceGrid.cells = deviceCells.data();
  deviceGrid.numCells = deviceCells.size();

  Pipeline pl(argc, argv, "icon_rt");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = dataRange;

    if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
    tf.setLUT(std::vector<vec4f>({
      {0.149f, 0.015f, 0.705f, 1.0f},
      {0.486f, 0.603f, 0.956f, 0.75f},
      {0.866f, 0.866f, 0.866f, 0.5f},
      {0.996f, 0.690f, 0.552f, 0.25f},
      {0.752f, 0.298f, 0.231f, 0.0f}
    }));
    pl.setTransfunc(&tf);
  }

  float magnitude = floorf(log10f(innerRadius));
  float scale = powf(10.f,magnitude-3);
  g_appState.unitDistance = 1.0f*scale;
  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.01f*scale, 5.f*scale);

  g_appState.accelActive = true;
  pl.uiParam("Use naive accel", &g_appState.accelActive);

  std::vector<std::string> options({
    "user geom mode",
    "triangle mode",
    "cuBQL mode",
  });
  pl.uiParam("Sampler mode", options, &g_appState.mode);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "woodcockTrackingWithAccel");
  pl.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  pl.setRayGen(woodcockTrackingWithAccel);
#endif

  LaunchParams parms;

#ifdef RTCORE
  // ######################################################
  // variant with triangle geometry
  // ######################################################

  std::vector<vec3f> vertex;
  std::vector<vec3i> index;
  for (size_t i=0; i<cells.size(); ++i) {
    const ICONCell &cell = cells[i];
    vec3f v1 = toCartesian({cell.height[0],cell.lat.x,cell.lon.x});
    vec3f v2 = toCartesian({cell.height[0],cell.lat.y,cell.lon.y});
    vec3f v3 = toCartesian({cell.height[0],cell.lat.z,cell.lon.z});
    vertex.push_back(v1);
    vertex.push_back(v2);
    vertex.push_back(v3);
    index.push_back({int(i)*3,int(i)*3+1,int(i)*3+2});
  }

  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(ICONTriangleGeom,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(ICONTriangleGeom,vertex)},
    { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType trianglesGeomType = owlGeomTypeCreate(pl.owlContext(),
                                                    OWL_TRIANGLES,
                                                    sizeof(ICONTriangleGeom),
                                                    trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType, 0, pl.owlModule(), "ICONTrianglesClosestHit");
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(pl.owlContext(),OWL_FLOAT3,vertex.size(),vertex.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(pl.owlContext(),OWL_INT3,index.size(),index.data());

  OWLGeom trianglesGeom = owlGeomCreate(pl.owlContext(),trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,vertex.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,index.size(),sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLGroup trianglesBLAS = owlTrianglesGeomGroupCreate(pl.owlContext(),1,&trianglesGeom);
  owlGroupBuildAccel(trianglesBLAS);

  g_appState.trianglesTLAS = owlInstanceGroupCreate(pl.owlContext(),1,&trianglesBLAS);
  owlGroupBuildAccel(g_appState.trianglesTLAS);


  // ######################################################
  // variant with user geometry
  // ######################################################

  OWLVarDecl iconGeomVars[]
  = {
     { "cells",  OWL_BUFPTR, OWL_OFFSETOF(ICONGrid,cells)},
     { "numCells",  OWL_UINT, OWL_OFFSETOF(ICONGrid,numCells)},
     { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType userGeomType = owlGeomTypeCreate(pl.owlContext(),
                                               OWL_GEOM_USER,
                                               sizeof(ICONGrid),
                                               iconGeomVars, -1);
  owlGeomTypeSetBoundsProg(userGeomType, pl.owlModule(), "ICONCellBounds");
  owlGeomTypeSetIntersectProg(userGeomType, 0, pl.owlModule(), "ICONCellIntersect");
  owlGeomTypeSetClosestHit(userGeomType, 0, pl.owlModule(), "ICONCellClosestHit");

  OWLGeom userGeom = owlGeomCreate(pl.owlContext(), userGeomType);
  owlGeomSetPrimCount(userGeom, cells.size());

  OWLBuffer cellBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                               OWL_USER_TYPE(ICONCell{}),
                                               cells.size(),
                                               cells.data());
  owlGeomSetBuffer(userGeom, "cells", cellBuffer);
  owlGeomSet1ui(userGeom, "numCells", (unsigned)cells.size());

  owlBuildPrograms(pl.owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pl.owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  g_appState.userGeomTLAS = owlInstanceGroupCreate(pl.owlContext(), 1);
  owlInstanceGroupSetChild(g_appState.userGeomTLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(g_appState.userGeomTLAS);


  // ######################################################
  // umesh variant
  // ######################################################
  std::vector<vec3f> vertexUMesh;
  std::vector<int> indexUMesh;
  std::vector<float> vertexScalars;
  for (size_t i=0; i<cells.size(); ++i) {
    const ICONCell &cell = cells[i];
    for (int h=0; h<cell.numLayers; ++h) {
      // bottom triangle:
      vec3f bv1 = toCartesian({cell.height[h],cell.lat.x,cell.lon.x});
      vec3f bv2 = toCartesian({cell.height[h],cell.lat.y,cell.lon.y});
      vec3f bv3 = toCartesian({cell.height[h],cell.lat.z,cell.lon.z});
      //top triangle:
      vec3f tv1 = toCartesian({cell.height[h+1],cell.lat.x,cell.lon.x});
      vec3f tv2 = toCartesian({cell.height[h+1],cell.lat.y,cell.lon.y});
      vec3f tv3 = toCartesian({cell.height[h+1],cell.lat.z,cell.lon.z});
      // value: (to-do: interpol)
      float v = cell.value[h];
      
      int idx0 = vertexUMesh.size();

      vertexUMesh.push_back(bv1); vertexScalars.push_back(v);
      vertexUMesh.push_back(bv2); vertexScalars.push_back(v);
      vertexUMesh.push_back(bv3); vertexScalars.push_back(v);

      vertexUMesh.push_back(tv1); vertexScalars.push_back(v);
      vertexUMesh.push_back(tv2); vertexScalars.push_back(v);
      vertexUMesh.push_back(tv3); vertexScalars.push_back(v);

      indexUMesh.push_back(idx0+0);
      indexUMesh.push_back(idx0+1);
      indexUMesh.push_back(idx0+2);
      indexUMesh.push_back(idx0+3);
      indexUMesh.push_back(idx0+4);
      indexUMesh.push_back(idx0+5);
    }
  }

  vec3f *d_vertices{nullptr};
  int *d_indices{nullptr};
  size_t numVertices = vertexUMesh.size();
  size_t numIndices = indexUMesh.size();
  size_t numWedges = indexUMesh.size()/6;

  CUDA_SAFE_CALL(cudaMalloc(&d_vertices, sizeof(vertexUMesh[0])*numVertices));
  CUDA_SAFE_CALL(cudaMemcpy(d_vertices,
                            vertexUMesh.data(),
                            sizeof(vertexUMesh[0])*numVertices,
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc(&d_indices, sizeof(indexUMesh[0])*numIndices));
  CUDA_SAFE_CALL(cudaMemcpy(d_indices,
                            indexUMesh.data(),
                            sizeof(indexUMesh[0])*numIndices,
                            cudaMemcpyHostToDevice));

  box3f *primBounds;
  CUDA_SAFE_CALL(cudaMalloc(&primBounds, sizeof(primBounds[0])*numWedges));

  computeBounds<<<iDivUp(numWedges,1024),1024>>>(primBounds,
                                                 d_vertices,
                                                 d_indices,
                                                 numWedges);

  bvh_t bvh;

  cuBQL::DeviceMemoryResource memResource;
  cuBQL::gpuBuilder(bvh,
                    (const cuBQL::box_t<float,3>*)primBounds,
                    numWedges,
                    cuBQL::BuildConfig(),
                    0,
                    memResource);

  CUDA_SAFE_CALL(cudaFree(primBounds));

  float *d_perVertex{nullptr};
  CUDA_SAFE_CALL(cudaMalloc(&d_perVertex, sizeof(vertexScalars[0])*numVertices));
  CUDA_SAFE_CALL(cudaMemcpy(d_perVertex,
                            vertexScalars.data(),
                            sizeof(vertexScalars[0])*numVertices,
                            cudaMemcpyHostToDevice));
#endif

  // volume
#ifdef RTCORE
  owlParamsSetGroup(pl.owlLaunchParams(), "volume.handle", g_appState.trianglesTLAS);
  pl.launchParam("volume.mode", parms.volume.mode) = TRIANGLE_MODE;
  pl.launchParam("volume.cubql.handle", (RawPointer &)parms.volume.cubql.handle) = &bvh;
  pl.launchParam("volume.cubql.vertices", (RawPointer &)parms.volume.cubql.vertices) = d_vertices;
  pl.launchParam("volume.cubql.indices", (RawPointer &)parms.volume.cubql.indices) = d_indices;
  pl.launchParam("volume.cubql.perVertex", (RawPointer &)parms.volume.cubql.perVertex) = d_perVertex;
#endif
  pl.launchParam("volume.cells", (RawPointer &)parms.volume.cells) = deviceCells.data();
  pl.launchParam("volume.numCells", parms.volume.numCells) = (int)deviceCells.size();
  pl.launchParam("volume.accel.innerRadius", parms.volume.accel.innerRadius) = innerRadius;
  pl.launchParam("volume.accel.outerRadius", parms.volume.accel.outerRadius) = outerRadius;
  pl.launchParam("volume.bounds", parms.volume.bounds) = volbounds;
  // lighting
  pl.launchParam("ambientColor", parms.ambientColor) = vec3f(1.f);
  pl.launchParam("ambientRadiance", parms.ambientRadiance) = 1.f;

  // Render and present...
  // For default (PNG image) pipeline this
  // loop returns immediately
  do {
    toggleRayGen(pl);
    toggleMode(pl,parms);

    struct {
      vec3f lower_left, horizontal, vertical;
    } screen;
    cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

    // update camera:
    pl.launchParam("camera.org", parms.camera.org) = cam.getPosition();
    pl.launchParam("camera.dir_00", parms.camera.dir_00) = screen.lower_left;
    pl.launchParam("camera.dir_du", parms.camera.dir_du) = screen.horizontal / imgWidth;
    pl.launchParam("camera.dir_dv", parms.camera.dir_dv) = screen.vertical / imgHeight;
    // update transfunc:
    pl.launchParam("transfunc.valueRange", parms.transfunc.valueRange) = pl.getTransfunc()->valueRange;
    pl.launchParam("transfunc.size", parms.transfunc.size) = pl.getTransfunc()->size;
    pl.launchParam("transfunc.values", (RawPointer &)parms.transfunc.values) = pl.getTransfunc()->rgbaLUT;
    // update framebuffer:
    pl.launchParam("fbPointer", (RawPointer &)parms.fbPointer) = fb.fbPointer;
    pl.launchParam("fbDepth", (RawPointer &)parms.fbDepth) = fb.fbDepth;
    pl.launchParam("accumBuffer", (RawPointer &)parms.accumBuffer) = fb.accumBuffer;
    // update DVR params:
    pl.launchParam("unitDistance", parms.unitDistance) = g_appState.unitDistance;
    // update accum:
    pl.launchParam("accumID", parms.accumID) = pl.frameID;

    // set params:
    SET_LAUNCH_PARAMS(parms);

    pl.launch();
    pl.present();
  } while (pl.isRunning());

  return 0;
}

} // namespace icon_rt



