// Copyright 2025-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <netcdf.h>
#ifdef WITH_UMESH
# include "umesh/UMesh.h"
#endif

struct {
  std::string horizontalGridFile; // -hgrid
  std::string hsurfFile; // -hsurf
  std::vector<std::string> hhlFiles; // -hhl
  std::vector<std::string> dataFiles; // -data
  std::string outfileBase{"out"};
  bool convertToIC{false};
  bool convertToUMesh{true};
  int maxLayers{5};
} g_appState;

static void printHelp() {
  std::cout << "SYNOPSIS\n\n";
  std::cout << "Convert DWD ICON data to internal format used by our tool chain.\n";
  std::cout << "Given data from the DWD, we require the appropriate \"horizontal grid file\",\n";
  std::cout << "e.g., \"icon_grid_0026_R03B07_G.nc\" from http://icon-downloads.mpimet.mpg.de/dwd_grids.xml,\n";
  std::cout << "the time-invariant grid containing HSURF, e.g.:\n";
  std::cout << "\"icon_global_icosahedral_time-invariant_2026010300_HSURF.nc\",";
  std::cout << "the level height grids containing HHL from here:\n";
  std::cout << "https://opendata.dwd.de/weather/nwp/icon/grib/00/hhl/\n";
  std::cout << "and the grid files for the variable of interest in NetCDF format.\n";
  std::cout << "Data files can, e.g., be found here: https://opendata.dwd.de/weather/nwp/icon/grib/00/\n";
  std::cout << "Files in grib2 format must first be converted to NetCDF using:\n";
  std::cout << "cdo -f nc copy <in.grib2> <out.nc>\n";
  std::cout << "We assume that certain NetCDF dims and variables are present, such as \"height\".\n";
  std::cout << "In case this these are not present this script should be adapted accordingly....\n";
}

inline int div_up(int a, int b) {
  return (a+b-1)/b;
}

inline umesh::vec3f toCartesian(const umesh::vec3f spherical)
{
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
  return {x,y,z};
}

static size_t readDimLength(int ncid, std::string name) {
  int retval, dimid;
  if ((retval = nc_inq_dimid(ncid, name.c_str(), &dimid)) != NC_NOERR) {
    fprintf(stderr, "dim %s not found: %s\n", name.c_str(), nc_strerror(retval));
    return ~0ull;
  }

  size_t result;
  if ((retval = nc_inq_dimlen(ncid, dimid, &result)) != NC_NOERR) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return ~0ull;
  }

  return result;
}

static std::vector<int> readIntVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<int> result(len);

  if ((retval = nc_get_var_int(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}

static std::vector<double> readDoubleVar(int ncid, std::string name, size_t len) {
  int retval, varid;
  if ((retval = nc_inq_varid(ncid, name.c_str(), &varid)) != NC_NOERR) {
    fprintf(stderr, "variable %s not found\n", name.c_str());
    return {};
  }

  std::vector<double> result(len);

  if ((retval = nc_get_var_double(ncid, varid, result.data())) != NC_NOERR) {
    fprintf(stderr, "cannot read from variable %s\n", name.c_str());
    return {};
  }

  if (result.size() != len) {
    fprintf(stderr, "variable %s found but size mismatch\n", name.c_str());
    return {};
  }

  return result;
}


static void parseCommandLine(int argc, char *argv[]) {
  enum Mode { Hgrid, Hsurf, Hhl, Data, None, };
  Mode mode{None};

  for (int i=1; i<argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-') {
      if (mode == Hgrid) {
        g_appState.horizontalGridFile = arg;
      }
      else if (mode == Hsurf) {
        g_appState.hsurfFile = arg;
      }
      else if (mode == Hhl) {
        g_appState.hhlFiles.push_back(arg);
      }
      else if (mode == Data) {
        g_appState.dataFiles.push_back(arg);
      }
      else {
        fprintf(stderr, "Unknown parm: %s\n", argv[i]);
        break;
      }
    }
    else if (arg == "-hgrid") {
      mode = Hgrid;
    }
    else if (arg == "-hsurf") {
      mode = Hsurf;
    }
    else if (arg == "-hhl") {
      mode = Hhl;
    }
    else if (arg == "-data") {
      mode = Data;
    }
    else if (arg == "-o") {
      g_appState.outfileBase = argv[++i];
    }
  }
}

static void printUsage() {
  fprintf(stderr, "%s\n",
    "Usage: ./convert_icon -hgrid <hg.nc> -hsurf <hs.nc> -hhl [hh.nc*] -data [df.nc*]");
}

int main(int argc, char *argv[]) {
  if (argc < 3 || std::string(argv[1]) == "help" ) {
    printHelp();
    return 1;
  }

  parseCommandLine(argc, argv);

  if (g_appState.horizontalGridFile.empty() ||
      g_appState.hsurfFile.empty() ||
      g_appState.hhlFiles.empty() || g_appState.hsurfFile.empty()) {
    printUsage();
    return 1;
  }

  int ncid, retval;

  // Horizontal grid file:

  if ((retval = nc_open(g_appState.horizontalGridFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
    printf("Error opening file: %s\n", nc_strerror(retval));
    return 1;
  }

  // read number of cells:
  size_t cell = readDimLength(ncid, "cell");
  printf("number of cells: %i\n",(int)cell);

  // read number of vertices:
  size_t vertex = readDimLength(ncid, "vertex");
  printf("number of vertices: %i\n",(int)vertex);

  // read clon_vertices & clat_vertices:

  auto clon_vertices = readDoubleVar(ncid, "clon_vertices", cell*3);
  auto clat_vertices = readDoubleVar(ncid, "clat_vertices", cell*3);

  nc_close(ncid);

  if (clon_vertices.empty() || clat_vertices.empty()) {
    fprintf(stderr, "%s\n", "Cannot proceed as lon/lat coordinates missing");
    nc_close(ncid);
    return 1;
  }

  struct DataField {
    int height{0};
    std::vector<float> value;
  };

  // HSURF file:

  if ((retval = nc_open(g_appState.hsurfFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
    printf("Error opening file: %s\n", nc_strerror(retval));
    return 1;
  }

  auto hsurf = readDoubleVar(ncid, "HSURF", cell);

  if (clon_vertices.empty() || clat_vertices.empty()) {
    fprintf(stderr, "%s\n", "Cannot proceed as HSURF is ill-formed");
    nc_close(ncid);
    return 1;
  }


  // HHL files:

  std::vector<DataField> hhl(g_appState.hhlFiles.size());
  int hhlHeightBounds[2] = {INT_MAX, INT_MIN};

  for (int i=0; i<g_appState.hhlFiles.size(); ++i) {
    std::string hhlFile = g_appState.hhlFiles[i];
    if ((retval = nc_open(hhlFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    DataField &field = hhl[i];

    auto height = readDoubleVar(ncid, "height", 1);
    if (height.empty()) {
      fprintf(stderr, "No height found in %s, aborting...\n", hhlFile.c_str());
      nc_close(ncid);
      return 1;
    }

    field.height = (int)height[0];

    hhlHeightBounds[0] = std::min(hhlHeightBounds[0],field.height);
    hhlHeightBounds[1] = std::max(hhlHeightBounds[1],field.height);

    auto hhlI = readDoubleVar(ncid, "HHL", cell);
    if (hhlI.empty()) {
      fprintf(stderr, "HHL not propery read from %s, aborting...\n", hhlFile.c_str());
      nc_close(ncid);
      return 1;
    }

    for (int j=0; j<cell; ++j) {
      field.value.push_back((float)hhlI[j]);
    }

    nc_close(ncid);
  }

  std::sort(hhl.begin(), hhl.end(), [](auto &a, auto &b) { return a.height > b.height; });


  // Data files:

  std::vector<DataField> values(g_appState.dataFiles.size());
  int valuesHeightBounds[2] = {INT_MAX, INT_MIN};

  for (int i=0; i<g_appState.dataFiles.size(); ++i) {
    std::string dataFile = g_appState.dataFiles[i];
    if ((retval = nc_open(dataFile.c_str(), NC_NOWRITE, &ncid)) != NC_NOERR) {
      printf("Error opening file: %s\n", nc_strerror(retval));
      return 1;
    }

    DataField &field = values[i];

    // read number of cells:
    size_t ncells = readDimLength(ncid, "ncells");
    printf("number of cells IN DATA FILE: %i\n",(int)ncells);

    auto height = readDoubleVar(ncid, "height", 1);
    if (height.empty()) {
      fprintf(stderr, "No height found in %s, aborting...\n", dataFile.c_str());
      nc_close(ncid);
      return 1;
    }

    field.height = (int)height[0];

    valuesHeightBounds[0] = std::min(valuesHeightBounds[0],field.height);
    valuesHeightBounds[1] = std::max(valuesHeightBounds[1],field.height);

    // read VARIABLE
    const char *varname = "pres";
    auto var = readDoubleVar(ncid, varname, ncells);
    if (var.empty()) {
      fprintf(stderr, "Error reading variable %s, error: %s\n",
              varname, nc_strerror(retval));
      nc_close(ncid);
      return 1;
    }

#if 1
    double minValue(DBL_MAX);
    double maxValue(-DBL_MAX);
    for (int j=0; j<ncells; ++j) {
      minValue = fmin(minValue,var[j]);
      maxValue = fmax(maxValue,var[j]);
    }
    for (int j=0; j<ncells; ++j) {
      var[j] -= minValue;
      var[j] /= maxValue-minValue;
    }
#endif

    for (int j=0; j<cell; ++j) {
      field.value.push_back((float)var[j]);
    }

    nc_close(ncid);
  }

  std::sort(values.begin(), values.end(), [](auto &a, auto &b) { return a.height > b.height; });

  if (!(hhlHeightBounds[0] == valuesHeightBounds[0] && hhlHeightBounds[1] == valuesHeightBounds[1])) {
    fprintf(stderr, "%s\n", "Heights of HHL and data field don't match, aborting...");
  }

  int heightBounds[2] = { hhlHeightBounds[0], hhlHeightBounds[1] };

  int numLayers(g_appState.dataFiles.size());

  if (numLayers > g_appState.maxLayers) {
    numLayers = g_appState.maxLayers;
  }

  #define LMAX 32

  if (g_appState.convertToIC) {
    std::string outfileName = g_appState.outfileBase + ".ic";
    std::ofstream out(outfileName,std::ios::binary);
    for (int cellID=0; cellID<cell; ++cellID) {
      float lat[3]{(float)clat_vertices[cellID*3],(float)clat_vertices[cellID*3+1],(float)clat_vertices[cellID*3+2]};
      float lon[3]{(float)clon_vertices[cellID*3],(float)clon_vertices[cellID*3+1],(float)clon_vertices[cellID*3+2]};
      constexpr float R = 6.371229E6f;
      int valueIt = 0, hhlIt = 0;
      float prevH = R + hsurf[cellID];
      for (int i=0; i<div_up(numLayers,LMAX-1); ++i) {
        int numLayersLocal = LMAX-1;
        if ((i+1) * numLayersLocal > numLayers) {
          numLayersLocal = numLayers % LMAX-1;
        }

        float H[LMAX];
        H[0] = prevH;
        for (int j=1; j<=numLayersLocal; ++j) {
          H[j] = R + hhl[hhlIt++].value[cellID]-hsurf[cellID];
          prevH = H[j];
        }
        float value[LMAX];
        for (int j=0; j<numLayersLocal; ++j) {
          value[j] = values[valueIt++].value[cellID];
        }
        if (cellID == 0) {
          for (int j=0; j<numLayersLocal; ++j) {
            std::cout << j << ": " << H[j]/1000.f << ',' << value[j] << '\n';
          }
        }
        out.write((const char *)lat,sizeof(lat));
        out.write((const char *)lon,sizeof(lon));
        out.write((const char *)&numLayersLocal,sizeof(numLayersLocal));
        out.write((const char *)H,sizeof(H));
        out.write((const char *)value,sizeof(value));
      }
    }
    out.close();
  }

  if (g_appState.convertToUMesh) {
#ifdef WITH_UMESH
    using namespace umesh;
    auto output = std::make_shared<UMesh>();
    output->perVertex = std::make_shared<Attribute>();
    for (int cellID=0; cellID<cell; ++cellID) {
      float lat[3]{(float)clat_vertices[cellID*3],(float)clat_vertices[cellID*3+1],(float)clat_vertices[cellID*3+2]};
      float lon[3]{(float)clon_vertices[cellID*3],(float)clon_vertices[cellID*3+1],(float)clon_vertices[cellID*3+2]};;
      //float value[32];
      constexpr float R = 6.371229E6f;
      constexpr float scale = 50.f;
      for (int j=0; j<numLayers; ++j) {
        float h1 = j==0 ? R + hsurf[cellID]*scale
                     : R + (hhl[j].value[cellID]-hsurf[cellID])*scale;
        float h2 = R + (hhl[j+1].value[cellID]-hsurf[cellID])*scale;

        // bottom triangle vertices
        vec3f bv1 = toCartesian({h1,lat[0],lon[0]});
        vec3f bv2 = toCartesian({h1,lat[1],lon[1]});
        vec3f bv3 = toCartesian({h1,lat[2],lon[2]});
        // bottom value
        float bot = values[j].value[cellID]; // TODO: interpolate

        // top triangle vertices
        vec3f tv1 = toCartesian({h2,lat[0],lon[0]});
        vec3f tv2 = toCartesian({h2,lat[1],lon[1]});
        vec3f tv3 = toCartesian({h2,lat[2],lon[2]});
        // top value
        float top = values[j].value[cellID]; // TODO: interpolate

        output->vertices.push_back(bv1); output->perVertex->values.push_back(bot);
        output->vertices.push_back(bv2); output->perVertex->values.push_back(bot);
        output->vertices.push_back(bv3); output->perVertex->values.push_back(bot);
        output->vertices.push_back(tv1); output->perVertex->values.push_back(top);
        output->vertices.push_back(tv2); output->perVertex->values.push_back(top);
        output->vertices.push_back(tv3); output->perVertex->values.push_back(top);

        UMesh::Wedge wedge;
        wedge[0] = (int)output->vertices.size()-6;
        wedge[1] = (int)output->vertices.size()-5;
        wedge[2] = (int)output->vertices.size()-4;
        wedge[3] = (int)output->vertices.size()-3;
        wedge[4] = (int)output->vertices.size()-2;
        wedge[5] = (int)output->vertices.size()-1;

        output->wedges.push_back(wedge);

        h1 = h2;
      }
    }

    output->finalize();
    std::cout << output->vertices.size() << '\n';
    std::cout << output->wedges.size() << '\n';
    std::string outfileName = g_appState.outfileBase + ".umesh";
    output->saveTo(outfileName);
#else
    std::cerr << "Not compiled with support for UMesh files!\n";
#endif
  }
}
