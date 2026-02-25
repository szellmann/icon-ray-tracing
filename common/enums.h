// ======================================================================== //
// Copyright 2025-2025 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// ========================================================
// Common enumerations
// ========================================================
namespace dvr_course {

// plain copy from owl:
typedef enum
  {
   DVR_INVALID_TYPE = 0,

   DVR_BUFFER=10,
   /*! a 64-bit int representing the number of elemnets in a buffer */
   DVR_BUFFER_SIZE,
   DVR_BUFFER_ID,
   DVR_BUFFER_POINTER,
   DVR_BUFPTR=DVR_BUFFER_POINTER,

   DVR_GROUP=20,

   /*! implicit variable of type integer that specifies the *index*
     of the given device. this variable type is implicit in the
     sense that it only gets _declared_ on the host, and gets set
     automatically during SBT creation */
   DVR_DEVICE=30,

   /*! texture(s) */
   DVR_TEXTURE=40,
   DVR_TEXTURE_2D=DVR_TEXTURE,


   /* all types that are naively copyable should be below this value,
      all that aren't should be above */
   _DVR_BEGIN_COPYABLE_TYPES = 1000,
   
   
   DVR_FLOAT=1000,
   DVR_FLOAT2,
   DVR_FLOAT3,
   DVR_FLOAT4,

   DVR_INT=1010,
   DVR_INT2,
   DVR_INT3,
   DVR_INT4,
   
   DVR_UINT=1020,
   DVR_UINT2,
   DVR_UINT3,
   DVR_UINT4,
   
   DVR_LONG=1030,
   DVR_LONG2,
   DVR_LONG3,
   DVR_LONG4,

   DVR_ULONG=1040,
   DVR_ULONG2,
   DVR_ULONG3,
   DVR_ULONG4,

   DVR_DOUBLE=1050,
   DVR_DOUBLE2,
   DVR_DOUBLE3,
   DVR_DOUBLE4,
    
   DVR_CHAR=1060,
   DVR_CHAR2,
   DVR_CHAR3,
   DVR_CHAR4,

   /*! unsigend 8-bit integer */
   DVR_UCHAR=1070,
   DVR_UCHAR2,
   DVR_UCHAR3,
   DVR_UCHAR4,

   DVR_SHORT=1080,
   DVR_SHORT2,
   DVR_SHORT3,
   DVR_SHORT4,

   /*! unsigend 8-bit integer */
   DVR_USHORT=1090,
   DVR_USHORT2,
   DVR_USHORT3,
   DVR_USHORT4,

   DVR_BOOL,
   DVR_BOOL2,
   DVR_BOOL3,
   DVR_BOOL4,
   
   /*! just another name for a 64-bit data type - unlike
     DVR_BUFFER_POINTER's (which gets translated from DVRBuffer's
     to actual device-side poiners) these DVR_RAW_POINTER types get
     copied binary without any translation. This is useful for
     owl-cuda interaction (where the user already has device
     pointers), but should not be used for logical buffers */
   DVR_RAW_POINTER=DVR_ULONG,
   DVR_BYTE = DVR_UCHAR,
   // DVR_BOOL = DVR_UCHAR,
   // DVR_BOOL2 = DVR_UCHAR2,
   // DVR_BOOL3 = DVR_UCHAR3,
   // DVR_BOOL4 = DVR_UCHAR4,


   /* matrix formats */
   DVR_AFFINE3F=1300,

   /*! at least for now, use that for buffers with user-defined types:
     type then is "DVR_USER_TYPE_BEGIN+sizeof(elementtype). Note
     that since we always _add_ the user type's size to this value
     this MUST be the last entry in the enum */
   DVR_USER_TYPE_BEGIN=10000
  }
  DVRDataType;

} // namespace dvr_course



