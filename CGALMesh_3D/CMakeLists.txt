cmake_minimum_required(VERSION 3.1...3.23)
project(Mesh_3_Examples)

add_definitions(-DCGAL_MESH_3_NO_DEPRECATED_SURFACE_INDEX
                -DCGAL_MESH_3_NO_DEPRECATED_C3T3_ITERATORS)

if(CGAL_MESH_3_VERBOSE)
  add_definitions(-DCGAL_MESH_3_VERBOSE)
endif()

find_package(CGAL REQUIRED COMPONENTS ImageIO)

option(CGAL_ACTIVATE_CONCURRENT_MESH_3 "Activate parallelism in Mesh_3" ON)

if(CGAL_ACTIVATE_CONCURRENT_MESH_3 OR "$ENV{CGAL_ACTIVATE_CONCURRENT_MESH_3}")
  message(STATUS "Adding TBB")
  add_definitions(-DCGAL_CONCURRENT_MESH_3)
  find_package(TBB REQUIRED)
  include(CGAL_TBB_support)
  if(NOT TARGET CGAL::TBB_support)
	  message("Notice: TBB support could not be found.")
	  return()
   endif()
endif()

# Use Eigen
find_package(Eigen3 3.1.0 QUIET) #(3.1.0 or greater)
include(CGAL_Eigen3_support)
if(NOT TARGET CGAL::Eigen3_support)
  message("NOTICE: All examples require the Eigen3 library, and will not be compiled.")
  return()
endif()

find_package(VTK QUIET COMPONENTS vtkImagingGeneral vtkIOImage NO_MODULE)
if(VTK_FOUND)
  if(VTK_USE_FILE)
    include(${VTK_USE_FILE})
  endif()
  if("${VTK_VERSION_MAJOR}" GREATER "5" OR VTK_VERSION VERSION_GREATER 5)
    message(STATUS "VTK found — NIfTI support enabled")
    add_definitions(-DCGAL_USE_VTK)
    if(TARGET VTK::IOImage)
      set(VTK_LIBRARIES VTK::ImagingGeneral VTK::IOImage)
    endif()
  else()
    message(STATUS "VTK version 6.0 or greater is required")
  endif()
else()
  message(STATUS "VTK was not found — only INR format supported")
endif()

find_package(TBB REQUIRED)
include(CGAL_TBB_support)

set(_itk_variants
  mesh_3D_image_without_features
  mesh_3D_image_with_polyline_features
  mesh_3D_image_with_weight_and_features)

if(TARGET CGAL::CGAL_ImageIO AND CGAL_ImageIO_USE_ZLIB)
  # mesh_3D_image no longer needs ITK after extracting mesh_image.cpp.
  create_single_source_cgal_program("mesh_3D_image.cpp")
  target_sources(mesh_3D_image PRIVATE mesh_image.cpp)
  target_link_libraries(mesh_3D_image PUBLIC CGAL::Eigen3_support)

  find_package(ITK NAMES ITK InsightToolkit QUIET COMPONENTS ITKCommon ITKThresholding ITKSmoothing ITKImageIntensity)
  if(ITK_FOUND)
    include(CGAL_ITK_support)
    foreach(_v IN LISTS _itk_variants)
      create_single_source_cgal_program("${_v}.cpp")
      target_link_libraries(${_v} PUBLIC CGAL::Eigen3_support CGAL::ITK_support)
    endforeach()
  else()
    message(STATUS "NOTICE: ITK not found — only mesh_3D_image will be built; other variants need ITK.")
  endif()
endif()

if(CGAL_ACTIVATE_CONCURRENT_MESH_3 AND TARGET CGAL::TBB_support)
  message(STATUS "NOTICE: Parallel meshing.")
  foreach(_t mesh_3D_image ${_itk_variants})
    if(TARGET ${_t})
      target_link_libraries(${_t} PUBLIC CGAL::TBB_support)
    endif()
  endforeach()
endif()

if(VTK_FOUND AND VTK_LIBRARIES)
  message(STATUS "Linking VTK for NIfTI support")
  foreach(_t mesh_3D_image ${_itk_variants})
    if(TARGET ${_t})
      target_link_libraries(${_t} PUBLIC ${VTK_LIBRARIES})
    endif()
  endforeach()
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
if(Python_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nb_query_result)
  if(_nb_query_result EQUAL 0)
    find_package(nanobind CONFIG QUIET HINTS "${nanobind_ROOT}")
  endif()
endif()

if(Python_FOUND AND nanobind_FOUND)
  message(STATUS "Building Python module cgalmesh3d via nanobind")
  nanobind_add_module(cgalmesh3d MODULE bindings.cpp mesh_image.cpp)
  target_link_libraries(cgalmesh3d PRIVATE CGAL::Eigen3_support CGAL::CGAL_ImageIO)
  if(CGAL_ACTIVATE_CONCURRENT_MESH_3 AND TARGET CGAL::TBB_support)
    target_link_libraries(cgalmesh3d PRIVATE CGAL::TBB_support)
  endif()
  if(VTK_FOUND AND VTK_LIBRARIES)
    target_link_libraries(cgalmesh3d PRIVATE ${VTK_LIBRARIES})
    # Embed VTK's library directory in the module's rpath so it loads without
    # LD_LIBRARY_PATH. The directory is taken from the imported VTK targets.
    get_target_property(_vtk_iface_loc VTK::IOImage IMPORTED_LOCATION_RELEASE)
    if(NOT _vtk_iface_loc)
      get_target_property(_vtk_iface_loc VTK::IOImage LOCATION)
    endif()
    if(_vtk_iface_loc)
      get_filename_component(_vtk_libdir "${_vtk_iface_loc}" DIRECTORY)
      set_target_properties(cgalmesh3d PROPERTIES
        INSTALL_RPATH "${_vtk_libdir}"
        BUILD_RPATH "${_vtk_libdir}")
    endif()
  endif()
  install(TARGETS cgalmesh3d LIBRARY DESTINATION .)
endif()
