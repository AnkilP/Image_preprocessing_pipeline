# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /JetBrains/Toolbox/apps/CLion/ch-0/182.3911.40/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /JetBrains/Toolbox/apps/CLion/ch-0/182.3911.40/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ClionProjects/Image_preprocessing_pipeline

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /ClionProjects/Image_preprocessing_pipeline/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Image_preprocessing_pipeline.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Image_preprocessing_pipeline.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Image_preprocessing_pipeline.dir/flags.make

CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o: CMakeFiles/Image_preprocessing_pipeline.dir/flags.make
CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o: ../adaptive_manifold.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ClionProjects/Image_preprocessing_pipeline/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o -c /ClionProjects/Image_preprocessing_pipeline/adaptive_manifold.cpp

CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ClionProjects/Image_preprocessing_pipeline/adaptive_manifold.cpp > CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.i

CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ClionProjects/Image_preprocessing_pipeline/adaptive_manifold.cpp -o CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.s

# Object files for target Image_preprocessing_pipeline
Image_preprocessing_pipeline_OBJECTS = \
"CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o"

# External object files for target Image_preprocessing_pipeline
Image_preprocessing_pipeline_EXTERNAL_OBJECTS =

Image_preprocessing_pipeline: CMakeFiles/Image_preprocessing_pipeline.dir/adaptive_manifold.cpp.o
Image_preprocessing_pipeline: CMakeFiles/Image_preprocessing_pipeline.dir/build.make
Image_preprocessing_pipeline: CMakeFiles/Image_preprocessing_pipeline.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ClionProjects/Image_preprocessing_pipeline/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Image_preprocessing_pipeline"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Image_preprocessing_pipeline.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Image_preprocessing_pipeline.dir/build: Image_preprocessing_pipeline

.PHONY : CMakeFiles/Image_preprocessing_pipeline.dir/build

CMakeFiles/Image_preprocessing_pipeline.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Image_preprocessing_pipeline.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Image_preprocessing_pipeline.dir/clean

CMakeFiles/Image_preprocessing_pipeline.dir/depend:
	cd /ClionProjects/Image_preprocessing_pipeline/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ClionProjects/Image_preprocessing_pipeline /ClionProjects/Image_preprocessing_pipeline /ClionProjects/Image_preprocessing_pipeline/cmake-build-debug /ClionProjects/Image_preprocessing_pipeline/cmake-build-debug /ClionProjects/Image_preprocessing_pipeline/cmake-build-debug/CMakeFiles/Image_preprocessing_pipeline.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Image_preprocessing_pipeline.dir/depend

