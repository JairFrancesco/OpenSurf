# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jairfrancesco/OpenSurf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jairfrancesco/OpenSurf

# Include any dependencies generated for this target.
include CMakeFiles/surf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/surf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/surf.dir/flags.make

CMakeFiles/surf.dir/surf.cpp.o: CMakeFiles/surf.dir/flags.make
CMakeFiles/surf.dir/surf.cpp.o: surf.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jairfrancesco/OpenSurf/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/surf.dir/surf.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surf.dir/surf.cpp.o -c /home/jairfrancesco/OpenSurf/surf.cpp

CMakeFiles/surf.dir/surf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surf.dir/surf.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jairfrancesco/OpenSurf/surf.cpp > CMakeFiles/surf.dir/surf.cpp.i

CMakeFiles/surf.dir/surf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surf.dir/surf.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jairfrancesco/OpenSurf/surf.cpp -o CMakeFiles/surf.dir/surf.cpp.s

CMakeFiles/surf.dir/surf.cpp.o.requires:
.PHONY : CMakeFiles/surf.dir/surf.cpp.o.requires

CMakeFiles/surf.dir/surf.cpp.o.provides: CMakeFiles/surf.dir/surf.cpp.o.requires
	$(MAKE) -f CMakeFiles/surf.dir/build.make CMakeFiles/surf.dir/surf.cpp.o.provides.build
.PHONY : CMakeFiles/surf.dir/surf.cpp.o.provides

CMakeFiles/surf.dir/surf.cpp.o.provides.build: CMakeFiles/surf.dir/surf.cpp.o

CMakeFiles/surf.dir/utils.cpp.o: CMakeFiles/surf.dir/flags.make
CMakeFiles/surf.dir/utils.cpp.o: utils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jairfrancesco/OpenSurf/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/surf.dir/utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surf.dir/utils.cpp.o -c /home/jairfrancesco/OpenSurf/utils.cpp

CMakeFiles/surf.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surf.dir/utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jairfrancesco/OpenSurf/utils.cpp > CMakeFiles/surf.dir/utils.cpp.i

CMakeFiles/surf.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surf.dir/utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jairfrancesco/OpenSurf/utils.cpp -o CMakeFiles/surf.dir/utils.cpp.s

CMakeFiles/surf.dir/utils.cpp.o.requires:
.PHONY : CMakeFiles/surf.dir/utils.cpp.o.requires

CMakeFiles/surf.dir/utils.cpp.o.provides: CMakeFiles/surf.dir/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/surf.dir/build.make CMakeFiles/surf.dir/utils.cpp.o.provides.build
.PHONY : CMakeFiles/surf.dir/utils.cpp.o.provides

CMakeFiles/surf.dir/utils.cpp.o.provides.build: CMakeFiles/surf.dir/utils.cpp.o

CMakeFiles/surf.dir/ipoint.cpp.o: CMakeFiles/surf.dir/flags.make
CMakeFiles/surf.dir/ipoint.cpp.o: ipoint.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jairfrancesco/OpenSurf/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/surf.dir/ipoint.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surf.dir/ipoint.cpp.o -c /home/jairfrancesco/OpenSurf/ipoint.cpp

CMakeFiles/surf.dir/ipoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surf.dir/ipoint.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jairfrancesco/OpenSurf/ipoint.cpp > CMakeFiles/surf.dir/ipoint.cpp.i

CMakeFiles/surf.dir/ipoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surf.dir/ipoint.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jairfrancesco/OpenSurf/ipoint.cpp -o CMakeFiles/surf.dir/ipoint.cpp.s

CMakeFiles/surf.dir/ipoint.cpp.o.requires:
.PHONY : CMakeFiles/surf.dir/ipoint.cpp.o.requires

CMakeFiles/surf.dir/ipoint.cpp.o.provides: CMakeFiles/surf.dir/ipoint.cpp.o.requires
	$(MAKE) -f CMakeFiles/surf.dir/build.make CMakeFiles/surf.dir/ipoint.cpp.o.provides.build
.PHONY : CMakeFiles/surf.dir/ipoint.cpp.o.provides

CMakeFiles/surf.dir/ipoint.cpp.o.provides.build: CMakeFiles/surf.dir/ipoint.cpp.o

# Object files for target surf
surf_OBJECTS = \
"CMakeFiles/surf.dir/surf.cpp.o" \
"CMakeFiles/surf.dir/utils.cpp.o" \
"CMakeFiles/surf.dir/ipoint.cpp.o"

# External object files for target surf
surf_EXTERNAL_OBJECTS =

libsurf.a: CMakeFiles/surf.dir/surf.cpp.o
libsurf.a: CMakeFiles/surf.dir/utils.cpp.o
libsurf.a: CMakeFiles/surf.dir/ipoint.cpp.o
libsurf.a: CMakeFiles/surf.dir/build.make
libsurf.a: CMakeFiles/surf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libsurf.a"
	$(CMAKE_COMMAND) -P CMakeFiles/surf.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/surf.dir/build: libsurf.a
.PHONY : CMakeFiles/surf.dir/build

CMakeFiles/surf.dir/requires: CMakeFiles/surf.dir/surf.cpp.o.requires
CMakeFiles/surf.dir/requires: CMakeFiles/surf.dir/utils.cpp.o.requires
CMakeFiles/surf.dir/requires: CMakeFiles/surf.dir/ipoint.cpp.o.requires
.PHONY : CMakeFiles/surf.dir/requires

CMakeFiles/surf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/surf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/surf.dir/clean

CMakeFiles/surf.dir/depend:
	cd /home/jairfrancesco/OpenSurf && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jairfrancesco/OpenSurf /home/jairfrancesco/OpenSurf /home/jairfrancesco/OpenSurf /home/jairfrancesco/OpenSurf /home/jairfrancesco/OpenSurf/CMakeFiles/surf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/surf.dir/depend

