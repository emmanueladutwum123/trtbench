# Record which source produced a result file. Evaluated at configure time, so
# re-run cmake after committing if the SHA in result JSON must be exact.
set(TB_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _sha OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc ERROR_QUIET)
  if(_rc EQUAL 0)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(TB_GIT_SHA "${_sha}")
    if(_dirty)
      set(TB_GIT_SHA "${_sha}-dirty")
    endif()
  endif()
endif()
