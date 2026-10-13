include(GoogleTest)

add_executable(trtbench_core_tests
  test_stats.cpp
  test_clocks.cpp
  test_roofline.cpp
  test_audit.cpp
  test_finalize.cpp
  test_compare.cpp
  test_json.cpp
  test_traps.cpp
  test_layer_profile.cpp
)
target_link_libraries(trtbench_core_tests PRIVATE trtbench_core GTest::gtest_main)
target_compile_options(trtbench_core_tests PRIVATE ${TB_HOST_WARNINGS})
gtest_discover_tests(trtbench_core_tests
  PROPERTIES ENVIRONMENT "TRTBENCH_SAMPLE_JSON_OUT=${CMAKE_BINARY_DIR}/sample_report.json")

if(TB_ENABLE_CUDA)
  # Each test GTEST_SKIPs when no CUDA device is present, so CI without a GPU
  # exercises the skip path and a GPU box runs the real thing.
  add_executable(trtbench_gpu_tests test_runner_gpu.cpp)
  target_link_libraries(trtbench_gpu_tests PRIVATE trtbench_cuda GTest::gtest_main)
  target_compile_options(trtbench_gpu_tests PRIVATE ${TB_HOST_WARNINGS})
  gtest_discover_tests(trtbench_gpu_tests DISCOVERY_MODE PRE_TEST)
endif()
