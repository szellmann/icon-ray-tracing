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

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "semaphore_.h"

namespace dvr_course {

//=============================================================================
// Thread pool
//=============================================================================
class thread_pool {
public:
  explicit thread_pool(unsigned num_threads) {
    sync_params.start_threads = false;
    sync_params.join_threads = false;
    reset(num_threads);
  }

  ~thread_pool() {
    join_threads();
  }

  void reset(unsigned num_threads) {
    join_threads();

    threads.reset(new std::thread[num_threads]);
    this->num_threads = num_threads;

    for (unsigned i = 0; i < num_threads; ++i) {
      threads[i] = std::thread([this](){ thread_loop(); });
    }
  }

  void join_threads() {
    if (num_threads == 0)
        return;

    sync_params.start_threads = true;
    sync_params.join_threads = true;
    sync_params.threads_start.notify_all();

    for (unsigned i = 0; i < num_threads; ++i) {
      if (threads[i].joinable()) {
        threads[i].join();
      }
    }

    sync_params.start_threads = false;
    sync_params.join_threads = false;
    threads.reset(nullptr);
  }

  // Function to return an integer index in [0,N) given an opaque
  // thread handle
  unsigned get_thread_index(std::thread::id tid) {
    for (unsigned i = 0; i < num_threads; ++i) {
      if (threads[i].get_id() == tid) {
        return i;
      }
    }

    return unsigned(-1);
  }

  template <typename Func>
  void run(Func f, long queue_length) {
    // Set worker function
    func = f;

    // Set counters
    sync_params.num_work_items = queue_length;
    sync_params.work_item_counter = 0;
    sync_params.work_items_finished_counter = 0;

    // Activate persistent threads
    sync_params.start_threads = true;
    sync_params.threads_start.notify_all();

    // Wait for all threads to finish
    sync_params.threads_ready.wait();

    // Idle w/o work
    sync_params.start_threads = false;
  }

  std::unique_ptr<std::thread[]> threads;
  unsigned num_threads = 0;

private:
  using func_t = std::function<void(unsigned)>;
  func_t func;

  struct {
    std::mutex              mutex;
    std::condition_variable threads_start;
    dvr_course::semaphore   threads_ready;

    std::atomic<bool>       start_threads;
    std::atomic<bool>       join_threads;

    std::atomic<long>       num_work_items;
    std::atomic<long>       work_item_counter;
    std::atomic<long>       work_items_finished_counter;
  } sync_params;

  void thread_loop() {
    for (;;) {
      // Wait until activated
      {
        std::unique_lock<std::mutex> lock(sync_params.mutex);
        sync_params.threads_start.wait(lock,
            [this]() -> std::atomic<bool> const& {
                return sync_params.start_threads;
            });
      }

      // Exit?
      if (sync_params.join_threads)
        break;


      // Perform work in queue
      for (;;)  {
        auto work_item = sync_params.work_item_counter.fetch_add(1);

        if (work_item >= sync_params.num_work_items)
          break;

        func(work_item);

        auto finished = sync_params.work_items_finished_counter.fetch_add(1);

        if (finished >= sync_params.num_work_items - 1) {
          assert(finished == sync_params.num_work_items - 1);
          sync_params.threads_ready.notify();
          break;
        }
      }
    }
  }
};

} // dvr_course



