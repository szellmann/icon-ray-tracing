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

#if defined(_WIN32) || defined(__APPLE__)
# define SEMAPHORE_USE_STD 1
#endif

#ifdef SEMAPHORE_USE_STD
# include <atomic>
# include <condition_variable>
# include <mutex>
#else
# include <semaphore.h>
#endif

namespace dvr_course {

#ifdef SEMAPHORE_USE_STD

class semaphore {
public:
  explicit semaphore(unsigned count = 0) : count_(count)
  {}

  void notify() {
    std::unique_lock<std::mutex> l(mutex_);
    ++count_;
    cond_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> l(mutex_);
    cond_.wait(l, [this]() { return count_ > 0; });
    count_--;
  }

private:
    std::condition_variable cond_;
    std::mutex mutex_;
    std::atomic<unsigned> count_;
};

#else

class semaphore
{
public:

  explicit semaphore(unsigned count = 0) {
    sem_init(&sem_, 0, count);
  }

  ~semaphore() {
    sem_close(&sem_);
    sem_destroy(&sem_);
  }

  void notify() {
    sem_post(&sem_);
  }

  void wait() {
    sem_wait(&sem_);
  }

private:
    sem_t sem_;
};

#endif

} // dvr_course


