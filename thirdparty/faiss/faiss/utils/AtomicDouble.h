//
// Created by dcy on 2022/6/27.
//

#ifndef FAISS_ATOMICDOUBLE_H
#define FAISS_ATOMICDOUBLE_H

#include <atomic>
#include "StopWatch.h"

class AtomicDouble {
   private:
    std::atomic<double> data;

   public:
    explicit AtomicDouble() {
        std::atomic_init(&data, 0.0);
    }
    explicit AtomicDouble(double value_) {
        std::atomic_init(&data, value_);
    }
    explicit AtomicDouble(const AtomicDouble& value_) {
        std::atomic_init(&data, value_.data.load());
    }
    void addMilliSecond(double increase_value) {
        increase_value = increase_value / 1000;
        for (double g = data.load();
             !data.compare_exchange_strong(g, g + increase_value);)
            ;
    }
    void add(StopWatch watch) {
        double increase_value = watch.getElapsedTime() / 1000;
        for (double g = data.load();
             !data.compare_exchange_strong(g, g + increase_value);)
            ;
    }

    void sub(StopWatch watch) {
        double increase_value = watch.getElapsedTime() / 1000;
        for (double g = data.load();
             !data.compare_exchange_strong(g, g - increase_value);)
            ;
    }
    double getValue() {
        return data.load();
    }
    AtomicDouble& operator=(const AtomicDouble& tmp) {
        return *this;
    }
};

#endif // FAISS_ATOMICDOUBLE_H
