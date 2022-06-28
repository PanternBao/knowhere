//
// Created by dcy on 2022/6/27.
//

#ifndef FAISS_STOPWATCH_H
#define FAISS_STOPWATCH_H
#include <stdexcept>
#include <faiss/utils/utils.h>
class StopWatch {
    double beginTime = 0;
    double endTime = 0;

   public:
    static StopWatch start() {
        StopWatch sw;
        sw.beginTime = faiss::getmillisecs();
        return sw;
    }
    void stop() {
        endTime = faiss::getmillisecs();
    }

    void restart() {
        beginTime = faiss::getmillisecs();
        endTime = 0;
    }
    double getElapsedTime() {
        if (beginTime == 0 || endTime == 0) {
            throw std::invalid_argument("illegal usage");
        }
        return endTime - beginTime;
    }
};

#endif // FAISS_STOPWATCH_H
