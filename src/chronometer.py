import time
import matplotlib.pyplot as plt


class Singleton(object):
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class Chronometer(Singleton):

    NON_TRAINING_CHRONO = 0
    TRAINING_CHRONO = 1

    def __init__(self):
        super().__init__()
        if hasattr(self, '_current_start_time'):
            return

        self._current_start_time = { Chronometer.NON_TRAINING_CHRONO: 0, Chronometer.TRAINING_CHRONO: 0}
        self._current_delta_time = { Chronometer.NON_TRAINING_CHRONO: 0, Chronometer.TRAINING_CHRONO: 0 }
        self._delta_times_saved = { Chronometer.NON_TRAINING_CHRONO: [0], Chronometer.TRAINING_CHRONO: [0] }
        self._chrono_paused = { Chronometer.NON_TRAINING_CHRONO: True, Chronometer.TRAINING_CHRONO: True }

    def plot_chrono_deltas(self):
        # print("Training deltas: ")
        # print(self._delta_times_saved[Chronometer.TRAINING_CHRONO])
        # print("Non Training deltas: ")
        # print(self._delta_times_saved[Chronometer.NON_TRAINING_CHRONO])

        plt.plot(self._delta_times_saved[Chronometer.TRAINING_CHRONO],label="Training")
        plt.plot(self._delta_times_saved[Chronometer.NON_TRAINING_CHRONO], label="non Training")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

    def resume_chrono(self, chrono_to_use):
        if self._chrono_paused[chrono_to_use]:
            self._current_start_time[chrono_to_use] = time.time()
            self._chrono_paused[chrono_to_use] = False
        # else:
        #     print("Chrono already resumed")

    def pause_chrono(self, chrono_to_use):
        if not self._chrono_paused[chrono_to_use]:
            self._current_delta_time[chrono_to_use] += time.time() - self._current_start_time[chrono_to_use]
            self._chrono_paused[chrono_to_use] = True
        # else:
        #     print("Chrono already paused")

    def checkpoint_chrono(self, chrono_to_use, dividor=1):
        # debug
        # if chrono_to_use == Chronometer.TRAINING_CHRONO:
        #     print(type(self._current_delta_time[chrono_to_use]))
        #     print("%f / %f = %f" % (self._current_delta_time[chrono_to_use], dividor, self._current_delta_time[chrono_to_use] / dividor))

        self._delta_times_saved[chrono_to_use].append(self._current_delta_time[chrono_to_use] / dividor)
        self._current_start_time[chrono_to_use] = 0
        self._current_delta_time[chrono_to_use] = 0
