import time
def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max: return "00:00:00"
    steps_left = t_max - t_current
    steps_done = t_current - t_start
    if steps_done == 0: return "?"
    time_taken = time.time() - start_time
    time_left = time_taken / steps_done * steps_left
    return time_str(time_left)
def time_str(s): return "{:02}:{:02}:{:02}".format(int(s) // 3600, int(s) % 3600 // 60, int(s) % 60)