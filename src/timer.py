

class Timer:
    def __init__(self, seed):
        self.seed = seed

    def restart_timer(self):
        self.description = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def start(self, description):
        self.restart_timer()
        self.description = description
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.duration = self.end_time-self.start_time
        print(f"{self.description} took {self.duration}s. ")
        self._log_time()

    def _log_time(self):
        with open(fp_time_log, "a+") as myfile:
            myfile.write(f"{self.seed},{self.description},{self.duration}\n")