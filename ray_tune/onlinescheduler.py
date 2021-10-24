
from ray.tune.schedulers import TrialScheduler
import time

class OnlineScheduler(TrialScheduler):
    """Simple scheduler that just runs trials in submission order."""

    def __init__(self, scheduler):

        self.scheduler = scheduler
        self.start_time = time.time()

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        return self.scheduler.on_trial_add(trial_runner, trial)

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner",
                       trial: Trial):
        return self.scheduler.on_trial_error(trial_runner, trial)

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        return self.scheduler.on_trial_result(trial_runner, trial, result)

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        return self.scheduler.on_trial_complete(trial_runner, trial, result)

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial):
        return self.scheduler.on_trial_remove(trial_runner, trial)

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:

        trial = self.scheduler.choose_trial_to_run(trial_runner)

        if trial is None:
            return trial
        # get submission time
        pending_time = trial.config.get('arrival', 0) - (time.time() - self.start_time) 
        if pending_time > 0:
            time.sleep(pending_time)

        return trial
        
        # for trial in trial_runner.get_trials():
        #     if (trial.status == Trial.PENDING
        #             and trial_runner.has_resources_for_trial(trial)):
        #         return trial
        # for trial in trial_runner.get_trials():
        #     if (trial.status == Trial.PAUSED
        #             and trial_runner.has_resources_for_trial(trial)):
        #         return trial
        # return None

    def debug_string(self) -> str:
        return self.scheduler.debug_string()
