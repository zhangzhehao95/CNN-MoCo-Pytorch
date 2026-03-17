class StopCriteria_NoImprove(object):
    """
        Analyze whether the loss gets at lease min_improve in the latest query_len iterations.
        Add the validation loss of each epoch into the criteria
        If stop criteria are met, return True
    """
    def __init__(self, query_len=5, num_min_epoch=10, min_improve=0):
        self.patience = query_len    # How many latest losses will be analysed
        self.num_min_epoch = num_min_epoch
        self.min_improve = min_improve
        self.loss_list = []
        self.loss_min = float('inf')
        self.cur_epoch = 0

    def add(self, loss):
        self.cur_epoch += 1
        if loss < self.loss_min - self.min_improve:
            self.loss_min = loss
            self.loss_list = [loss]
        else:
            self.loss_list.append(loss)

    def stop(self):
        # If the epoch number exceed the predefined number, and
        # loss doesn't have improvement at least min_improve in query_len iterations, then stop.
        if self.cur_epoch >= self.num_min_epoch and len(self.loss_list) >= self.patience:
            return True
        else:
            return False
