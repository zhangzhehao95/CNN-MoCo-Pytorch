import sys


# Modify the print() function to save the print information to a log file, while also displaying on the terminal
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")  # Opens a file for appending, creates the file if it does not exist

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
