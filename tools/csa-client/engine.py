import math
import queue
import subprocess
import threading
import tomllib
import copy

class Engine:
    def __init__(self, config, verbose: bool):
        self.config = config
        self.verbose = verbose
        self.latest_info = None
        self.enable_info_logging = False

        self.my_turn = None

        self.process = subprocess.Popen(
            config["command"].split(),
            cwd=config["wd"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if config["stderr"] else subprocess.DEVNULL,
        )

        self.message_queue = queue.Queue()
        threading.Thread(target=self._message_reader).start()
        if config["stderr"]:
            threading.Thread(target=self._stderr_reader).start()

        self.send_message_callbacks = []
        self.read_message_callbacks = []

    def add_send_message_callback(self, callback):
        self.send_message_callbacks.append(callback)

    def add_read_message_callback(self, callback):
        self.read_message_callbacks.append(callback)

    def _message_reader(self):
        with self.process.stdout:
            for line in iter(self.process.stdout.readline, b""):
                message = line.decode("utf-8").rstrip("\r\n")
                self.message_queue.put(message)

                if self.verbose:
                    print("Engine>", message)

                for callback in self.read_message_callbacks:
                    callback(message)

                if message.startswith("info"):
                    info = self._parse_info(message)

                    if self.enable_info_logging:
                        # Retrieve the previous information
                        # if they are not provided this time.
                        if self.latest_info is not None:
                            for key in self.latest_info:
                                if key not in info:
                                    info[key] = self.latest_info[key]

                        self.latest_info = info

                if message.startswith("bestmove"):
                    self.enable_info_logging = False

    def _stderr_reader(self):
        with self.process.stderr:
            for line in iter(self.process.stderr.readline, b""):
                message = line.decode("utf-8").rstrip("\r\n")
                print("Engine (stderr) >", message)

    def _parse_info(self, message):
        splitted = message.split(" ")

        info = dict()

        index = 0
        while index < len(splitted):
            if splitted[index] == "nps":
                info["nps"] = int(splitted[index + 1])
                index += 1

            elif splitted[index] == "score":
                if splitted[index + 1] == "mate":
                    ply = int(splitted[index + 2])
                    info["draw_rate"] = 0
                    info["win_rate"] = 1 if ply > 0 else 0
                elif splitted[index + 1] == "cp":
                    info["draw_rate"] = 0
                    score = int(splitted[index + 2])
                    info["win_rate"] = 1 / (1 + math.exp(-score / 600))
                elif splitted[index + 1] == "windraw":
                    info["win_rate"] = float(splitted[index + 2])
                    info["draw_rate"] = float(splitted[index + 3])

                index += 2

            elif splitted[index] == "pv":
                info["pv"] = splitted[index + 1 :]
                break

            index += 1

        return info

    def _send_message(self, message):
        if self.verbose:
            print("Engine<", message)

        for callback in self.send_message_callbacks:
            callback(message)

        message = (message + "\n").encode("utf-8")
        self.process.stdin.write(message)
        self.process.stdin.flush()

    def readline(self):
        message = self.message_queue.get()
        return message

    def usi(self):
        self._send_message("usi")

        while True:
            line = self.readline()

            if line == "usiok":
                break

    def isready(self):
        for k, v in self.config["usioptions"].items():
            if isinstance(v, bool):
                v = "true" if v else "false"
            self._send_message(f"setoption name {k} value {v}")

        self._send_message("isready")

        while True:
            line = self.readline()

            if line == "readyok":
                break

    def think_next_move_blocking(
        self, sfen: str, my_turn: int, b_time, w_time, b_inc, w_inc, byoyomi=None
    ):
        self._my_turn = my_turn
        self._send_message(f"position sfen {sfen}")

        self.latest_info = None
        self.enable_info_logging = True

        if byoyomi is not None:
            self._send_message(f"go btime {b_time} wtime {w_time} byoyomi {byoyomi}")
        else:
            self._send_message(
                f"go btime {b_time} wtime {w_time} binc {b_inc} winc {w_inc}"
            )

        while True:
            message = self.readline()
            if message.startswith("bestmove"):
                return message.split(" ")[1]

    def get_latest_info(self):
        return copy.copy(self.latest_info)

    def quit(self):
        self._send_message("quit")


if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    engine = Engine(config["engine"], True)

    engine.usi()
    engine.isready()
    engine.quit()
