import copy
import socket
import tomllib
import threading
import time
import math
import sys
import os

import flask
import flask_socketio

import nshogi

from engine import Engine


class CSAClient:
    def __init__(
        self,
        target: str,
        port: int,
        username: str,
        password: str,
        config,
        verbose: bool = True,
    ):
        self.config = config
        self._verbose = verbose

        self.engine = None

        self.sending_lock = threading.Lock()

        self._buffer = b""

        self.game_info = {}

        self.time_info = {
            # milli second for all entries.
            "unit": 1000,
            "total": 0,
            "byoyomi": 0,
            "increment": None,
        }

        self.remaining_time = [0, 0]
        self.evaluation_history = [None for _ in range(1024)]

        self._state: None | nshogi.State = None

        # Prepare viewer.
        self.viewer = None
        self.socketio = None

        if config["viewer"]["enabled"]:
            self.viewer = flask.Flask(__name__, template_folder="viewer")
            self.viewer.static_folder = self.viewer.root_path + "/viewer/static"
            self.socketio = flask_socketio.SocketIO(self.viewer)

            @self.viewer.route("/")
            def viewer_index():
                return flask.render_template("index.html")

            def run_viewer():
                self.socketio.run(self.viewer,
                      host=self.config["viewer"]["host"], port=self.config["viewer"]["port"])

            self.viewer_thread = threading.Thread(target=run_viewer)
            self.viewer_thread.start()

        while True:
            try:
                # Prepare engine.
                self.engine = Engine(config["engine"], True)

                if config["viewer"]["enabled"]:
                    self.engine.add_read_message_callback(self.callback_read_message)
                    self.engine.add_send_message_callback(self.callback_send_message)

                self.engine.usi()
                self.engine.isready()

                # Start communication.
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.connect((target, port))

                self._login(username, password)

                self.client_connected = True
                self.keepalive_thread = None
                self._start_keepalive_thread()

                self._wait_next_match()

                self.remaining_time = [self.time_info["total"] for _ in range(2)]

                self._agree()

                self.evaluation_history = [None for _ in range(1024)]

                if self.viewer is not None:
                    gi = self.game_info.copy()
                    if "Your_Turn" in gi:
                        gi["Your_Turn"] = "BLACK" if gi["Your_Turn"] == nshogi.Color.BLACK else "WHITE"
                    self.socketio.emit("update", {
                        "engine": self.config["engine"],
                        "timeinfo": self.time_info,
                        "time": self.remaining_time,
                        "gameinfo": gi,
                    })

                self._start_match()
                self._logout()
                self.client_connected = False
                self.keepalive_thread.join()

                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            except socket.error:
                pass
            except IOError:
                pass
            finally:
                self.socket.close()

            self.engine.quit()

            if not config["client"]["loop"]:
                break

    def _login(self, username, password):
        self._send_message(f"LOGIN {username} {password}")
        confirm_message = f"LOGIN:{username} OK"

        res = self._read_message()
        if res != confirm_message:
            raise RuntimeError("failed to login.")

    def _start_keepalive_thread(self):
        def send_keepalive():
            while self.client_connected:
                self._send_message("")
                time.sleep(self.config["server"]["keepalive"])

        self.keepalive_thread = threading.Thread(target=send_keepalive)
        self.keepalive_thread.start()

    def _logout(self):
        self._send_message("LOGOUT")

    def _wait_next_match(self):
        while True:
            message = self._read_message()
            if message != "BEGIN Game_Summary":
                continue

            self._read_game_summary()
            break

    def _read_game_summary(self):
        while True:
            message = self._read_message()

            if message == "END Game_Summary":
                break

            if message == "BEGIN Time":
                self._read_time()

            if message == "BEGIN Position":
                self._read_position()

            if ":" in message:
                key, value = message.split(":")

                if key == "Your_Turn":
                    if value == "+":
                        self.game_info["Your_Turn"] = nshogi.Color.BLACK
                    else:
                        self.game_info["Your_Turn"] = nshogi.Color.WHITE

                else:
                    self.game_info[key] = value

    def _read_time(self):
        while True:
            message = self._read_message()

            if message == "END Time":
                break

            if ":" in message:
                key, value = message.split(":")

                if key == "Time_Unit":
                    if value == "1sec":
                        self.time_info["unit"] = 1000
                    else:
                        raise RuntimeError("unknown time unit.")

                elif key == "Total_Time":
                    if not value.isnumeric():
                        raise RuntimeError("Total_Time entry must be an integer.")
                    self.time_info["total"] = int(value)

                elif key == "Byoyomi":
                    if not value.isnumeric():
                        raise RuntimeError("Byoyomi entry must be an integer.")
                    self.time_info["byoyomi"] = int(value)

                elif key == "Increment":
                    if not value.isnumeric():
                        raise RuntimeError("Increment entry must be an integer.")
                    self.time_info["increment"] = int(value)

        self.time_info["total"] *= self.time_info["unit"]
        self.time_info["byoyomi"] *= self.time_info["unit"]
        self.time_info["increment"] *= self.time_info["unit"]

    def _read_position(self):
        lines = []

        while True:
            message = self._read_message()

            if message == "END Position":
                break

            lines.append(message)

        self._state = nshogi.io.csa.make_state_from_csa("\n".join(lines))

    def _agree(self):
        if "Game_ID" in self.game_info:
            self._send_message("AGREE " + self.game_info["Game_ID"])
        else:
            self._send_message("AGREE")

        message = self._read_message()

        if message.startswith("REJECT"):
            raise RuntimeError("sent AGREE but got REJECT")

        if not message.startswith("START"):
            raise RuntimeError("unknown message: " + message)

    def _start_match(self):
        while True:
            if self.game_info["Your_Turn"] == self._state.side_to_move:
                if self.time_info["increment"] is None:
                    sfen_move = self.engine.think_next_move_blocking(
                        self._state.to_sfen(),
                        int(self.game_info["Your_Turn"]),
                        self.remaining_time[0],
                        self.remaining_time[1],
                        0,
                        0,
                        self.time_info["byoyomi"],
                    )
                else:
                    sfen_move = self.engine.think_next_move_blocking(
                        self._state.to_sfen(),
                        int(self.game_info["Your_Turn"]),
                        self.remaining_time[0],
                        self.remaining_time[1],
                        self.time_info["increment"],
                        self.time_info["increment"],
                        None,
                    )

                m = nshogi.io.sfen.move_from_sfen(self._state, sfen_move)
                csa_move = nshogi.io.csa.stringify(m, self._state.side_to_move)
                info = self.engine.get_latest_info()

                cp = 0
                black_win_rate, draw_rate = 0.5, 0.0
                pv = None
                if info is not None:
                    if "win_rate" in info:
                        win_rate, draw_rate_ = info["win_rate"], info["draw_rate"]

                        if self._state.side_to_move == nshogi.Color(1):
                            win_rate = 1.0 - win_rate - draw_rate_

                        cp = self.convert_win_rate_to_cp(win_rate, draw_rate_)

                    if "nshogiext" in info:
                        nshogiExt = info["nshogiext"]
                        if "black_win_rate" in nshogiExt and "draw_rate" in nshogiExt:
                            black_win_rate, draw_rate = nshogiExt["black_win_rate"], nshogiExt["draw_rate"]

                    if "pv" in info:
                        pv = info["pv"]

                self.evaluation_history[self._state.ply] = (black_win_rate, draw_rate)
                if not config["client"]["send_pv"]:
                    message = f"{csa_move}"
                else:
                    message = f"{csa_move},'* {cp}"
                    if pv is not None:
                        try:
                            stm = int(self._state.side_to_move)
                            csa_pv = []
                            s = self._state.clone()
                            for i, m in enumerate(pv[:-1]):
                                m = nshogi.io.sfen.move_from_sfen(s, m)
                                s.do_move(m)
                                csa_pv.append(nshogi.io.csa.stringify(m, nshogi.Color(stm)))
                                stm = 1 - stm
                            message += " " + " ".join(csa_pv[1:])
                        except RuntimeError:
                            pass

                self._send_message(message)

            game_finished = False
            while True:
                res = self._read_message()

                if res[0] == "%":
                    continue

                if res[0] == "#":
                    if res == "#WIN" or res == "#DRAW" or res == "#LOSE":
                        game_finished = True
                        break
                    else:
                        continue

                if "," in res:
                    csa_move, elapsed = res.split(",")
                    break

            if game_finished:
                if config["log"]["save_log"]:
                    if not os.path.exists(config["log"]["dir"]):
                        os.mkdir(config["log"]["dir"])
                    self._save_csa(config["log"]["dir"] + "/" + self.game_info["Game_ID"] + ".csa")
                break

            # Process time consumption.
            self.remaining_time[int(self._state.side_to_move)] += self.time_info[
                "increment"
            ]
            elapsed = int(elapsed[1:]) * self.time_info["unit"]
            self.remaining_time[int(self._state.side_to_move)] -= elapsed

            self._state.do_move(nshogi.io.csa.move_from_csa(self._state, csa_move))
            if self.socketio is not None:
                self.socketio.emit("update", {
                    "current_state": None if self._state is None else self._state.position.to_csa(),
                    "last_move_position": None if self._state is None else None if self._state.ply == 0 else int(self._state.last_move.to),
                })


    def _save_csa(self, filename):
        with open(filename, "w") as f:
            f.write("V2.2\n")
            f.write("N+" + self.game_info["Name+"] + "\n")
            f.write("N-" + self.game_info["Name-"] + "\n")
            f.write("'Max_Moves:" + str(self.game_info["Max_Moves"]) + "\n")
            f.write("'Increment:" + str(self.time_info["increment"]) + "\n")

            f.write(self._state.initial_position.to_csa())

            stm = int(self._state.initial_position.side_to_move)
            for m in self._state.history:
                csa_move = nshogi.io.csa.stringify(m, nshogi.Color(stm))
                f.write(csa_move + "\n")
                stm = 1 - stm

    def _send_message(self, message: str):
        with self.sending_lock:
            if self._verbose:
                print(f"CSA< {message}")

            if self.socketio is not None:
                self.socketio.emit("update", {
                    "csa_log": {
                        "type": "out",
                        "message": message
                    }
                })

            self.socket.sendall(f"{message}\n".encode())

    def _read_message(self, allow_broken=False):
        while True:
            if b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                s = line.decode().strip()

                if self._verbose:
                    print(f"CSA> {s}")

                if self.socketio is not None:
                    gi = self.game_info.copy()
                    if "Your_Turn" in gi:
                        gi["Your_Turn"] = int(gi["Your_Turn"])
                    self.socketio.emit("update", {
                        "csa_log": {
                            "type": "in",
                            "message": s
                        },
                        "current_state": None if self._state is None else self._state.position.to_csa(),
                        "last_move_position": None if self._state is None else None if self._state.ply == 0 else int(self._state.last_move.to),
                        "game_info": gi,
                    })

                if s == "":
                    continue

                return s

            data = self.socket.recv(1024)

            if len(data) == 0:
                if allow_broken:
                    return None

                raise RuntimeError("socket connection has been broken.")

            self._buffer += data

    def convert_win_rate_to_cp(self, x, d):
        if d < 1:
            x = x / (1 - d)
        else:
            x = 0.5

        if x <= 0:
            return -93039

        if x >= 1:
            return 93039

        return int(-600.0 * math.log(1.0 / x - 1.0))

    def callback_send_message(self, message):
        self.socketio.emit("update", {
            "usi_log": {
                "type": "in",
                "message": message
            },
        })

    def callback_read_message(self, message):
        remaining_time = self.remaining_time.copy()
        split = message.split(" ")

        if "Your_Turn" in self.game_info and self._state is not None and "time" in message:
            if self.game_info["Your_Turn"] == self._state.side_to_move:
                remaining_time[int(self._state.side_to_move)] -= int(split[split.index("time") + 1])
                if self.time_info["increment"] is not None:
                    remaining_time[int(self._state.side_to_move)] += self.time_info["increment"]

        self.socketio.emit("update", {
            "usi_log": {
                "type": "out",
                "message": message,
                "my_turn": None if not "Your_Turn" in self.game_info else "BLACK" if self.game_info["Your_Turn"] == nshogi.Color.BLACK else "WHITE"
            },
            "current_state": None if self._state is None else self._state.position.to_csa(),
            "last_move_position": None if self._state is None else None if self._state.ply == 0 else int(self._state.last_move.to),
            "remaining_time": remaining_time,
            "evaluation_history": None if self._state is None else self.evaluation_history[:self._state.ply]
        })

        if "nshogiext" in message:
            if "black_win_rate" in message and "draw_rate" in message and "white_win_rate" in message:
                self.socketio.emit("update", {
                    "nshogiext": {
                        "black_win_rate": split[split.index("black_win_rate") + 1],
                        "white_win_rate": split[split.index("white_win_rate") + 1],
                        "draw_rate": split[split.index("draw_rate") + 1],
                    },
                })

        if "pv" in message:
            clone_state = self._state.clone()

            pv_csa = []
            contains_invalid_move = False
            for m in split[split.index("pv") + 1:]:
                try:
                    m = nshogi.io.sfen.move_from_sfen(clone_state, m)
                    pv_csa.append(nshogi.io.csa.stringify(m, clone_state.side_to_move))
                    clone_state.do_move(m)
                except RuntimeError:
                    contains_invalid_move = True
                    break

            if not contains_invalid_move:
                self.socketio.emit("update", {
                    "pv": pv_csa
                })

if __name__ == "__main__":
    config_file = "config.toml" if len(sys.argv) == 1 else sys.argv[1]

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    try:
        CSAClient(
            config["server"]["host"],
            config["server"]["port"],
            config["client"]["id"],
            config["client"]["password"],
            config,
        )
    except RuntimeError as e:
        print(e)
